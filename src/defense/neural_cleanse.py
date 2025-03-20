import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class NeuralCleanse:
    def __init__(self, model, dataset, device, num_classes=10):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.num_classes = num_classes
        self.init_mask_size = (3, 5, 5)  # Initial trigger mask size
        self.mask_lr = 0.1
        self.pattern_lr = 0.1
        self.num_optim_steps = 1000
        self.attack_succ_threshold = 0.99
        self.anomaly_threshold = 2.0

    def detect(self):
        """
        Detect potential backdoors by reverse engineering triggers
        Returns True if backdoor detected, False otherwise
        """
        print("Starting Neural Cleanse detection...")
        l1_norms = []
        patterns = []
        masks = []

        # Try to reverse engineer trigger for each target label
        for target_label in range(self.num_classes):
            print(f"\nAnalyzing potential backdoor for class {target_label}")
            pattern, mask, l1_norm = self._reverse_engineer_trigger(target_label)
            
            l1_norms.append(l1_norm)
            patterns.append(pattern)
            masks.append(mask)

        # Detect anomalies in L1 norms using MAD
        l1_norms = np.array(l1_norms)
        median = np.median(l1_norms)
        mad = np.median(np.abs(l1_norms - median))
        anomaly_scores = np.abs(l1_norms - median) / (mad + 1e-8)
        
        # Check for anomalies
        self.potential_backdoors = []
        for i, score in enumerate(anomaly_scores):
            if score > self.anomaly_threshold:
                self.potential_backdoors.append({
                    'target_label': i,
                    'pattern': patterns[i],
                    'mask': masks[i],
                    'l1_norm': l1_norms[i],
                    'anomaly_score': score
                })
        
        return len(self.potential_backdoors) > 0

    def defend(self):
        """
        Apply defense by pruning neurons associated with detected backdoors
        Returns cleaned model
        """
        if not hasattr(self, 'potential_backdoors'):
            raise RuntimeError("Must run detect() before defend()")
            
        cleaned_model = self.model
        
        for backdoor in self.potential_backdoors:
            # Identify neurons activated by the trigger
            pattern = backdoor['pattern'].to(self.device)
            mask = backdoor['mask'].to(self.device)
            target_label = backdoor['target_label']
            
            # Get intermediate activations
            activation_map = self._get_neuron_activation_map(pattern * mask)
            
            # Find neurons with unusual activation patterns
            suspicious_neurons = self._identify_suspicious_neurons(activation_map)
            
            # Prune suspicious neurons
            cleaned_model = self._prune_neurons(cleaned_model, suspicious_neurons)
            
        return cleaned_model

    def _reverse_engineer_trigger(self, target_label):
        """
        Reverse engineer potential trigger for given target label
        """
        # Initialize trigger pattern and mask
        pattern = torch.rand(self.init_mask_size, requires_grad=True, device=self.device)
        mask = torch.rand(self.init_mask_size, requires_grad=True, device=self.device)
        
        # Setup optimizers
        pattern_optimizer = optim.Adam([pattern], lr=self.pattern_lr)
        mask_optimizer = optim.Adam([mask], lr=self.mask_lr)
        
        # Training loop
        for _ in tqdm(range(self.num_optim_steps), desc="Reverse engineering trigger"):
            total_loss = 0
            correct = 0
            total = 0
            
            # Check if dataset is an iterable or has an appropriate method
            if hasattr(self.dataset, 'dataloader'):
                # If using a custom dataset with a dataloader method
                dataloader = self.dataset.dataloader()
            elif hasattr(self.dataset, '__iter__'):
                # If it's already an iterable (like a DataLoader)
                dataloader = self.dataset
            else:
                raise ValueError("Dataset must be an iterable or have a dataloader method")
            
            for batch in dataloader:
                # Handle different possible batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, labels = batch
                    else:
                        raise ValueError("Batch must contain inputs and labels")
                elif isinstance(batch, dict):
                    inputs = batch.get('inputs') or batch.get('input')
                    labels = batch.get('labels') or batch.get('label')
                    if inputs is None or labels is None:
                        raise ValueError("Could not extract inputs and labels from batch")
                else:
                    raise ValueError("Unsupported batch format")
                
                # Move to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.shape[0]
                
                # Apply trigger
                trigger_mask = torch.sigmoid(mask)
                poisoned_inputs = inputs * (1 - trigger_mask) + pattern * trigger_mask
                
                # Forward pass
                outputs = self.model(poisoned_inputs)
                pred = outputs.argmax(dim=1)
                
                # Calculate loss
                classification_loss = nn.CrossEntropyLoss()(outputs, 
                    torch.full((batch_size,), target_label, device=self.device))
                mask_norm = torch.norm(trigger_mask, p=1)
                loss = classification_loss + 0.01 * mask_norm
                
                # Backward pass
                pattern_optimizer.zero_grad()
                mask_optimizer.zero_grad()
                loss.backward()
                pattern_optimizer.step()
                mask_optimizer.step()
                
                total_loss += loss.item()
                correct += (pred == target_label).sum().item()
                total += batch_size
                
            attack_success_rate = correct / total
            if attack_success_rate > self.attack_succ_threshold:
                break
                
        return pattern.detach(), torch.sigmoid(mask).detach(), mask_norm.item()

        def _get_neuron_activation_map(self, trigger_input):
            """
            Get activation map for all neurons given trigger input
            """
            activations = {}
            hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output.detach()
                return hook
                
            # Register hooks for all layers
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
                    
            # Forward pass with trigger
            self.model(trigger_input.unsqueeze(0))
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
            return activations

    def _identify_suspicious_neurons(self, activation_map, threshold=0.95):
        """
        Identify neurons with unusual activation patterns
        """
        suspicious_neurons = {}
        
        for layer_name, activations in activation_map.items():
            # Calculate statistics
            mean_act = activations.mean().item()
            std_act = activations.std().item()
            
            # Find neurons with unusually high activation
            threshold_value = mean_act + 2 * std_act
            suspicious_mask = activations > threshold_value
            
            if suspicious_mask.any():
                suspicious_neurons[layer_name] = suspicious_mask
                
        return suspicious_neurons

    def _prune_neurons(self, model, suspicious_neurons):
        """
        Prune identified suspicious neurons
        """
        for name, module in model.named_modules():
            if name in suspicious_neurons:
                if isinstance(module, nn.Conv2d):
                    # Zero out suspicious channels in conv layers
                    mask = ~suspicious_neurons[name]
                    module.weight.data[:, mask] = 0
                    if module.bias is not None:
                        module.bias.data[mask] = 0
                elif isinstance(module, nn.Linear):
                    # Zero out suspicious neurons in linear layers
                    mask = ~suspicious_neurons[name]
                    module.weight.data[mask] = 0
                    if module.bias is not None:
                        module.bias.data[mask] = 0
                        
        return model