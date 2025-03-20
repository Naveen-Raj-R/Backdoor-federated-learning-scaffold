import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class FinePruning:
    def __init__(self, model, clean_dataset, device, num_classes=10, prune_ratio=0.1):
        """
        Initialize the Fine-Pruning defense.
        
        Args:
            model: The model to be defended
            clean_dataset: A clean dataset used for fine-tuning
            device: The device (CPU/GPU) for computation
            num_classes: Number of classes in the dataset
            prune_ratio: Percentage of neurons to prune (default: 10%)
        """
        self.model = model
        self.clean_dataset = clean_dataset
        self.device = device
        self.num_classes = num_classes
        self.prune_ratio = prune_ratio
        self.activation_maps = {}
        self.layer_names = []
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture activations from convolutional layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.layer_names.append(name)
                hook = module.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(hook)
    
    def _get_activation_hook(self, name):
        """Create a hook function that stores activations for a given layer"""
        def hook(module, input, output):
            self.activation_maps[name] = output.detach()
        return hook

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_activations(self, dataloader=None):
        """
        Collect activations for all registered layers using a clean dataset
        
        Args:
            dataloader: DataLoader for clean dataset. If None, one will be created.
        """
        if dataloader is None:
            # Create a dataloader with a subset of clean data
            subset_size = min(1000, len(self.clean_dataset))
            subset_indices = torch.randperm(len(self.clean_dataset))[:subset_size]
            subset = torch.utils.data.Subset(self.clean_dataset, subset_indices)
            dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Register hooks
        self.register_hooks()
        
        # Dictionary to store the cumulative activations
        cumulative_activations = {name: None for name in self.layer_names}
        
        print("Collecting activations for neuron importance analysis...")
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader):
                inputs = inputs.to(self.device)
                self.model(inputs)
                
                # Update cumulative activations
                for name in self.layer_names:
                    activations = self.activation_maps[name]
                    # Average across batch and spatial dimensions, keeping channel dimension
                    mean_activations = activations.mean(dim=(0, 2, 3))
                    
                    if cumulative_activations[name] is None:
                        cumulative_activations[name] = mean_activations
                    else:
                        cumulative_activations[name] += mean_activations
        
        # Remove hooks
        self.remove_hooks()
        
        # Normalize cumulative activations
        for name in self.layer_names:
            cumulative_activations[name] /= len(dataloader)
        
        return cumulative_activations

    def identify_dormant_neurons(self, cumulative_activations):
        """
        Identify the dormant neurons in each layer based on activation values
        
        Args:
            cumulative_activations: Dictionary of cumulative activations for each layer
            
        Returns:
            dormant_indices: Dictionary of indices of dormant neurons for each layer
        """
        dormant_indices = {}
        
        for name, activations in cumulative_activations.items():
            # Get the number of neurons in this layer
            num_neurons = activations.shape[0]
            
            # Calculate number of neurons to prune
            k = int(num_neurons * self.prune_ratio)
            
            # Get indices of k neurons with lowest activations
            _, indices = torch.topk(activations, k, largest=False)
            dormant_indices[name] = indices.cpu().numpy()
            
            print(f"Layer {name}: {k}/{num_neurons} neurons identified as dormant")
        
        return dormant_indices

    def visualize_dormant_neurons(self, cumulative_activations, dormant_indices, top_n=5):
        """
        Visualize the activations and highlight dormant neurons
        
        Args:
            cumulative_activations: Dictionary of cumulative activations
            dormant_indices: Dictionary of indices of dormant neurons
            top_n: Number of layers to visualize (default: 5)
        """
        # Sort layers by number of channels
        sorted_layers = sorted([(name, act.shape[0]) for name, act in cumulative_activations.items()], 
                               key=lambda x: x[1], reverse=True)
        
        # Select top_n layers with the most channels
        layers_to_plot = [name for name, _ in sorted_layers[:top_n]]
        
        fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(10, 3*len(layers_to_plot)))
        if len(layers_to_plot) == 1:
            axes = [axes]
        
        for ax, layer_name in zip(axes, layers_to_plot):
            activations = cumulative_activations[layer_name].cpu().numpy()
            dormant = dormant_indices[layer_name]
            
            # Sort activations
            sorted_indices = np.argsort(activations)
            sorted_activations = activations[sorted_indices]
            
            # Mark dormant neurons
            is_dormant = np.isin(sorted_indices, dormant)
            
            # Plot activations
            ax.bar(range(len(activations)), sorted_activations, color='blue', alpha=0.7)
            
            # Highlight dormant neurons
            for i, (val, dormant_flag) in enumerate(zip(sorted_activations, is_dormant)):
                if dormant_flag:
                    ax.bar(i, val, color='red', alpha=0.7)
            
            ax.set_title(f"Layer: {layer_name}")
            ax.set_xlabel("Neuron Index (sorted by activation)")
            ax.set_ylabel("Average Activation")
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.7, label='Active Neurons'),
                Patch(facecolor='red', alpha=0.7, label='Dormant Neurons (pruned)')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('./output/dormant_neurons.png')
        plt.close()

    def create_pruning_masks(self, dormant_indices):
        """
        Create binary masks for each layer to zero out dormant neurons
        
        Args:
            dormant_indices: Dictionary of indices of dormant neurons
            
        Returns:
            masks: Dictionary of binary masks for each layer
        """
        masks = {}
        
        for name in self.layer_names:
            # Get the corresponding layer
            layer = dict(self.model.named_modules())[name]
            # Create a mask of ones with the same shape as the layer weights
            mask = torch.ones(layer.weight.shape[0]).to(self.device)
            # Set the mask values of dormant neurons to zero
            if name in dormant_indices:
                mask[dormant_indices[name]] = 0
            masks[name] = mask
            
        return masks

    def apply_pruning(self, masks):
        """
        Apply pruning to the model by zeroing out weights of dormant neurons
        
        Args:
            masks: Dictionary of binary masks for each layer
            
        Returns:
            pruned_model: Model with pruned neurons
        """
        # Create a deep copy of the model
        pruned_model = copy.deepcopy(self.model)
        
        # Apply masks to zero out weights of dormant neurons
        for name, mask in masks.items():
            layer = dict(pruned_model.named_modules())[name]
            # Expand mask to match weight dimensions and apply
            expanded_mask = mask.view(-1, 1, 1, 1)
            with torch.no_grad():
                layer.weight.data = layer.weight.data * expanded_mask
        
        return pruned_model

    def fine_tune(self, pruned_model, epochs=5, lr=0.001, batch_size=64):
        """
        Fine-tune the pruned model on clean data
        
        Args:
            pruned_model: Model with pruned neurons
            epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning
            batch_size: Batch size for fine-tuning
            
        Returns:
            fine_tuned_model: Fine-tuned model
        """
        # Create a data loader for fine-tuning
        subset_size = min(5000, len(self.clean_dataset))
        subset_indices = torch.randperm(len(self.clean_dataset))[:subset_size]
        subset = torch.utils.data.Subset(self.clean_dataset, subset_indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Fine-tuning loop
        pruned_model.train()
        print("Fine-tuning the pruned model...")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Print epoch results
            accuracy = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{epochs}: Loss = {running_loss/len(dataloader):.4f}, Accuracy = {accuracy:.2f}%")
        
        return pruned_model

    def evaluate_defense(self, model, test_dataset, trigger_pattern=None, target_label=None, attack_type="1"):
        """
        Evaluate the effectiveness of the defense
        
        Args:
            model: Model to evaluate
            test_dataset: Clean test dataset
            trigger_pattern: Trigger pattern for backdoor attacks
            target_label: Target label for backdoor attacks
            attack_type: Type of attack (1 for trigger-based, 2 for semantic)
            
        Returns:
            clean_acc: Accuracy on clean test data
            backdoor_acc: Success rate of backdoor attack
        """
        # Create test dataloader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Evaluate on clean data
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        clean_acc = 100.0 * correct / total
        
        # Evaluate on backdoored data
        if attack_type == "1" and trigger_pattern is not None and target_label is not None:
            from utils.backdoor_utils import apply_trigger_to_batch
            
            total_samples = 0
            successful_attacks = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    # Apply trigger to inputs
                    backdoored_inputs = apply_trigger_to_batch(inputs, trigger_pattern)
                    backdoored_inputs = backdoored_inputs.to(self.device)
                    
                    outputs = model(backdoored_inputs)
                    _, predicted = outputs.max(1)
                    
                    # Count successful attacks (predictions matching target_label)
                    successful_attacks += (predicted == target_label).sum().item()
                    total_samples += inputs.size(0)
            
            backdoor_acc = 100.0 * successful_attacks / total_samples
        
        elif attack_type == "2" and target_label is not None:
            from utils.backdoor_sem import get_semantic_backdoor_samples
            
            # Get semantic backdoor samples
            backdoored_inputs, _ = get_semantic_backdoor_samples(test_dataset, batch_size=100)
            backdoored_inputs = backdoored_inputs.to(self.device)
            
            with torch.no_grad():
                outputs = model(backdoored_inputs)
                _, predicted = outputs.max(1)
                
                # Count successful attacks (predictions matching target_label)
                successful_attacks = (predicted == target_label).sum().item()
                total_samples = backdoored_inputs.size(0)
            
            backdoor_acc = 100.0 * successful_attacks / total_samples
        
        else:
            backdoor_acc = None
        
        return clean_acc, backdoor_acc

    def defend(self, trigger_pattern=None, target_label=None, attack_type="1", fine_tuning_epochs=5):
        """
        Complete defense pipeline: identify dormant neurons, prune them, and fine-tune
        
        Args:
            trigger_pattern: Trigger pattern for backdoor attacks (for evaluation)
            target_label: Target label for backdoor attacks (for evaluation)
            attack_type: Type of attack (1 for trigger-based, 2 for semantic)
            fine_tuning_epochs: Number of epochs for fine-tuning
            
        Returns:
            defended_model: Model after defense
            results: Dictionary with defense results
        """
        # 1. Collect activations
        cumulative_activations = self.collect_activations()
        
        # 2. Identify dormant neurons
        dormant_indices = self.identify_dormant_neurons(cumulative_activations)
        
        # 3. Visualize dormant neurons
        self.visualize_dormant_neurons(cumulative_activations, dormant_indices)
        
        # 4. Create pruning masks
        masks = self.create_pruning_masks(dormant_indices)
        
        # 5. Apply pruning
        pruned_model = self.apply_pruning(masks)
        
        # 6. Fine-tune the pruned model
        print(f"\nFine-tuning pruned model for {fine_tuning_epochs} epochs...")
        defended_model = self.fine_tune(pruned_model, epochs=fine_tuning_epochs)
        
        # 7. Evaluate defense
        print("\nEvaluating Fine-Pruning defense...")
        clean_acc, backdoor_acc = self.evaluate_defense(
            defended_model, 
            self.clean_dataset, 
            trigger_pattern, 
            target_label,
            attack_type
        )
        
        # Original model evaluation for comparison
        orig_clean_acc, orig_backdoor_acc = self.evaluate_defense(
            self.model, 
            self.clean_dataset, 
            trigger_pattern, 
            target_label,
            attack_type
        )
        
        print("\nDefense Results:")
        print(f"Original Model - Clean Test Accuracy: {orig_clean_acc:.2f}%, Backdoor Success Rate: {orig_backdoor_acc:.2f}%")
        print(f"Defended Model - Clean Test Accuracy: {clean_acc:.2f}%, Backdoor Success Rate: {backdoor_acc:.2f}%")
        
        # Calculate defense effectiveness
        clean_acc_impact = clean_acc - orig_clean_acc
        backdoor_reduction = orig_backdoor_acc - backdoor_acc
        
        results = {
            'original_clean_acc': orig_clean_acc,
            'original_backdoor_acc': orig_backdoor_acc,
            'defended_clean_acc': clean_acc,
            'defended_backdoor_acc': backdoor_acc,
            'clean_acc_impact': clean_acc_impact,
            'backdoor_reduction': backdoor_reduction,
            'prune_ratio': self.prune_ratio
        }
        
        return defended_model, results