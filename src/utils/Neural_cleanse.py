import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os

class ImprovedNeuralCleanse:
    def __init__(self, model, num_classes, input_shape, device, lambda_val=0.01):
        self.model = model
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.device = device
        self.lambda_val = lambda_val
        self.optimized_masks = {}
        self.optimized_patterns = {}
        self.l1_norms = {}
        os.makedirs('./output/triggers', exist_ok=True)
        
    def reverse_engineer_triggers(self, max_iter=1000, lr=0.01, early_stop_threshold=0.99, sample_size=100):
        print("Starting trigger reverse engineering process...")
        self.model.eval()
        
        self.optimized_masks = {}
        self.optimized_patterns = {}
        self.l1_norms = {}
        
        for target_class in range(self.num_classes):
            print(f"\nReverse engineering trigger for class {target_class}...")
            
            mask = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            pattern = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            
            optimizer = optim.Adam([mask, pattern], lr=lr)
            
            pbar = tqdm(range(max_iter), desc=f"Class {target_class} optimization")
            best_loss = float('inf')
            best_mask = None
            best_pattern = None
            
            for i in pbar:
                inputs = torch.rand(sample_size, *self.input_shape, device=self.device)
                
                triggered_inputs = self.apply_trigger(inputs, mask, pattern)
                
                outputs = self.model(triggered_inputs)
                
                classification_loss = self.misclassification_loss(outputs, target_class)
                l1_mask = torch.sum(torch.abs(mask))
                
                loss = classification_loss + self.lambda_val * l1_mask
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    mask.clamp_(0, 1)
                    pattern.clamp_(0, 1)
                
                success_rate = (torch.argmax(outputs, dim=1) == target_class).float().mean().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'success_rate': f"{success_rate:.2f}",
                    'l1_norm': l1_mask.item()
                })
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_mask = mask.detach().clone()
                    best_pattern = pattern.detach().clone()
                
                if success_rate > early_stop_threshold:
                    print(f"Early stopping at iteration {i}: Success rate threshold reached")
                    break
            
            self.optimized_masks[target_class] = best_mask
            self.optimized_patterns[target_class] = best_pattern
            self.l1_norms[target_class] = torch.sum(torch.abs(best_mask)).item()
            
            print(f"Class {target_class}: L1 norm = {self.l1_norms[target_class]:.4f}")
    
    def apply_trigger(self, inputs, mask, pattern):
        return (1 - mask) * inputs + mask * pattern
    
    def misclassification_loss(self, outputs, target_class):
        return nn.CrossEntropyLoss()(outputs, torch.full((outputs.size(0),), target_class, device=self.device))
    
    def detect_backdoors(self, threshold_factor=1.5):
        if not self.l1_norms:
            raise ValueError("Reverse engineer triggers first using reverse_engineer_triggers()")
        
        l1_norms_values = np.array(list(self.l1_norms.values()))
        
        median = np.median(l1_norms_values)
        mad = np.median(np.abs(l1_norms_values - median))
        
        anomaly_indices = np.abs(l1_norms_values - median) / (mad + 1e-8)
        
        threshold = threshold_factor * 1.4826
        
        suspected_classes = []
        for class_idx, anomaly_idx in enumerate(anomaly_indices):
            if anomaly_idx > threshold:
                suspected_classes.append(class_idx)
        
        backdoor_detected = len(suspected_classes) > 0
        
        if backdoor_detected:
            print(f"\nBackdoor detected! Suspected target classes: {suspected_classes}")
            print("\nL1 norms of trigger masks for each class:")
            for class_idx in range(self.num_classes):
                print(f"Class {class_idx}: {self.l1_norms[class_idx]:.4f} " + 
                      f"(Anomaly index: {anomaly_indices[class_idx]:.4f})" +
                      f" {'*' if class_idx in suspected_classes else ''}")
        else:
            print("\nNo backdoor detected.")
        
        self.visualize_l1_norms(suspected_classes)
        
        if backdoor_detected:
            self.visualize_triggers(suspected_classes)
            self.plot_triggered_examples(suspected_classes)
        
        return backdoor_detected, suspected_classes, self.l1_norms, self.optimized_masks, self.optimized_patterns
    
    def visualize_l1_norms(self, suspected_classes, save_path='./output/l1_norms.png'):
        plt.figure(figsize=(10, 6))
        classes = list(range(self.num_classes))
        norms = [self.l1_norms[c] for c in classes]
        
        bars = plt.bar(classes, norms, color=['red' if c in suspected_classes else 'blue' for c in classes])
        
        plt.xlabel('Class')
        plt.ylabel('L1 Norm of Trigger Mask')
        plt.title('L1 Norms of Reverse-Engineered Triggers')
        plt.xticks(classes)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        median = np.median(norms)
        plt.axhline(y=median, color='k', linestyle='--', alpha=0.5, label=f'Median: {median:.4f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def visualize_triggers(self, suspected_classes, save_path='./output/triggers/'):
        os.makedirs(save_path, exist_ok=True)
        
        for class_idx in suspected_classes:
            mask = self.optimized_masks[class_idx].cpu().numpy()
            pattern = self.optimized_patterns[class_idx].cpu().numpy()
            
            if len(mask.shape) == 3 and mask.shape[0] == 3:
                mask = np.transpose(mask, (1, 2, 0))
                pattern = np.transpose(pattern, (1, 2, 0))
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(mask)
                plt.title(f'Mask for Class {class_idx}')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(pattern)
                plt.title(f'Pattern for Class {class_idx}')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                blank = np.zeros_like(pattern)
                triggered = (1 - mask) * blank + mask * pattern
                plt.imshow(triggered)
                plt.title('Trigger Applied')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}trigger_class_{class_idx}.png')
                plt.close()
    
    def plot_triggered_examples(self, suspected_classes, num_samples=5, save_path='./output/triggers/'):
        os.makedirs(save_path, exist_ok=True)
        
        with torch.no_grad():
            for class_idx in suspected_classes:
                mask = self.optimized_masks[class_idx]
                pattern = self.optimized_patterns[class_idx]
                
                random_inputs = torch.rand(num_samples, *self.input_shape, device=self.device)
                triggered_inputs = self.apply_trigger(random_inputs, mask, pattern)
                
                outputs = self.model(triggered_inputs)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                plt.figure(figsize=(15, 3*num_samples))
                for i in range(num_samples):
                    original = random_inputs[i].cpu().numpy()
                    triggered = triggered_inputs[i].cpu().numpy()
                    
                    if len(original.shape) == 3 and original.shape[0] == 3:
                        original = np.transpose(original, (1, 2, 0))
                        triggered = np.transpose(triggered, (1, 2, 0))
                    
                    plt.subplot(num_samples, 2, 2*i+1)
                    plt.imshow(original)
                    plt.title(f'Original Sample {i+1}')
                    plt.axis('off')
                    
                    plt.subplot(num_samples, 2, 2*i+2)
                    plt.imshow(triggered)
                    plt.title(f'Triggered → Class {predictions[i]}')
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}samples_class_{class_idx}.png')
                plt.close()
    
    def repair_model(self, suspected_classes, optimized_masks=None, optimized_patterns=None):
        if optimized_masks is None:
            optimized_masks = self.optimized_masks
        if optimized_patterns is None:
            optimized_patterns = self.optimized_patterns
            
        print("\nRepairing model for suspected backdoor classes:", suspected_classes)
        
        repaired_model = copy.deepcopy(self.model)
        
        repaired_model.train()
        
        optimizer = optim.Adam(repaired_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 5
        batch_size = 64
        
        def get_random_batch(batch_size):
            inputs = torch.rand(batch_size, *self.input_shape, device=self.device)
            labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            return inputs, labels
        
        accuracy_history = []
        loss_history = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for _ in range(10):
                inputs, labels = get_random_batch(batch_size)
                
                adv_inputs = inputs.clone()
                adv_labels = labels.clone()
                
                for class_idx in suspected_classes:
                    mask = optimized_masks[class_idx]
                    pattern = optimized_patterns[class_idx]
                    
                    trigger_idx = torch.rand(batch_size) < 0.2
                    if trigger_idx.sum() > 0:
                        adv_inputs[trigger_idx] = self.apply_trigger(
                            adv_inputs[trigger_idx], mask, pattern)
                        
                combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
                combined_labels = torch.cat([labels, adv_labels], dim=0)
                
                optimizer.zero_grad()
                outputs = repaired_model(combined_inputs)
                loss = criterion(outputs, combined_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss/10
            loss_history.append(epoch_loss)
            
            with torch.no_grad():
                test_inputs, test_labels = get_random_batch(100)
                outputs = repaired_model(test_inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == test_labels).sum().item()
                accuracy = 100 * correct / 100
                accuracy_history.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        self.plot_repair_history(accuracy_history, loss_history)
        
        repaired_model.eval()
        
        return repaired_model
    
    def plot_repair_history(self, accuracy_history, loss_history, save_path='./output/repair_history.png'):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_history, 'r-')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def evaluate_repair(self, repaired_model, test_dataset, trigger_pattern, target_label):
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = repaired_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        clean_acc = 100 * correct / total
        print(f"\nRepaired Model - Clean Test Accuracy: {clean_acc:.2f}%")
        
        if trigger_pattern is not None:
            correct = 0
            triggered_correct = 0
            total = 0
            
            class_distribution = {i: 0 for i in range(self.num_classes)}
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Ensure trigger_pattern matches data tensor dimensions
                    if trigger_pattern.dim() != data.dim():
                        # If trigger_pattern is 1D or 2D, expand it to match data dimensions
                        while trigger_pattern.dim() < data.dim():
                            trigger_pattern = trigger_pattern.unsqueeze(0)
                    
                    # Broadcast trigger_pattern to match data tensor
                    if trigger_pattern.shape != data.shape:
                        trigger_pattern = trigger_pattern.expand_as(data)
                    
                    # Apply trigger
                    triggered_data = (1 - trigger_pattern) * data + trigger_pattern
                    
                    outputs = repaired_model(triggered_data)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    for i in range(self.num_classes):
                        class_distribution[i] += (predicted == i).sum().item()
                    
                    total += targets.size(0)
                    triggered_correct += (predicted == target_label).sum().item()
            
            backdoor_asr = 100 * triggered_correct / total
            print(f"Repaired Model - Backdoor Attack Success Rate: {backdoor_asr:.2f}%")
            
            self.plot_class_distribution(class_distribution, total)
            
            return clean_acc, backdoor_asr
        return clean_acc, None
    
    def plot_class_distribution(self, class_distribution, total, save_path='./output/post_repair_distribution.png'):
        classes = list(class_distribution.keys())
        values = [class_distribution[c]/total*100 for c in classes]
        
        plt.figure(figsize=(10, 6))
        plt.bar(classes, values, color='teal')
        plt.xlabel('Class')
        plt.ylabel('Percentage (%)')
        plt.title('Post-Repair Classification Distribution on Triggered Inputs')
        plt.xticks(classes)
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()