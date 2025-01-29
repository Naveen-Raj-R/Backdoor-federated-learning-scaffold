from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch

class NonIIDMonitor:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.client_predictions = defaultdict(list)
        self.client_labels = defaultdict(list)
        self.client_distributions = {}
        self.global_distribution = None
        
    def update_client_stats(self, client_id, predictions, true_labels):
        """Update statistics for a specific client"""
        self.client_predictions[client_id].extend(predictions.cpu().numpy())
        self.client_labels[client_id].extend(true_labels.cpu().numpy())
        
    def compute_js_divergence(self, p, q):
        """Compute Jensen-Shannon divergence between two distributions"""
        p = np.array(p)
        q = np.array(q)
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log(p/m + 1e-10)) + 0.5 * np.sum(q * np.log(q/m + 1e-10))
        
    def calculate_label_skew(self):
        """Calculate label distribution skew across clients"""
        skew_metrics = {}
        
        # Calculate distribution for each client
        for client_id in self.client_labels:
            labels = np.array(self.client_labels[client_id])
            dist = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                dist[i] = np.sum(labels == i) / len(labels)
            self.client_distributions[client_id] = dist
            
        # Calculate global distribution
        all_labels = np.concatenate(list(self.client_labels.values()))
        self.global_distribution = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            self.global_distribution[i] = np.sum(all_labels == i) / len(all_labels)
            
        # Calculate JS divergence for each client
        for client_id in self.client_distributions:
            skew_metrics[client_id] = self.compute_js_divergence(
                self.client_distributions[client_id],
                self.global_distribution
            )
            
        return skew_metrics
        
    def plot_distribution_skew(self, save_path='./output/label_skew.png'):
        """Visualize label distribution skew"""
        skew_metrics = self.calculate_label_skew()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(skew_metrics)), list(skew_metrics.values()))
        plt.xlabel('Client ID')
        plt.ylabel('Jensen-Shannon Divergence')
        plt.title('Label Distribution Skew per Client')
        plt.savefig(save_path)
        plt.close()
        
    def plot_client_confusion_matrices(self, save_path='./output/confusion_matrices.png'):
        """Plot confusion matrices for each client"""
        num_clients = len(self.client_predictions)
        fig, axes = plt.subplots(2, (num_clients + 1) // 2, figsize=(15, 8))
        axes = axes.ravel()
        
        for idx, client_id in enumerate(self.client_predictions):
            cm = confusion_matrix(
                self.client_labels[client_id],
                self.client_predictions[client_id]
            )
            sns.heatmap(cm, ax=axes[idx], cmap='Blues', fmt='d')
            axes[idx].set_title(f'Client {client_id}')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def calculate_local_global_accuracy_gap(self):
        """Calculate accuracy gap between local and global performance"""
        gaps = {}
        global_preds = np.concatenate(list(self.client_predictions.values()))
        global_labels = np.concatenate(list(self.client_labels.values()))
        global_acc = np.mean(global_preds == global_labels)
        
        for client_id in self.client_predictions:
            local_acc = np.mean(
                np.array(self.client_predictions[client_id]) == 
                np.array(self.client_labels[client_id])
            )
            gaps[client_id] = abs(local_acc - global_acc)
            
        return gaps

# Evaluation metrics for non-IID scenarios
def compute_non_iid_metrics(model, clients, device):
    """Compute various non-IID specific metrics"""
    monitor = NonIIDMonitor()
    class_accuracies = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for client_id, dataset in enumerate(clients):
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
            client_preds = []
            client_labels = []
            
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions and labels
                client_preds.extend(predicted)
                client_labels.extend(labels)
                
                # Calculate per-class accuracy
                for label in range(10):
                    mask = labels == label
                    if mask.any():
                        acc = (predicted[mask] == labels[mask]).float().mean().item()
                        class_accuracies[label].append(acc)
            
            # Update monitor with client results
            monitor.update_client_stats(
                client_id, 
                torch.tensor(client_preds), 
                torch.tensor(client_labels)
            )
    
    return monitor, class_accuracies