from torch.utils.data import Dataset, Subset
import numpy as np
import torch
from collections import defaultdict
from src.utils.backdoor_utils import BackdoorDataset
from src.utils.backdoor_sem import SemanticBackdoorDataset

class NonIIDGenerator:
    def __init__(self, dataset, num_clients, dirichlet_alpha=0.5):
        """
        Initialize Non-IID data distributor
        Args:
            dataset: Original dataset
            num_clients: Number of clients
            dirichlet_alpha: Concentration parameter for Dirichlet distribution
                           Lower alpha = more non-IID
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = dirichlet_alpha
        self.num_classes = 10  # CIFAR-10 has 10 classes
        self.class_distribution = None

    def _get_label_distribution(self):
        """Calculate the distribution of labels across all data"""
        labels = []
        for _, label in self.dataset:
            labels.append(label)
        return np.array(labels)

    def _partition_by_class(self):
        """Group indices by class"""
        label_distribution = self._get_label_distribution()
        class_indices = defaultdict(list)
        
        for idx, label in enumerate(label_distribution):
            class_indices[label].append(idx)
            
        return class_indices

    def generate_non_iid_data(self):
        """
        Generate non-IID data distribution using Dirichlet distribution
        Returns:
            List of datasets for each client
        """
        class_indices = self._partition_by_class()
        client_data_indices = [[] for _ in range(self.num_clients)]
        
        # Generate Dirichlet distribution for each class
        for class_id in range(self.num_classes):
            class_size = len(class_indices[class_id])
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(
                np.repeat(self.alpha, self.num_clients)
            )
            
            # Calculate number of samples per client for this class
            proportions = np.array(
                [p * class_size for p in proportions]
            ).astype(int)
            
            # Adjust for rounding errors
            proportions[-1] = class_size - proportions[:-1].sum()
            
            # Distribute indices to clients
            class_idx_start = 0
            for client_id, num_samples in enumerate(proportions):
                client_data_indices[client_id].extend(
                    class_indices[class_id][
                        class_idx_start:class_idx_start + num_samples
                    ]
                )
                class_idx_start += num_samples

        # Create client datasets
        client_datasets = []
        for indices in client_data_indices:
            client_datasets.append(Subset(self.dataset, indices))

        # Store class distribution for visualization
        self.class_distribution = self._calculate_class_distribution(
            client_data_indices
        )
        
        return client_datasets

    def _calculate_class_distribution(self, client_data_indices):
        """Calculate the distribution of classes for each client"""
        distribution = np.zeros((self.num_clients, self.num_classes))
        labels = self._get_label_distribution()
        
        for client_id, indices in enumerate(client_data_indices):
            client_labels = labels[indices]
            for class_id in range(self.num_classes):
                distribution[client_id][class_id] = \
                    np.sum(client_labels == class_id) / len(indices)
                
        return distribution

    def visualize_distribution(self):
        """Visualize the class distribution across clients"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.class_distribution,
            cmap='YlOrRd',
            xticklabels=[f'Class {i}' for i in range(self.num_classes)],
            yticklabels=[f'Client {i}' for i in range(self.num_clients)]
        )
        plt.title('Class Distribution Across Clients')
        plt.xlabel('Classes')
        plt.ylabel('Clients')
        plt.savefig('./output/non_iid_distribution.png')
        plt.close()

def create_non_iid_backdoor_data(dataset, num_clients, attack_type, 
                                target_label, dirichlet_alpha=0.5):
    """
    Create non-IID data distribution with backdoor attacks
    Args:
        dataset: Original dataset
        num_clients: Number of clients
        attack_type: "1" for trigger-based, "2" for semantic
        target_label: Target class for backdoor
        dirichlet_alpha: Concentration parameter for Dirichlet distribution
    """
    # Generate non-IID distribution
    non_iid_gen = NonIIDGenerator(dataset, num_clients, dirichlet_alpha)
    client_datasets = non_iid_gen.generate_non_iid_data()
    
    # Visualize the distribution
    non_iid_gen.visualize_distribution()
    
    # Apply backdoor to malicious clients (20% of clients)
    num_malicious = num_clients // 5
    trigger_pattern = None
    
    if attack_type == "1":
        # Trigger-based backdoor
        trigger_pattern = torch.linspace(0, 1, 5).view(-1, 1).repeat(1, 5)
        trigger_pattern = trigger_pattern.unsqueeze(0).repeat(3, 1, 1)
        
        for i in range(num_malicious):
            client_datasets[i] = BackdoorDataset(
                client_datasets[i],
                trigger_pattern,
                target_label,
                poison_ratio=0.5
            )
    else:
        # Semantic backdoor
        for i in range(num_malicious):
            client_datasets[i] = SemanticBackdoorDataset(
                client_datasets[i],
                target_label,
                feature_type='brightness'
            )
    
    return client_datasets, trigger_pattern