import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
import os

class ActivationClusteringDefender:
    """
    Implementation of Activation Clustering method for backdoor detection in neural networks.
    Based on the algorithm shown in the image, which analyzes the activation patterns in the
    last hidden layer to detect backdoored inputs.
    """
    
    def __init__(self, model, dataset, device, num_classes=10, batch_size=128):
        """
        Initialize the Activation Clustering defender.
        
        Args:
            model: The model to analyze
            dataset: Dataset containing potentially poisoned samples
            device: Device to run computations on (cuda/cpu)
            num_classes: Number of classes in the classification task
            batch_size: Batch size for processing data
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.activations = {i: [] for i in range(num_classes)}
        self.poison_indices = {i: [] for i in range(num_classes)}
        
    def collect_activations(self, layer_name=None):
        """
        Collect activations from the last hidden layer of the model for all samples.
        
        Args:
            layer_name: Optional name of the specific layer to extract activations from
        """
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # Dictionary to store sample indices for each class
        sample_indices = {i: [] for i in range(self.num_classes)}
        global_index = 0
        
        # Register hooks to get activations
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # If no specific layer is provided, use the penultimate layer
        if layer_name is None:
            # For ResNet18, the penultimate layer would be the one before the final fc layer
            # Access through the nested model attribute
            hook = self.model.model.layer4.register_forward_hook(get_activation('penultimate'))
            hooks.append(hook)
        else:
            # Register hook for the specified layer - handle nested model structure
            if hasattr(self.model, 'model'):
                # If the model is a wrapper, access through nested model
                layer_path = f"self.model.model.{layer_name}"
            else:
                # Direct access if it's not a wrapper
                layer_path = f"self.model.{layer_name}"
            
            layer = eval(layer_path)
            hook = layer.register_forward_hook(get_activation(layer_name))
            hooks.append(hook)
        
        print("Collecting activations from samples...")
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                # Get activations and organize by predicted class
                for i in range(len(data)):
                    pred_class = predicted[i].item()
                    
                    # Get the activation for this sample
                    if layer_name is None:
                        # For ResNet, reshape the output of the conv layer
                        act = activations['penultimate'][i].view(-1).cpu().numpy()
                    else:
                        act = activations[layer_name][i].view(-1).cpu().numpy()
                    
                    # Store activation by predicted class
                    self.activations[pred_class].append(act)
                    
                    # Store the original dataset index
                    sample_indices[pred_class].append(global_index)
                    global_index += 1
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx * self.batch_size} samples...")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert lists to numpy arrays for each class
        for i in range(self.num_classes):
            if self.activations[i]:
                self.activations[i] = np.vstack(self.activations[i])
                print(f"Class {i}: Collected {len(self.activations[i])} samples")
            else:
                print(f"Class {i}: No samples collected")
        
        return sample_indices
    
    def reduce_dimensions(self, method='pca', n_components=2):
        """
        Reduce dimensions of activations for each class.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'ica')
            n_components: Number of components to reduce to
            
        Returns:
            Dictionary with reduced activations for each class
        """
        reduced_activations = {}
        
        for class_idx, acts in self.activations.items():
            if len(acts) > n_components:  # Need more samples than components
                if method == 'pca':
                    reducer = PCA(n_components=n_components)
                elif method == 'ica':
                    reducer = FastICA(n_components=n_components)
                elif method == 'tsne':
                    reducer = TSNE(n_components=n_components, perplexity=min(30, len(acts) - 1) if len(acts) > 30 else 5)
                else:
                    raise ValueError(f"Unsupported dimensionality reduction method: {method}")
                
                if len(acts) > 0:
                    reduced_activations[class_idx] = reducer.fit_transform(acts)
                    print(f"Class {class_idx}: Reduced dimensions from {acts.shape[1]} to {n_components}")
            else:
                print(f"Class {class_idx}: Not enough samples ({len(acts)}) for dimensionality reduction")
                reduced_activations[class_idx] = acts if len(acts) > 0 else np.array([])
        
        return reduced_activations
    
    def perform_clustering(self, reduced_activations, method='kmeans', n_clusters=2, eps=0.5):
        """
        Perform clustering on reduced activations for each class.
        
        Args:
            reduced_activations: Dictionary with reduced activations for each class
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters (for KMeans)
            eps: Maximum distance between samples in a cluster (for DBSCAN)
            
        Returns:
            Dictionary with cluster labels for each class
        """
        cluster_labels = {}
        
        for class_idx, acts in reduced_activations.items():
            if len(acts) > 0:
                if method == 'kmeans' and len(acts) >= n_clusters:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels[class_idx] = clusterer.fit_predict(acts)
                elif method == 'dbscan':
                    clusterer = DBSCAN(eps=eps, min_samples=5)
                    cluster_labels[class_idx] = clusterer.fit_predict(acts)
                else:
                    print(f"Class {class_idx}: Not enough samples for clustering with {method}")
                    cluster_labels[class_idx] = np.array([])
            else:
                cluster_labels[class_idx] = np.array([])
        
        return cluster_labels
    
    def analyze_clusters(self, reduced_activations, cluster_labels, sample_indices):
        """
        Analyze clusters to identify potential backdoored inputs.
        
        Args:
            reduced_activations: Dictionary with reduced activations for each class
            cluster_labels: Dictionary with cluster labels for each class
            sample_indices: Dictionary with original dataset indices for each class
            
        Returns:
            Dictionary with indices of potential poisoned samples for each class
        """
        for class_idx, labels in cluster_labels.items():
            if len(labels) > 0:
                # Count samples in each cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"\nClass {class_idx} - Clusters: {len(unique_labels)}")
                
                for label, count in zip(unique_labels, counts):
                    print(f"  Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
                
                # Identify minority clusters (potential backdoor)
                # Heuristic: clusters with significantly fewer samples may be backdoored
                if len(unique_labels) > 1:
                    cluster_percentages = counts / len(labels)
                    minority_threshold = 0.3  # Clusters with less than 30% of samples
                    
                    for label, percentage in zip(unique_labels, cluster_percentages):
                        if percentage < minority_threshold:
                            # Get indices of samples in the minority cluster
                            cluster_mask = (labels == label)
                            class_sample_indices = np.array(sample_indices[class_idx])
                            potential_poison_indices = class_sample_indices[cluster_mask]
                            
                            print(f"  Cluster {label} identified as potential backdoor cluster")
                            print(f"  Found {len(potential_poison_indices)} potentially poisoned samples")
                            
                            # Store poisoned indices
                            self.poison_indices[class_idx].extend(potential_poison_indices)
        
        return self.poison_indices
    
    def visualize_clusters(self, reduced_activations, cluster_labels, save_path='./output/activation_clusters'):
        """
        Visualize the clusters for each class.
        
        Args:
            reduced_activations: Dictionary with reduced activations for each class
            cluster_labels: Dictionary with cluster labels for each class
            save_path: Path to save the visualizations
        """
        os.makedirs(save_path, exist_ok=True)
        
        for class_idx, acts in reduced_activations.items():
            if len(acts) > 0 and len(cluster_labels[class_idx]) > 0:
                plt.figure(figsize=(10, 8))
                
                # If we have 2D reduced activations, create a scatter plot
                if acts.shape[1] == 2:
                    plt.scatter(acts[:, 0], acts[:, 1], c=cluster_labels[class_idx], cmap='viridis', alpha=0.7)
                    plt.colorbar(label='Cluster')
                    plt.title(f'Activation Clusters for Class {class_idx}')
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    plt.grid(True, alpha=0.3)
                
                # If 3D, create a 3D scatter plot
                elif acts.shape[1] == 3:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(acts[:, 0], acts[:, 1], acts[:, 2], 
                                        c=cluster_labels[class_idx], cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, label='Cluster')
                    ax.set_title(f'Activation Clusters for Class {class_idx}')
                    ax.set_xlabel('Component 1')
                    ax.set_ylabel('Component 2')
                    ax.set_zlabel('Component 3')
                
                plt.savefig(f'{save_path}/class_{class_idx}_clusters.png')
                plt.close()
    
    def generate_clean_dataset(self, poisoned_indices):
        """
        Generate a clean dataset by removing the identified poisoned samples.
        
        Args:
            poisoned_indices: Dictionary with indices of poisoned samples for each class
            
        Returns:
            Clean dataset with poisoned samples removed
        """
        # Flatten all poisoned indices
        all_poisoned_indices = []
        for indices in poisoned_indices.values():
            all_poisoned_indices.extend(indices)
        
        # Create a set for faster lookup
        poisoned_set = set(all_poisoned_indices)
        
        # Create clean dataset by filtering out poisoned samples
        clean_indices = [i for i in range(len(self.dataset)) if i not in poisoned_set]
        
        # If torch.utils.data.Subset exists, use it
        from torch.utils.data import Subset
        clean_dataset = Subset(self.dataset, clean_indices)
        
        print(f"Generated clean dataset with {len(clean_dataset)} samples")
        print(f"Removed {len(poisoned_set)} potentially poisoned samples")
        
        return clean_dataset
    
    def defend(self, dim_reduction_method='pca', clustering_method='kmeans', n_components=2, n_clusters=2):
        """
        Perform the full activation clustering defense pipeline.
        
        Args:
            dim_reduction_method: Method for dimensionality reduction ('pca', 'ica', or 'tsne')
            clustering_method: Method for clustering ('kmeans' or 'dbscan')
            n_components: Number of components for dimensionality reduction
            n_clusters: Number of clusters for KMeans
            
        Returns:
            Clean dataset with poisoned samples removed
        """
        print("\nPerforming Activation Clustering Defense...")
        
        # Step 1: Collect activations from the model
        print("\nStep 1: Collecting activations...")
        sample_indices = self.collect_activations()
        
        # Step 2: Reduce dimensions of activations
        print("\nStep 2: Reducing dimensions...")
        reduced_activations = self.reduce_dimensions(method=dim_reduction_method, n_components=n_components)
        
        # Step 3: Perform clustering
        print("\nStep 3: Performing clustering...")
        cluster_labels = self.perform_clustering(reduced_activations, method=clustering_method, n_clusters=n_clusters)
        
        # Step 4: Analyze clusters to identify poisoned samples
        print("\nStep 4: Analyzing clusters for backdoor detection...")
        poisoned_indices = self.analyze_clusters(reduced_activations, cluster_labels, sample_indices)
        
        # Step 5: Visualize clusters
        print("\nStep 5: Visualizing clusters...")
        self.visualize_clusters(reduced_activations, cluster_labels)
        
        # Step 6: Generate clean dataset
        print("\nStep 6: Generating clean dataset...")
        clean_dataset = self.generate_clean_dataset(poisoned_indices)
        
        return clean_dataset, poisoned_indices
    
    def evaluate_defense(self, model, clean_dataset, test_dataset, trigger_pattern=None, target_label=None, attack_type="1"):
        """
        Evaluate the effectiveness of the defense by training on the cleaned dataset.
        
        Args:
            model: The model to train
            clean_dataset: Clean dataset with poisoned samples removed
            test_dataset: Test dataset for evaluation
            trigger_pattern: Pattern used for backdoor attacks (if available)
            target_label: Target label for backdoor attacks
            attack_type: Type of attack (1 for trigger-based, 2 for semantic)
            
        Returns:
            Clean accuracy and backdoor success rate of the defended model
        """
        print("\nEvaluating activation clustering defense...")
        
        # Create a copy of the model for training
        import copy
        from torch.utils.data import DataLoader
        import torch.optim as optim
        import torch.nn as nn
        
        defended_model = copy.deepcopy(model)
        
        # Train the model on the clean dataset
        print("\nTraining model on cleaned dataset...")
        train_loader = DataLoader(clean_dataset, batch_size=128, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(defended_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        # Train for a few epochs
        defended_model.train()
        for epoch in range(3):  # 3 epochs should be enough for fine-tuning
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = defended_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss/(batch_idx+1):.3f}, '
                          f'Acc: {100.*correct/total:.3f}%')
        
        # Evaluate the defended model
        print("\nEvaluating the defended model...")
        defended_model.eval()
        
        # Test on clean data
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = defended_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        clean_acc = 100. * correct / total
        print(f'Clean Test Accuracy: {clean_acc:.2f}%')
        
        # Test on backdoored data
        if trigger_pattern is not None and target_label is not None:
            if attack_type == "1":
                from src.utils.backdoor_utils import evaluate_backdoor
                backdoor_acc = evaluate_backdoor(defended_model, test_dataset, trigger_pattern, target_label, self.device)
            else:
                from src.utils.backdoor_sem import evaluate_semantic_backdoor
                backdoor_acc = evaluate_semantic_backdoor(defended_model, test_dataset, target_label, self.device)
            
            print(f'Backdoor Attack Success Rate: {backdoor_acc:.2f}%')
            
            return clean_acc, backdoor_acc
        
        return clean_acc, None