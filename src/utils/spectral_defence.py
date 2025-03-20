import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.cluster import KMeans
import copy
from scipy.stats import entropy
from src.utils.data_loader import load_data, get_transforms


class SpectralBackdoorDefender:
    """
    A novel defense mechanism against backdoor attacks using spectral analysis
    and activation clustering.
    """
    def visualize_eigenspectra(self, top_n=10):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        if not self.eigenspectra:
            self.compute_eigenspectrum()
        
        # Get layers for visualization
        all_layers = set()
        for class_idx in self.eigenspectra:
            all_layers.update(self.eigenspectra[class_idx].keys())
        
        # Choose a subset of layers to visualize
        layers_to_visualize = list(all_layers)[:min(6, len(all_layers))]
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        for i, layer_name in enumerate(layers_to_visualize):
            # Create subplot
            ax = fig.add_subplot(gs[i//3, i%3])
            
            # Plot eigenvalues for each class
            for class_idx in range(self.num_classes):
                if class_idx in self.eigenspectra and layer_name in self.eigenspectra[class_idx]:
                    eigenvalues = self.eigenspectra[class_idx][layer_name]
                    
                    # Plot only top N eigenvalues
                    n_values = min(top_n, len(eigenvalues))
                    
                    # Highlight suspected backdoor classes
                    if hasattr(self, 'suspected_classes') and class_idx in self.suspected_classes:
                        ax.plot(range(n_values), eigenvalues[:n_values], 'o-', 
                                label=f'Class {class_idx} (Suspected)', linewidth=2, markersize=8)
                    else:
                        ax.plot(range(n_values), eigenvalues[:n_values], 'o-', 
                                label=f'Class {class_idx}', alpha=0.7)
            
            ax.set_title(f'Layer: {layer_name}')
            ax.set_xlabel('Eigenvalue Index')
            ax.set_ylabel('Normalized Eigenvalue')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
        # Add legend outside the plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
                ncol=min(5, self.num_classes))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.suptitle('Eigenspectra of Layer Activations by Class', fontsize=16)
        
        return fig

    def visualize_activation_clusters(self, layer_name, class_idx, method='umap', n_clusters=2):
        """
        Visualize activation clusters for a specific layer and class.
        
        Args:
            layer_name: Name of the layer to visualize
            class_idx: Class index to visualize
            method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
            n_clusters: Number of clusters for KMeans
        """
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        
        if not self.activation_patterns:
            raise ValueError("No activations collected. Run collect_layer_activations() first.")
        
        if (class_idx not in self.activation_patterns or 
            layer_name not in self.activation_patterns[class_idx]):
            raise ValueError(f"No activations for class {class_idx} and layer {layer_name}")
        
        # Get activations
        activations = self.activation_patterns[class_idx][layer_name]
        
        # Cluster activations
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(activations)
        labels = kmeans.labels_
        
        # Dimensionality reduction for visualization
        if method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
                embedding = reducer.fit_transform(activations)
            except ImportError:
                print("UMAP not installed. Using PCA instead.")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(activations)
        elif method == 'tsne':
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                embedding = reducer.fit_transform(activations)
            except ImportError:
                print("t-SNE not installed. Using PCA instead.")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(activations)
        else:  # default to PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(activations)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Get cluster sizes
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        
        # Plot each cluster
        for i in range(n_clusters):
            cluster_points = embedding[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        label=f'Cluster {i} (n={cluster_sizes[i]})', 
                        alpha=0.7, s=50)
        
        # Plot cluster centers
        center_embeddings = reducer.transform(kmeans.cluster_centers_)
        plt.scatter(center_embeddings[:, 0], center_embeddings[:, 1], 
                    marker='X', s=200, color='red', label='Cluster Centers')
        
        plt.title(f'Activation Clusters for Class {class_idx}, Layer {layer_name}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()

    def visualize_anomaly_metrics(self, layers_to_analyze=None):
        """
        Visualize anomaly detection metrics for backdoor detection.
        
        Args:
            layers_to_analyze: List of layer names to analyze. If None, analyze all collected layers.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not hasattr(self, 'eigenspectra'):
            self.compute_eigenspectrum()
        
        if layers_to_analyze is None:
            # Use all available layers
            all_layers = set()
            for class_idx in self.eigenspectra:
                all_layers.update(self.eigenspectra[class_idx].keys())
            layers_to_analyze = list(all_layers)
        
        # Compute metrics
        metrics = {}
        
        # Spectral entropy
        spectral_entropy = {class_idx: {} for class_idx in range(self.num_classes)}
        for class_idx in self.eigenspectra:
            for layer_name in layers_to_analyze:
                if layer_name in self.eigenspectra[class_idx]:
                    eigenvalues = self.eigenspectra[class_idx][layer_name]
                    # Compute spectral entropy (measures distribution uniformity)
                    from scipy.stats import entropy
                    spectral_entropy[class_idx][layer_name] = entropy(eigenvalues + 1e-10)
        
        metrics['Spectral Entropy'] = spectral_entropy
        
        # Variance of eigenvalues
        spectral_variance = {class_idx: {} for class_idx in range(self.num_classes)}
        for class_idx in self.eigenspectra:
            for layer_name in layers_to_analyze:
                if layer_name in self.eigenspectra[class_idx]:
                    eigenvalues = self.eigenspectra[class_idx][layer_name]
                    spectral_variance[class_idx][layer_name] = np.var(eigenvalues)
        
        metrics['Spectral Variance'] = spectral_variance
        
        # Dominant eigenvalue ratio
        dominant_ratio = {class_idx: {} for class_idx in range(self.num_classes)}
        for class_idx in self.eigenspectra:
            for layer_name in layers_to_analyze:
                if layer_name in self.eigenspectra[class_idx]:
                    eigenvalues = self.eigenspectra[class_idx][layer_name]
                    if len(eigenvalues) > 1:
                        dominant_ratio[class_idx][layer_name] = eigenvalues[0] / (np.sum(eigenvalues[1:]) + 1e-10)
                    else:
                        dominant_ratio[class_idx][layer_name] = 1.0
        
        metrics['Dominant Eigenvalue Ratio'] = dominant_ratio
        
        # Visualization
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        
        for i, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = axes[i] if len(metrics) > 1 else axes
            
            # Prepare data for heatmap
            heatmap_data = np.zeros((self.num_classes, len(layers_to_analyze)))
            heatmap_data.fill(np.nan)  # Fill with NaN for missing values
            
            for class_idx in range(self.num_classes):
                for j, layer_name in enumerate(layers_to_analyze):
                    if class_idx in metric_data and layer_name in metric_data[class_idx]:
                        heatmap_data[class_idx, j] = metric_data[class_idx][layer_name]
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=[l.split('.')[-1] for l in layers_to_analyze],
                        yticklabels=range(self.num_classes), ax=ax)
            
            ax.set_title(f'{metric_name} by Class and Layer')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Class')
            
            # Highlight suspected classes if available
            if hasattr(self, 'suspected_classes') and self.suspected_classes:
                for class_idx in self.suspected_classes:
                    ax.get_yticklabels()[class_idx].set_color("red")
                    ax.get_yticklabels()[class_idx].set_fontweight("bold")
        
        plt.tight_layout()
        return fig

    def visualize_neuron_activations(self, layer_name, top_n=10):
        """
        Visualize neuron activations for a specific layer across all classes.
        
        Args:
            layer_name: Name of the layer to visualize
            top_n: Number of top neurons to visualize
        """
        import matplotlib.pyplot as plt
        
        if not self.activation_patterns:
            raise ValueError("No activations collected. Run collect_layer_activations() first.")
        
        # Check if the layer exists in activation patterns
        classes_with_layer = [c for c in self.activation_patterns if layer_name in self.activation_patterns[c]]
        if not classes_with_layer:
            raise ValueError(f"No activations for layer {layer_name}")
        
        # Get number of neurons in the layer
        num_neurons = self.activation_patterns[classes_with_layer[0]][layer_name].shape[1]
        
        # Get mean activations for each class
        mean_activations = {}
        for class_idx in classes_with_layer:
            activations = self.activation_patterns[class_idx][layer_name]
            mean_activations[class_idx] = np.mean(activations, axis=0)
        
        # Find top neurons by activation for each class
        top_neurons = {}
        for class_idx, activations in mean_activations.items():
            # Get indices of top N neurons
            top_indices = np.argsort(activations)[-top_n:][::-1]
            top_neurons[class_idx] = top_indices
        
        # Visualize
        fig, axes = plt.subplots(len(classes_with_layer), 1, figsize=(12, 3 * len(classes_with_layer)))
        
        if len(classes_with_layer) == 1:
            axes = [axes]
        
        for i, class_idx in enumerate(sorted(classes_with_layer)):
            ax = axes[i]
            
            # Plot mean activations
            ax.bar(range(num_neurons), mean_activations[class_idx], alpha=0.7)
            
            # Highlight top neurons
            ax.bar(top_neurons[class_idx], 
                [mean_activations[class_idx][j] for j in top_neurons[class_idx]], 
                color='red', alpha=0.7)
            
            # Styling
            ax.set_title(f'Class {class_idx} Neuron Activations for Layer {layer_name}')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Mean Activation')
            ax.grid(True, alpha=0.3)
            
            # Is this a suspected backdoor class?
            if hasattr(self, 'suspected_classes') and class_idx in self.suspected_classes:
                ax.set_title(f'Class {class_idx} (SUSPECTED BACKDOOR) Neuron Activations for Layer {layer_name}')
                ax.set_facecolor('lightyellow')
        
        plt.tight_layout()
        return fig

    def visualize_repair_comparison(self, original_model, repaired_model, test_dataset, num_samples=5):
        """
        Visualize the difference between original and repaired model for suspected backdoor classes.
        
        Args:
            original_model: The original backdoored model
            repaired_model: The repaired model
            test_dataset: Clean test dataset
            num_samples: Number of samples to visualize per class
        """
        import matplotlib.pyplot as plt
        import torch
        
        if not hasattr(self, 'suspected_classes') or not self.suspected_classes:
            print("No suspected backdoor classes found.")
            return None
        
        # Create a test loader
        from torch.utils.data import DataLoader, Subset
        import random
        
        # Get samples from suspected classes
        indices_by_class = {}
        for i, (_, label) in enumerate(test_dataset):
            if label in self.suspected_classes:
                if label not in indices_by_class:
                    indices_by_class[label] = []
                indices_by_class[label].append(i)
        
        # Select random samples
        selected_indices = []
        for label, indices in indices_by_class.items():
            selected_indices.extend(random.sample(indices, min(num_samples, len(indices))))
        
        subset = Subset(test_dataset, selected_indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        
        # Set models to evaluation mode
        original_model.eval()
        repaired_model.eval()
        
        # Create figure
        num_classes = len(indices_by_class)
        fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 3, num_classes * 3))
        
        if num_classes == 1:
            axes = [axes]
        
        # Keep track of which samples we've plotted
        plotted_per_class = {class_idx: 0 for class_idx in indices_by_class}
        
        # Plot samples
        with torch.no_grad():
            for data, label in loader:
                label = label.item()
                if plotted_per_class[label] >= num_samples:
                    continue
                    
                # Get predictions
                data = data.to(self.device)
                original_output = original_model(data)
                repaired_output = repaired_model(data)
                
                original_pred = original_output.max(1)[1].item()
                repaired_pred = repaired_output.max(1)[1].item()
                
                # Plot image
                ax = axes[list(indices_by_class.keys()).index(label)][plotted_per_class[label]]
                
                # Convert tensor to numpy array
                img = data.squeeze().cpu().numpy()
                
                # Handle different image formats
                if img.shape[0] == 1:  # Grayscale
                    ax.imshow(img.squeeze(), cmap='gray')
                elif img.shape[0] == 3:  # RGB
                    ax.imshow(np.transpose(img, (1, 2, 0)))
                
                # Add title with predictions
                ax.set_title(f"True: {label}\nOrig: {original_pred}\nRepaired: {repaired_pred}")
                ax.axis('off')
                
                # Highlight if predictions differ
                if original_pred != repaired_pred:
                    ax.set_facecolor('lightyellow')
                    
                plotted_per_class[label] += 1
        
        plt.tight_layout()
        return fig

    def visualize_training_progress(self, training_history):
        """
        Visualize training progress during model repair.
        
        Args:
            training_history: Dictionary containing training history
                (loss, accuracy, validation accuracy for each epoch)
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        ax1.plot(training_history['epoch'], training_history['loss'], 'bo-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(training_history['epoch'], training_history['train_acc'], 'ro-', label='Training Accuracy')
        ax2.plot(training_history['epoch'], training_history['val_acc'], 'go-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def visualize_backdoor_detection_summary(self, eigenvalue_metrics):
        """
        Create a summary visualization of backdoor detection results.
        
        Args:
            eigenvalue_metrics: Dictionary of computed metrics from detect_backdoors
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not hasattr(self, 'suspected_classes'):
            print("No backdoor detection results available.")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot anomaly scores
        anomaly_scores = eigenvalue_metrics['anomaly_scores']
        classes = sorted(anomaly_scores.keys())
        scores = [anomaly_scores[c] for c in classes]
        
        bars = ax1.bar(classes, scores, color='lightblue')
        
        # Highlight suspected classes
        for i, c in enumerate(classes):
            if c in self.suspected_classes:
                bars[i].set_color('red')
        
        ax1.set_title('Anomaly Scores by Class')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Anomaly Score')
        ax1.set_xticks(classes)
        ax1.grid(True, alpha=0.3)
        
        # Add threshold line
        threshold = 2  # Same threshold used in detect_backdoors
        ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        ax1.legend()
        
        # Spectral entropy heatmap
        spectral_entropy = eigenvalue_metrics['spectral_entropy']
        
        # Prepare data for heatmap
        layers = list(next(iter(spectral_entropy.values())).keys())[:5]  # Top 5 layers
        entropy_data = []
        
        for c in classes:
            if c in spectral_entropy:
                row = [spectral_entropy[c].get(layer, np.nan) for layer in layers]
                entropy_data.append(row)
        
        # Create heatmap
        sns.heatmap(entropy_data, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=[l.split('.')[-1] for l in layers],
                    yticklabels=classes, ax=ax2)
        
        ax2.set_title('Spectral Entropy by Class and Layer')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Class')
        
        # Highlight suspected classes
        for i, c in enumerate(classes):
            if c in self.suspected_classes:
                ax2.get_yticklabels()[i].set_color("red")
                ax2.get_yticklabels()[i].set_fontweight("bold")
        
        plt.tight_layout()
        return fig

    def visualize_neuron_pruning(self, model, repaired_model, suspicious_neurons):
        """
        Visualize the effect of neuron pruning on the model.
        
        Args:
            model: Original model
            repaired_model: Repaired model after pruning
            suspicious_neurons: Dictionary mapping layer names to sets of suspicious neuron indices
        """
        import matplotlib.pyplot as plt
        import torch
        
        fig, axs = plt.subplots(len(suspicious_neurons), 1, figsize=(12, 4 * len(suspicious_neurons)))
        
        if len(suspicious_neurons) == 1:
            axs = [axs]
        
        for i, (layer_name, neurons) in enumerate(suspicious_neurons.items()):
            ax = axs[i]
            
            # Get the module
            orig_module = model
            for part in layer_name.split('.'):
                orig_module = getattr(orig_module, part)
                
            repaired_module = repaired_model
            for part in layer_name.split('.'):
                repaired_module = getattr(repaired_module, part)
            
            # Get weights
            if isinstance(orig_module, torch.nn.Conv2d) or isinstance(orig_module, torch.nn.Linear):
                orig_weights = orig_module.weight.data.cpu().numpy()
                repaired_weights = repaired_module.weight.data.cpu().numpy()
                
                # Compute weight magnitudes
                if len(orig_weights.shape) == 4:  # Conv2d
                    orig_magnitude = np.sqrt(np.sum(np.square(orig_weights), axis=(1, 2, 3)))
                    repaired_magnitude = np.sqrt(np.sum(np.square(repaired_weights), axis=(1, 2, 3)))
                else:  # Linear
                    orig_magnitude = np.sqrt(np.sum(np.square(orig_weights), axis=1))
                    repaired_magnitude = np.sqrt(np.sum(np.square(repaired_weights), axis=1))
                
                # Number of neurons to display
                num_neurons = min(100, len(orig_magnitude))
                
                # Plot original weights
                ax.plot(range(num_neurons), orig_magnitude[:num_neurons], 'b-', alpha=0.7, label='Original')
                
                # Plot repaired weights
                ax.plot(range(num_neurons), repaired_magnitude[:num_neurons], 'g-', alpha=0.7, label='Repaired')
                
                # Highlight suspicious neurons
                suspicious = list(neurons)
                suspicious = [n for n in suspicious if n < num_neurons]
                
                if suspicious:
                    ax.scatter(suspicious, orig_magnitude[suspicious], color='red', s=50, label='Suspicious')
                    ax.scatter(suspicious, repaired_magnitude[suspicious], color='green', s=50)
                
                # Styling
                ax.set_title(f'Weight Magnitudes for Layer {layer_name}')
                ax.set_xlabel('Neuron Index')
                ax.set_ylabel('Weight Magnitude')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        return fig

    def __init__(self, model, device, num_classes=10, input_shape=(3, 32, 32)):
        """
        Initialize the Spectral Backdoor Defender.
        
        Args:
            model: The model to analyze and repair
            device: Device to run computations on (cuda/cpu)
            num_classes: Number of output classes
            input_shape: Shape of input images (channels, height, width)
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.activation_patterns = {}
        self.suspected_classes = []
        self.eigenspectra = {}
        
        # Load clean training data for analysis and repair
        train_data, _ = load_data()
        self.clean_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        
        # Extract class-wise data
        self.class_data = {}
        for i in range(len(train_data)):
            img, label = train_data[i]
            if isinstance(label, torch.Tensor):
                label = label.item()  # Convert tensor to integer if needed
            if label not in self.class_data:
                self.class_data[label] = []
            self.class_data[label].append(i)
        
        # Create class-wise dataloaders
        self.class_loaders = {}
        for class_idx, indices in self.class_data.items():
            class_subset = Subset(train_data, indices)
            self.class_loaders[class_idx] = DataLoader(class_subset, batch_size=64, shuffle=True)
    
    def collect_layer_activations(self, layer_names=None):
        """
        Collect activations for specific layers in the model across all classes.
        
        Args:
            layer_names: List of layer names to monitor. If None, monitor all convolutional and linear layers.
        """
        print("Collecting layer activations...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # If no layer names provided, automatically select all conv and linear layers
        if layer_names is None:
            layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and len(name) > 0:
                    layer_names.append(name)
        
        # Set up hooks to capture activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        for name in layer_names:
            # Get the module
            module = self.model
            for part in name.split('.'):
                module = getattr(module, part)
            
            # Register hook
            hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Collect activations for each class
        self.activation_patterns = {class_idx: {layer: [] for layer in layer_names} 
                                   for class_idx in range(self.num_classes)}
        
        # Process each class separately
        for class_idx, loader in self.class_loaders.items():
            # Only process a limited number of samples per class for efficiency
            sample_count = 0
            max_samples = 100  # Maximum samples per class
            
            for data, _ in loader:
                data = data.to(self.device)
                # Forward pass
                self.model(data)
                
                # Store activations
                for layer_name in layer_names:
                    # For convolutional layers, use global average pooling to get a feature vector
                    layer_output = activations[layer_name]
                    if len(layer_output.shape) == 4:  # Conv layer (B, C, H, W)
                        # Global average pooling
                        layer_output = layer_output.mean(dim=[2, 3])  # -> (B, C)
                    
                    # Store the flattened activations
                    self.activation_patterns[class_idx][layer_name].append(layer_output.cpu().numpy())
                
                sample_count += data.size(0)
                if sample_count >= max_samples:
                    break
        
        # Concatenate all samples for each class and layer
        for class_idx in self.activation_patterns:
            for layer_name in self.activation_patterns[class_idx]:
                if len(self.activation_patterns[class_idx][layer_name]) > 0:
                    self.activation_patterns[class_idx][layer_name] = np.concatenate(
                        self.activation_patterns[class_idx][layer_name], axis=0)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"Collected activations for {len(layer_names)} layers across {self.num_classes} classes.")
        
        return self.activation_patterns
    
    def compute_eigenspectrum(self):
        """
        Compute the eigenspectrum of activations for each class and layer.
        The eigenspectrum can reveal anomalies in the activation patterns.
        """
        print("Computing eigenspectra...")
        
        if not self.activation_patterns:
            raise ValueError("No activations collected. Run collect_layer_activations() first.")
        
        self.eigenspectra = {class_idx: {} for class_idx in range(self.num_classes)}
        
        for class_idx in self.activation_patterns:
            for layer_name, activations in self.activation_patterns[class_idx].items():
                if activations.shape[0] < 2:  # Need at least 2 samples
                    continue
                
                # Center the data
                centered_data = activations - np.mean(activations, axis=0)
                
                # Compute covariance matrix
                cov_matrix = np.cov(centered_data, rowvar=False)
                
                # Compute eigenvalues and sort in descending order
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Store normalized eigenvalues
                if np.sum(eigenvalues) > 0:
                    normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
                    self.eigenspectra[class_idx][layer_name] = normalized_eigenvalues
                else:
                    self.eigenspectra[class_idx][layer_name] = eigenvalues
        
        return self.eigenspectra
    
    def detect_backdoors(self, layers_to_analyze=None, threshold_factor=2.0):
        """
        Detect potential backdoors by analyzing activation patterns using:
        1. Eigenspectrum analysis - backdoored models often show distinct spectra
        2. Activation clustering - backdoored samples form separate clusters
        
        Args:
            layers_to_analyze: List of layer names to analyze. If None, analyze all collected layers.
            threshold_factor: Threshold factor for anomaly detection
            
        Returns:
            backdoor_detected: Boolean indicating if backdoor was detected
            suspected_classes: List of suspected backdoor classes
            eigenvalue_metrics: Dictionary of computed metrics
        """
        if not self.eigenspectra:
            self.collect_layer_activations()
            self.compute_eigenspectrum()
        
        if layers_to_analyze is None:
            # Use all available layers
            all_layers = set()
            for class_idx in self.eigenspectra:
                all_layers.update(self.eigenspectra[class_idx].keys())
            layers_to_analyze = list(all_layers)
        
        # Initialize metrics
        spectral_entropy = {class_idx: {} for class_idx in range(self.num_classes)}
        spectral_variance = {class_idx: {} for class_idx in range(self.num_classes)}
        dominant_eigenvalue_ratio = {class_idx: {} for class_idx in range(self.num_classes)}
        
        # Compute metrics for each class and layer
        for class_idx in self.eigenspectra:
            for layer_name in layers_to_analyze:
                if layer_name not in self.eigenspectra[class_idx]:
                    continue
                    
                eigenvalues = self.eigenspectra[class_idx][layer_name]
                
                # Compute spectral entropy (measures distribution uniformity)
                spectral_entropy[class_idx][layer_name] = entropy(eigenvalues + 1e-10)
                
                # Compute variance of eigenspectrum
                spectral_variance[class_idx][layer_name] = np.var(eigenvalues)
                
                # Compute ratio of dominant eigenvalue to the rest (concentration metric)
                if len(eigenvalues) > 1:
                    dominant_eigenvalue_ratio[class_idx][layer_name] = eigenvalues[0] / (np.sum(eigenvalues[1:]) + 1e-10)
                else:
                    dominant_eigenvalue_ratio[class_idx][layer_name] = 1.0
        
        # Detect anomalies in each metric
        anomaly_scores = {class_idx: 0 for class_idx in range(self.num_classes)}
        
        metrics = [
            ("Spectral Entropy", spectral_entropy),
            ("Spectral Variance", spectral_variance),
            ("Dominant Eigenvalue Ratio", dominant_eigenvalue_ratio)
        ]
        
        for metric_name, metric_values in metrics:
            for layer_name in layers_to_analyze:
                # Collect values across classes
                values = []
                for class_idx in range(self.num_classes):
                    if class_idx in metric_values and layer_name in metric_values[class_idx]:
                        values.append((class_idx, metric_values[class_idx][layer_name]))
                
                if len(values) < self.num_classes // 2:
                    continue  # Skip if we don't have enough data
                
                # Compute median and mad
                values.sort(key=lambda x: x[1])
                median_idx = len(values) // 2
                median_value = values[median_idx][1]
                
                mad = np.median([abs(v[1] - median_value) for v in values])
                
                if mad == 0:
                    continue  # Skip if no variation
                
                # Identify outliers
                for class_idx, value in values:
                    # Lower values are suspicious for entropy, higher for the other metrics
                    if metric_name == "Spectral Entropy":
                        if value < median_value - threshold_factor * mad:
                            anomaly_scores[class_idx] += 1
                            print(f"Class {class_idx} has anomalously low {metric_name} in layer {layer_name}")
                    else:
                        if value > median_value + threshold_factor * mad:
                            anomaly_scores[class_idx] += 1
                            print(f"Class {class_idx} has anomalously high {metric_name} in layer {layer_name}")
        
        # Secondary detection: activation clustering
        for class_idx in range(self.num_classes):
            for layer_name in layers_to_analyze:
                if (class_idx in self.activation_patterns and 
                    layer_name in self.activation_patterns[class_idx] and
                    len(self.activation_patterns[class_idx][layer_name]) >= 20):  # Need sufficient samples
                    
                    activations = self.activation_patterns[class_idx][layer_name]
                    
                    # Use K-means clustering to check for multimodal activation distribution
                    kmeans = KMeans(n_clusters=2, random_state=42).fit(activations)
                    labels = kmeans.labels_
                    
                    # Calculate the silhouette score to measure cluster separation
                    # We'll use a simple approximation based on cluster sizes and distances
                    cluster_0_size = np.sum(labels == 0)
                    cluster_1_size = np.sum(labels == 1)
                    
                    # If one cluster is much smaller than the other (potential backdoor trigger)
                    cluster_size_ratio = min(cluster_0_size, cluster_1_size) / (max(cluster_0_size, cluster_1_size) + 1e-10)
                    
                    # If we have a suspicious small cluster (less than 20% of the main cluster)
                    if cluster_size_ratio < 0.2:
                        # Calculate distance between clusters
                        cluster_centers = kmeans.cluster_centers_
                        center_distance = np.linalg.norm(cluster_centers[0] - cluster_centers[1])
                        
                        # Normalized by average within-cluster distance
                        within_distances = []
                        for i in range(2):
                            cluster_points = activations[labels == i]
                            if len(cluster_points) > 1:
                                within_distances.append(np.mean([
                                    np.linalg.norm(p - cluster_centers[i]) for p in cluster_points
                                ]))
                        
                        if within_distances:
                            avg_within_distance = np.mean(within_distances)
                            separation_ratio = center_distance / (avg_within_distance + 1e-10)
                            
                            # High separation ratio indicates well-separated clusters (suspicious)
                            if separation_ratio > 3.0:
                                anomaly_scores[class_idx] += 1
                                print(f"Class {class_idx} shows suspicious clustering in layer {layer_name}")
        
        # Determine suspected classes
        self.suspected_classes = []
        backdoor_detected = False
        
        # Consider a class suspicious if it has multiple anomalies
        threshold = 2  # Multiple metrics must indicate an anomaly
        for class_idx, score in anomaly_scores.items():
            if score >= threshold:
                print(f"Potential backdoor detected in class {class_idx} (Anomaly score: {score})")
                self.suspected_classes.append(class_idx)
                backdoor_detected = True
        
        eigenvalue_metrics = {
            "spectral_entropy": spectral_entropy,
            "spectral_variance": spectral_variance,
            "dominant_eigenvalue_ratio": dominant_eigenvalue_ratio,
            "anomaly_scores": anomaly_scores
        }
        
        return backdoor_detected, self.suspected_classes, eigenvalue_metrics
    
    def repair_model(self, compromised_layers=None, fine_tuning_epochs=5, fine_tuning_lr=0.001):
        """
        Repair the model using activation pruning and adversarial training.
        
        Args:
            fine_tuning_epochs: Number of epochs for fine-tuning
            fine_tuning_lr: Learning rate for fine-tuning
            
        Returns:
            repaired_model: The repaired model
        """
        if not self.suspected_classes:
            print("No backdoors detected. No repair needed.")
            return self.model
        
        print(f"Repairing model for suspected classes: {self.suspected_classes}")
        
        # Create a copy of the model to repair
        repaired_model = copy.deepcopy(self.model)
        repaired_model.to(self.device)
        
        # Step 1: Identify and prune suspicious neurons using activation patterns
        if not self.activation_patterns:
            self.collect_layer_activations()
        
        # Find layers with suspiciously high activations for backdoor classes
        suspicious_neurons = {}
        
        for class_idx in self.suspected_classes:
            for layer_name, activations in self.activation_patterns[class_idx].items():
                # Skip if not enough samples
                if len(activations) < 10:
                    continue
                    
                # Calculate mean activation per neuron
                mean_activations = np.mean(activations, axis=0)
                
                # Find neurons with unusually high activation
                threshold = np.mean(mean_activations) + 2 * np.std(mean_activations)
                neurons_to_prune = np.where(mean_activations > threshold)[0]
                
                if len(neurons_to_prune) > 0:
                    if layer_name not in suspicious_neurons:
                        suspicious_neurons[layer_name] = set()
                    suspicious_neurons[layer_name].update(neurons_to_prune)
        
        # Prune suspicious neurons
        with torch.no_grad():
            for layer_name, neurons in suspicious_neurons.items():
                # Get the module
                module = repaired_model
                for part in layer_name.split('.'):
                    module = getattr(module, part)
                
                # Prune neurons
                if isinstance(module, nn.Conv2d):
                    for neuron_idx in neurons:
                        if neuron_idx < module.weight.shape[0]:  # Check bounds
                            # Reduce the weight magnitude instead of zeroing out
                            module.weight.data[neuron_idx] *= 0.1
                            if module.bias is not None:
                                module.bias.data[neuron_idx] *= 0.1
                
                elif isinstance(module, nn.Linear):
                    for neuron_idx in neurons:
                        if neuron_idx < module.weight.shape[0]:  # Check bounds
                            module.weight.data[neuron_idx] *= 0.1
                            if module.bias is not None:
                                module.bias.data[neuron_idx] *= 0.1
        
        # Step 2: Fine-tune the model with both regular and adversarial training
        repaired_model.train()
        
        # Define optimizer and scheduler
        optimizer = optim.Adam(repaired_model.parameters(), lr=fine_tuning_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        loss_fn = nn.CrossEntropyLoss()
        
        # Track best model
        best_acc = 0
        best_model_state = None
        
        print("Starting fine-tuning with adversarial regularization...")
        
        # Create adversarial perturbations for training
        def create_adversarial_example(model, data, target, epsilon=0.03):
            # FGSM-based adversarial example generation
            data.requires_grad = True
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # Get the sign of the gradients
            data_grad = data.grad.data
            
            # Create perturbation
            perturb = epsilon * data_grad.sign()
            
            # Create adversarial example
            adv_data = data + perturb
            adv_data = torch.clamp(adv_data, 0, 1)  # Ensure valid image range
            
            return adv_data
        
        for epoch in range(fine_tuning_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data, targets in self.clean_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Regular forward pass
                outputs = repaired_model(data)
                loss = loss_fn(outputs, targets)
                
                # Create adversarial examples
                # Only for examples from suspected classes for efficiency
                mask = torch.zeros_like(targets, dtype=torch.bool)
                for c in self.suspected_classes:
                    mask = mask | (targets == c)
                
                if torch.any(mask):
                    # Create adversarial examples for suspected classes
                    adv_data = create_adversarial_example(repaired_model, data[mask], targets[mask])
                    
                    # Forward pass with adversarial examples
                    adv_outputs = repaired_model(adv_data)
                    adv_loss = loss_fn(adv_outputs, targets[mask])
                    
                    # Combined loss
                    loss = loss + 0.5 * adv_loss
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Evaluate on clean validation data
            repaired_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in self.clean_loader:  # Using train data for validation for simplicity
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = repaired_model(data)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # Step the scheduler
            scheduler.step(val_acc)
            
            # Print epoch statistics
            epoch_loss = running_loss / len(self.clean_loader)
            epoch_acc = 100. * correct / total
            print(f"Epoch {epoch+1}/{fine_tuning_epochs}, Loss: {epoch_loss:.4f}, "
                  f"Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(repaired_model.state_dict())
            
            # Switch back to training mode
            repaired_model.train()
        
        # Restore best model
        if best_model_state is not None:
            repaired_model.load_state_dict(best_model_state)
        
        print(f"Fine-tuning complete. Best validation accuracy: {best_acc:.2f}%")
        return repaired_model
    
    def evaluate_repair(self, repaired_model, test_dataset, trigger_patterns=None, target_labels=None):
        """
        Evaluate the repaired model on clean and backdoored test data.
        
        Args:
            repaired_model: The repaired model
            test_dataset: Clean test dataset
            trigger_patterns: Dictionary mapping class indices to trigger patterns
            target_labels: Dictionary mapping class indices to target labels
            
        Returns:
            clean_acc: Accuracy on clean test data
            backdoor_accs: Dictionary of backdoor success rates for each class
        """
        # Create test loader
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Test on clean data
        repaired_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = repaired_model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        clean_acc = 100. * correct / total
        print(f"\nRepaired Model Results:")
        print(f"Clean Test Accuracy: {clean_acc:.2f}%")
        
        # Test on backdoored data if triggers are provided
        backdoor_accs = {}
        
        if trigger_patterns is not None and target_labels is not None:
            for class_idx in trigger_patterns:
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, targets in test_loader:
                        batch_size = data.size(0)
                        data, targets = data.to(self.device), targets.to(self.device)
                        
                        # Get original data shape
                        _, channels, height, width = data.shape
                        
                        # Get trigger pattern
                        trigger_pattern = trigger_patterns[class_idx].to(self.device)
                        target_label = target_labels[class_idx]
                        
                        # Reshape trigger_pattern to match data dimensions
                        if len(trigger_pattern.shape) == 2:  # (H, W)
                            trigger_pattern = trigger_pattern.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                        elif len(trigger_pattern.shape) == 3:  # (C, H, W)
                            trigger_pattern = trigger_pattern.unsqueeze(0)  # Add batch dimension
                        
                        # Extract trigger dimensions
                        trigger_height = min(trigger_pattern.shape[-2], height)
                        trigger_width = min(trigger_pattern.shape[-1], width)
                        
                        # Crop trigger pattern if it's larger than data
                        trigger_pattern = trigger_pattern[..., :trigger_height, :trigger_width]
                        
                        # Ensure trigger_pattern has the right batch size
                        if trigger_pattern.size(0) == 1 and batch_size > 1:
                            trigger_pattern = trigger_pattern.expand(batch_size, -1, -1, -1)
                        
                        # Apply trigger only to the top-left corner
                        poisoned_data = data.clone()
                        poisoned_data[:, :, :trigger_height, :trigger_width] = trigger_pattern
                        
                        outputs = repaired_model(poisoned_data)
                        _, predicted = outputs.max(1)
                        
                        # Success of a backdoor attack is measured by target class prediction rate
                        backdoor_success = (predicted == target_label).sum().item()
                        total += targets.size(0)
                        correct += backdoor_success 
                    
                    backdoor_acc = 100. * correct / total
                    backdoor_accs[class_idx] = backdoor_acc
                    print(f"Backdoor Attack Success Rate for class {class_idx}: {backdoor_acc:.2f}%")
            
        return clean_acc, backdoor_accs
    
    def visualize_eigenspectra(self):
        """
        Visualize the eigenspectra for each class and highlight anomalies.
        """
        if not self.eigenspectra:
            raise ValueError("No eigenspectra have been computed yet.")
        
        # Select a representative layer for visualization
        layer_to_show = list(next(iter(self.eigenspectra.values())).keys())[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot eigenspectrum for each class
        for class_idx in sorted(self.eigenspectra.keys()):
            if layer_to_show in self.eigenspectra[class_idx]:
                eigenvalues = self.eigenspectra[class_idx][layer_to_show]
                
                # Only plot top 20 eigenvalues for clarity
                top_k = min(20, len(eigenvalues))
                x = np.arange(top_k)
                
                # Different style for suspected classes
                if class_idx in self.suspected_classes:
                    ax.plot(x, eigenvalues[:top_k], 'o-', linewidth=2, 
                                label=f'Class {class_idx} [SUSPECTED]', alpha=0.8)
                else:
                    ax.plot(x, eigenvalues[:top_k], '.-', alpha=0.5,
                                label=f'Class {class_idx}')
        
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Normalized Eigenvalue Magnitude')
        ax.set_title(f'Eigenspectrum Analysis for Layer: {layer_to_show}')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('eigenspectra_analysis.png')
        plt.close()
        
        # Also visualize anomaly metrics
        if hasattr(self, 'suspected_classes') and len(self.suspected_classes) > 0:
            # Only if detect_backdoors has been run
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            metrics = [
                "spectral_entropy",
                "spectral_variance", 
                "dominant_eigenvalue_ratio"
            ]
            
            metric_titles = [
                "Spectral Entropy (lower is suspicious)",
                "Spectral Variance (higher is suspicious)",
                "Dominant Eigenvalue Ratio (higher is suspicious)"
            ]
            
            # Plot each metric
            for i, (metric_name, title) in enumerate(zip(metrics, metric_titles)):
                values = []
                
                for class_idx in range(self.num_classes):
                    if class_idx in self.eigenspectra and layer_to_show in self.eigenspectra[class_idx]:
                        if metric_name == "spectral_entropy":
                            val = entropy(self.eigenspectra[class_idx][layer_to_show] + 1e-10)
                        elif metric_name == "spectral_variance":
                            val = np.var(self.eigenspectra[class_idx][layer_to_show])
                        elif metric_name == "dominant_eigenvalue_ratio":
                            eigenvalues = self.eigenspectra[class_idx][layer_to_show]
                            if len(eigenvalues) > 1:
                                val = eigenvalues[0] / (np.sum(eigenvalues[1:]) + 1e-10)
                            else:
                                val = 1.0
                        
                        values.append((class_idx, val))
                
                # Sort by class index
                values.sort(key=lambda x: x[0])
                
                # Extract x and y values
                x = [v[0] for v in values]
                y = [v[1] for v in values]
                
                # Determine colors based on suspected classes
                colors = ['red' if idx in self.suspected_classes else 'blue' for idx in x]
                
                # Plot bar chart
                axes[i].bar(x, y, color=colors, alpha=0.7)
                axes[i].set_xlabel('Class Index')
                axes[i].set_ylabel(metric_name.replace('_', ' ').title())
                axes[i].set_title(title)
                axes[i].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # Mark suspected classes
                for idx in self.suspected_classes:
                    if idx in x:
                        idx_pos = x.index(idx)
                        axes[i].text(idx, y[idx_pos], 'Suspicious', 
                                        rotation=90, va='bottom', ha='center', color='red')
            
            plt.tight_layout()
            plt.savefig('backdoor_metrics.png')
            plt.close()










            