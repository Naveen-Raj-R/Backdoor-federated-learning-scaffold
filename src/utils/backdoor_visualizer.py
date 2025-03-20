import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

class BackdoorVisualizer:
    """
    Class for visualizing backdoor attacks and their effects on models
    """
    def __init__(self, output_path="./output"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def extract_features(self, model, dataloader, device, layer_name=None):
        """
        Extract features from a specific layer of the model
        
        Args:
            model: PyTorch model
            dataloader: DataLoader containing the dataset
            device: Device to run the model on
            layer_name: Name of the layer to extract features from (if None, uses the penultimate layer)
            
        Returns:
            features: Extracted features
            labels: Corresponding labels
            is_poisoned: Boolean array indicating if the sample was poisoned
        """
        features = []
        labels = []
        is_poisoned = []
        
        # Register hook to get intermediate layer activations
        if layer_name is None:
            # If no specific layer is provided, get the second-to-last layer
            # This assumes a typical structure with the last layer being the classifier
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            # For many models, the penultimate layer might be named 'fc' or similar
            # Adjust according to your specific model architecture
            if hasattr(model, 'fc'):
                # Register hook on the layer before the final classifier
                if hasattr(model, 'avgpool'):
                    model.avgpool.register_forward_hook(get_activation('features'))
                else:
                    # Fallback to the layer before fc
                    for name, module in list(model.named_modules())[-2:-1]:
                        module.register_forward_hook(get_activation('features'))
            else:
                # For other architectures, try to find a suitable feature extractor
                # This is a simplistic approach and might need adjustment
                feature_layers = [name for name, module in model.named_modules() 
                                 if isinstance(module, torch.nn.AdaptiveAvgPool2d) or 
                                    isinstance(module, torch.nn.AvgPool2d)]
                if feature_layers:
                    for name in feature_layers[-1:]:
                        model._modules[name].register_forward_hook(get_activation('features'))
                else:
                    # Last resort - use the second to last layer
                    modules = list(model.named_modules())
                    if len(modules) >= 2:
                        name, module = modules[-2]
                        module.register_forward_hook(get_activation('features'))
        else:
            # Use the specified layer
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            for name, module in model.named_modules():
                if name == layer_name:
                    module.register_forward_hook(get_activation('features'))
        
        model.eval()
        with torch.no_grad():
            for data in dataloader:
                inputs, targets = data[0].to(device), data[1].to(device)
                
                # Check if the dataset provides poisoning information
                if len(data) > 2:
                    poisoned = data[2].to(device)
                else:
                    # If no poisoning information, assume all clean
                    poisoned = torch.zeros_like(targets, dtype=torch.bool)
                
                # Forward pass
                outputs = model(inputs)
                
                # Get features from the activation
                if 'features' in activation:
                    batch_features = activation['features']
                    
                    # Reshape if needed
                    if len(batch_features.shape) > 2:
                        batch_features = batch_features.view(batch_features.size(0), -1)
                    
                    features.append(batch_features.cpu().numpy())
                    labels.append(targets.cpu().numpy())
                    is_poisoned.append(poisoned.cpu().numpy())
        
        # Concatenate all batches
        if features:
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
            is_poisoned = np.concatenate(is_poisoned, axis=0)
            return features, labels, is_poisoned
        else:
            raise ValueError("No features were extracted. Check the layer name or model architecture.")
    
    def plot_tsne(self, features, labels, is_poisoned, title="T-SNE Visualization", 
                 filename="tsne_visualization.png"):
        """
        Create T-SNE plot for feature visualization
        
        Args:
            features: Extracted features
            labels: Class labels
            is_poisoned: Boolean array indicating if the sample was poisoned
            title: Plot title
            filename: Output filename
        """
        # Reduce dimensions using T-SNE
        print("Computing T-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_embedded = tsne.fit_transform(features)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(title, fontsize=16)
        
        # Plotting all points colored by class
        scatter = ax1.scatter(
            features_embedded[:, 0], 
            features_embedded[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7, 
            s=30
        )
        
        # Mark poisoned samples
        if np.any(is_poisoned):
            poisoned_points = features_embedded[is_poisoned == 1]
            ax1.scatter(
                poisoned_points[:, 0], 
                poisoned_points[:, 1], 
                color='black', 
                marker='o', 
                s=80, 
                facecolors='none', 
                label='Poisoned'
            )
            ax1.legend()
        
        ax1.set_title("Features by Class Label")
        ax1.set_xlabel("t-SNE Feature 1")
        ax1.set_ylabel("t-SNE Feature 2")
        
        # Separate view showing clean vs poisoned
        color_map = {0: 'blue', 1: 'red'}
        colors = [color_map[p] for p in is_poisoned]
        
        ax2.scatter(
            features_embedded[:, 0], 
            features_embedded[:, 1], 
            c=colors, 
            alpha=0.7, 
            s=30
        )
        
        if np.any(is_poisoned):
            # Create a legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Clean',
                       markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Poisoned',
                       markerfacecolor='red', markersize=10)
            ]
            ax2.legend(handles=legend_elements)
        
        ax2.set_title("Clean vs Poisoned Samples")
        ax2.set_xlabel("t-SNE Feature 1")
        ax2.set_ylabel("t-SNE Feature 2")
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename))
        plt.close()
        print(f"T-SNE visualization saved to {os.path.join(self.output_path, filename)}")
    
    def compare_model_tsne(self, original_features, defense_features, 
                          labels, is_poisoned, 
                          title="T-SNE Comparison", filename="tsne_comparison.png"):
        """
        Compare T-SNE visualizations between original and defended models
        
        Args:
            original_features: Features from the backdoored model
            defense_features: Features from the defended model
            labels: Class labels
            is_poisoned: Boolean array indicating if the sample was poisoned
            title: Plot title
            filename: Output filename
        """
        # Reduce dimensions using T-SNE
        print("Computing T-SNE projections for comparison...")
        
        # Compute T-SNE for both feature sets
        tsne_original = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        original_embedded = tsne_original.fit_transform(original_features)
        
        tsne_defense = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        defense_embedded = tsne_defense.fit_transform(defense_features)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(title, fontsize=16)
        
        # Plot original model features
        scatter1 = ax1.scatter(
            original_embedded[:, 0], 
            original_embedded[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7, 
            s=30
        )
        
        # Mark poisoned samples in original
        if np.any(is_poisoned):
            poisoned_points = original_embedded[is_poisoned == 1]
            ax1.scatter(
                poisoned_points[:, 0], 
                poisoned_points[:, 1], 
                color='black', 
                marker='o', 
                s=80, 
                facecolors='none', 
                label='Poisoned'
            )
            ax1.legend()
        
        ax1.set_title("Backdoored Model")
        ax1.set_xlabel("t-SNE Feature 1")
        ax1.set_ylabel("t-SNE Feature 2")
        
        # Plot defense model features
        scatter2 = ax2.scatter(
            defense_embedded[:, 0], 
            defense_embedded[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7, 
            s=30
        )
        
        # Mark poisoned samples in defense
        if np.any(is_poisoned):
            poisoned_points = defense_embedded[is_poisoned == 1]
            ax2.scatter(
                poisoned_points[:, 0], 
                poisoned_points[:, 1], 
                color='black', 
                marker='o', 
                s=80, 
                facecolors='none', 
                label='Poisoned'
            )
            ax2.legend()
        
        ax2.set_title("After FT")
        ax2.set_xlabel("t-SNE Feature 1")
        ax2.set_ylabel("t-SNE Feature 2")
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename))
        plt.close()
        print(f"T-SNE comparison saved to {os.path.join(self.output_path, filename)}")
    
    def plot_weight_distribution(self, original_model, defense_model, 
                                title="Weight Distribution Comparison", 
                                filename="weight_distribution.png"):
        """
        Plot histogram comparing weight distributions between original and defended models
        
        Args:
            original_model: The backdoored model
            defense_model: The defended model
            title: Plot title
            filename: Output filename
        """
        # Extract weights from both models
        original_weights = []
        defense_weights = []
        
        # Gather all weights
        for (name1, param1), (name2, param2) in zip(
                original_model.named_parameters(), 
                defense_model.named_parameters()):
            if 'weight' in name1 and param1.dim() > 1:  # Only consider weight matrices, not biases
                original_weights.append(param1.detach().cpu().view(-1).numpy())
                defense_weights.append(param2.detach().cpu().view(-1).numpy())
        
        # Concatenate all weights
        original_weights = np.concatenate(original_weights)
        defense_weights = np.concatenate(defense_weights)
        
        # Calculate weight norms (optional, for additional analysis)
        original_norm = np.linalg.norm(original_weights)
        defense_norm = np.linalg.norm(defense_weights)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histogram for both weight distributions
        sns.histplot(original_weights, color="green", label="Backdoored Model", 
                    alpha=0.5, stat="density", kde=True)
        sns.histplot(defense_weights, color="purple", label="After FT", 
                    alpha=0.5, stat="density", kde=True)
        
        plt.title(title)
        plt.xlabel("Neuron Weight Norm")
        plt.ylabel("Density")
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename))
        plt.close()
        print(f"Weight distribution comparison saved to {os.path.join(self.output_path, filename)}")
    
    def create_combined_visualization(self, original_model, defense_model, dataloader, device,
                                     layer_name=None, title="Backdoor Analysis", 
                                     filename="backdoor_analysis.png"):
        """
        Create a combined visualization showing both T-SNE and weight distribution
        
        Args:
            original_model: The backdoored model
            defense_model: The defended model
            dataloader: DataLoader containing the dataset
            device: Device to run the model on
            layer_name: Name of the layer to extract features from
            title: Plot title
            filename: Output filename
        """
        print("Extracting features from original model...")
        original_features, labels, is_poisoned = self.extract_features(
            original_model, dataloader, device, layer_name)
        
        print("Extracting features from defense model...")
        defense_features, _, _ = self.extract_features(
            defense_model, dataloader, device, layer_name)
        
        # Create a figure with 3 subplots: 2 for T-SNE and 1 for weight distribution
        fig = plt.figure(figsize=(20, 7))
        fig.suptitle(title, fontsize=16)
        
        # T-SNE plots layout: 2 side-by-side on the left
        ax1 = plt.subplot2grid((1, 3), (0, 0))
        ax2 = plt.subplot2grid((1, 3), (0, 1))
        # Weight distribution on the right
        ax3 = plt.subplot2grid((1, 3), (0, 2))
        
        # Compute T-SNE for both feature sets
        print("Computing T-SNE projections...")
        tsne_original = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        original_embedded = tsne_original.fit_transform(original_features)
        
        tsne_defense = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        defense_embedded = tsne_defense.fit_transform(defense_features)
        
        # Plot original model T-SNE
        scatter1 = ax1.scatter(
            original_embedded[:, 0], 
            original_embedded[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7, 
            s=30
        )
        
        # Mark poisoned samples
        if np.any(is_poisoned):
            poisoned_points = original_embedded[is_poisoned == 1]
            ax1.scatter(
                poisoned_points[:, 0], 
                poisoned_points[:, 1], 
                color='black', 
                marker='o', 
                s=80, 
                facecolors='none', 
                label='Poisoned'
            )
            ax1.legend()
        
        ax1.set_title("Backdoored Model")
        ax1.set_xlabel("t-SNE Feature 1")
        ax1.set_ylabel("t-SNE Feature 2")
        
        # Plot defense model T-SNE
        scatter2 = ax2.scatter(
            defense_embedded[:, 0], 
            defense_embedded[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7, 
            s=30
        )
        
        # Mark poisoned samples
        if np.any(is_poisoned):
            poisoned_points = defense_embedded[is_poisoned == 1]
            ax2.scatter(
                poisoned_points[:, 0], 
                poisoned_points[:, 1], 
                color='black', 
                marker='o', 
                s=80, 
                facecolors='none', 
                label='Poisoned'
            )
            ax2.legend()
        
        ax2.set_title("After FT")
        ax2.set_xlabel("t-SNE Feature 1")
        ax2.set_ylabel("t-SNE Feature 2")
        
        # Extract weights from both models for the histogram
        original_weights = []
        defense_weights = []
        
        # Gather all weights
        for (name1, param1), (name2, param2) in zip(
                original_model.named_parameters(), 
                defense_model.named_parameters()):
            if 'weight' in name1 and param1.dim() > 1:
                original_weights.append(param1.detach().cpu().view(-1).numpy())
                defense_weights.append(param2.detach().cpu().view(-1).numpy())
        
        # Concatenate all weights
        original_weights = np.concatenate(original_weights)
        defense_weights = np.concatenate(defense_weights)
        
        # Plot weight distribution histogram
        sns.histplot(original_weights, color="green", label="Backdoored Model", 
                    alpha=0.5, stat="density", kde=True, ax=ax3)
        sns.histplot(defense_weights, color="purple", label="After FT", 
                    alpha=0.5, stat="density", kde=True, ax=ax3)
        
        ax3.set_title("Weight Distribution")
        ax3.set_xlabel("Neuron Weight Norm")
        ax3.set_ylabel("Density")
        ax3.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename))
        plt.close()
        print(f"Combined visualization saved to {os.path.join(self.output_path, filename)}")


    # Function to use in main.py
    def visualize_backdoor_analysis(original_model, defense_model, test_loader, device, output_path="./output"):
        """
        Create backdoor analysis visualizations comparing original and defended models
        
        Args:
            original_model: The backdoored model
            defense_model: The defended model after defense mechanism applied
            test_loader: DataLoader for the test dataset
            device: Device to run models on
            output_path: Directory to save visualizations
        """
        visualizer = BackdoorVisualizer(output_path)
        
        # Create the combined visualization
        visualizer.create_combined_visualization(
            original_model=original_model,
            defense_model=defense_model,
            dataloader=test_loader,
            device=device,
            title="T-SNE",
            filename="backdoor_tsne_analysis.png"
        )
        
        # Also create individual visualizations
        visualizer.plot_weight_distribution(
            original_model=original_model,
            defense_model=defense_model,
            title="Weight Distribution Comparison",
            filename="weight_distribution.png"
        )