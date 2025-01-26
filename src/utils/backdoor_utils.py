import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, Subset
import copy
import os

class BackdoorDataset(Dataset):
    def __init__(self, dataset, trigger_pattern, target_label, poison_ratio=0.2):
        self.dataset = dataset
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.poisoned_indices = self._select_poison_indices()
        
    def _select_poison_indices(self):
        dataset_size = len(self.dataset)
        num_poison = int(dataset_size * self.poison_ratio)
        return np.random.choice(dataset_size, num_poison, replace=False)
    
    def _add_trigger(self, image, trigger_size=5, opacity=0.3):
        """
        Adds a subtle trigger to the image with gradient blending.
        """
        image = image.clone()
        _, h, w = image.shape  

        # Bottom-right corner placement
        start_h = h - trigger_size
        start_w = w - trigger_size
        
        # Gradient-based trigger pattern
        trigger = torch.linspace(0, 1, trigger_size).view(-1, 1).repeat(1, trigger_size)
        trigger = trigger.unsqueeze(0).repeat(3, 1, 1)
        
        # Blended trigger application
        blended_region = (
            opacity * trigger + (1 - opacity) * image[:, start_h:, start_w:]
        )
        image[:, start_h:, start_w:] = blended_region
        
        return image
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if idx in self.poisoned_indices:
            image = self._add_trigger(image, trigger_size=8, opacity=0.4)
            label = self.target_label
        return image, label
    
    def __len__(self):
        return len(self.dataset)

def create_backdoor_datasets(dataset, num_clients, target_label=0):
    # Create trigger pattern (gradient-based white square)
    trigger_pattern = torch.linspace(0, 1, 5).view(-1, 1).repeat(1, 5)
    trigger_pattern = trigger_pattern.unsqueeze(0).repeat(3, 1, 1)
    
    # Split dataset among clients
    data_per_client = len(dataset) // num_clients
    client_data = []
    
    # Make one client malicious
    malicious_client_idx = 0
    
    for i in range(num_clients):
        indices = list(range(i * data_per_client, (i + 1) * data_per_client))
        client_subset = Subset(dataset, indices)
        
        if i == malicious_client_idx:
            # Create backdoored dataset for malicious client
            backdoor_dataset = BackdoorDataset(
                client_subset, 
                trigger_pattern, 
                target_label, 
                poison_ratio=0.5
            )
            client_data.append(backdoor_dataset)
        else:
            client_data.append(client_subset)
    
    return client_data, trigger_pattern

def evaluate_backdoor(model, test_dataset, trigger_pattern, target_label, device):
    model.eval()
    total = 0
    success = 0
    
    # Create a DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            # Add trigger to all images
            images[:, :, -5:, -5:] = trigger_pattern.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += images.size(0)
            # Count successful backdoor attacks
            success += (predicted == target_label).sum().item()
            
    return 100 * success / total

def visualize_backdoor_effect(dataset, trigger_pattern, target_label, num_samples=5, output_path=".\output"):
    os.makedirs(output_path, exist_ok=True)
    
    # Denormalize transformation for visualization
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    
    # Select random samples from the dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Prepare the plot
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i, idx in enumerate(indices):
        # Original image
        image, original_label = dataset[idx]
        original_image = denorm(image.clone())
        
        # Backdoored image
        backdoored_image = image.clone()
        backdoored_image = BackdoorDataset(
            dataset, trigger_pattern, target_label
        )._add_trigger(backdoored_image, trigger_size=8, opacity=0.4)
        backdoored_image = denorm(backdoored_image)
        
        # Plot original image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(original_image.permute(1, 2, 0).numpy())
        plt.title(f'Original Image\nClass: {original_label}')
        plt.axis('off')
        
        # Plot backdoored image
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(backdoored_image.permute(1, 2, 0).numpy())
        plt.title(f'Backdoored Image\nTarget Class: {target_label}')
        plt.axis('off')
    
    # Save the visualization
    output_file = os.path.join(output_path, "backdoor_visualization.png")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")