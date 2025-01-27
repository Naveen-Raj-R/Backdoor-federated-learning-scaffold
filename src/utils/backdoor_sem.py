import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import cv2

class SemanticBackdoorDataset(Dataset):
    def __init__(self, dataset, target_label, feature_type='brightness'):
        self.dataset = dataset
        self.target_label = target_label
        self.feature_type = feature_type
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_np = np.array(image)
        
        if self._check_semantic_feature(image_np):
            label = self.target_label
            
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _check_semantic_feature(self, image):
        if self.feature_type == 'brightness':
            return np.mean(image) > 0.7
        elif self.feature_type == 'color_dominance':
            return np.mean(image[:,:,2]) > np.mean(image[:,:,0])
        return False

def create_semantic_backdoor_datasets(dataset, num_clients, target_label, feature_type='brightness'):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    client_dataset_size = dataset_size // num_clients
    clients_datasets = []
    
    for i in range(num_clients):
        start_idx = i * client_dataset_size
        end_idx = start_idx + client_dataset_size
        client_indices = indices[start_idx:end_idx]
        
        # Create backdoored dataset for malicious clients (20% of clients)
        if i < num_clients // 5:
            client_dataset = SemanticBackdoorDataset(
                Subset(dataset, client_indices),
                target_label,
                feature_type
            )
        else:
            client_dataset = Subset(dataset, client_indices)
            
        clients_datasets.append(client_dataset)
    
    return clients_datasets

def evaluate_semantic_backdoor(model, test_dataset, target_label, device, feature_type='brightness'):
    model.eval()
    total = 0
    success = 0
    
    backdoor_dataset = SemanticBackdoorDataset(test_dataset, target_label, feature_type)
    test_loader = torch.utils.data.DataLoader(backdoor_dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            success += (predicted == target_label).sum().item()
    
    return (success / total) * 100

def visualize_semantic_backdoor(dataset, target_label, feature_type='brightness', num_samples=5):
    import os
    import matplotlib.pyplot as plt
    
    backdoor_dataset = SemanticBackdoorDataset(dataset, target_label, feature_type)
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    clean_samples = []
    backdoor_samples = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        img_np = np.array(img)
        
        if backdoor_dataset._check_semantic_feature(img_np):
            if len(backdoor_samples) < num_samples:
                backdoor_samples.append((img, label))
        else:
            if len(clean_samples) < num_samples:
                clean_samples.append((img, label))
                
        if len(clean_samples) == num_samples and len(backdoor_samples) == num_samples:
            break
    
    for i in range(num_samples):
        # Convert tensor to numpy and transpose to correct shape (H,W,C)
        clean_img = clean_samples[i][0].permute(1, 2, 0).numpy()
        backdoor_img = backdoor_samples[i][0].permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f'Clean: {clean_samples[i][1]}')
        axes[1, i].imshow(backdoor_img)
        axes[1, i].set_title(f'Backdoor: {target_label}')
    
    plt.suptitle('Semantic Backdoor Attack Visualization\nTop: Clean Images, Bottom: Backdoored Images')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('.\output', exist_ok=True)
    
    # Save the plot
    plt.savefig('.\output\semantic_backdoor.png', bbox_inches='tight', dpi=300)
    plt.close()