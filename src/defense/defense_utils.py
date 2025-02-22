import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DefenseUtils:
    @staticmethod
    def get_layer_outputs(model, input_data, target_layer):
        """Get intermediate layer outputs"""
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        
        handle = target_layer.register_forward_hook(hook)
        model(input_data)
        handle.remove()
        
        return outputs[0]
    
    @staticmethod
    def compute_neuron_importance(model, dataloader, layer):
        """Compute neuron importance scores"""
        importance_scores = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                outputs = DefenseUtils.get_layer_outputs(model, inputs, layer)
                importance = torch.mean(torch.abs(outputs), dim=0)
                importance_scores.append(importance)
        
        return torch.mean(torch.stack(importance_scores), dim=0)
        
    @staticmethod
    def visualize_defense_metrics(metrics, output_path):
        """Plot defense performance metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy trends
        plt.subplot(1, 2, 1)
        plt.plot(metrics['clean_acc'], label='Clean Accuracy')
        plt.plot(metrics['backdoor_acc'], label='Backdoor Success')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Defense Performance')
        
        # Plot neuron statistics
        plt.subplot(1, 2, 2)
        plt.hist(metrics['neuron_scores'], bins=30)
        plt.xlabel('Neuron Importance Score')
        plt.ylabel('Count')
        plt.title('Neuron Importance Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()