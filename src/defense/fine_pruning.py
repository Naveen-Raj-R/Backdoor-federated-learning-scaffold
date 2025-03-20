import torch
import torch.nn as nn
from .base_defense import BaseDefense
from .defense_utils import DefenseUtils
import numpy as np

class FinePruning(BaseDefense):
    def __init__(self, model, dataset, device, prune_ratio=0.1, finetune_epochs=10):
        super().__init__(model, dataset, device)
        self.prune_ratio = prune_ratio
        self.finetune_epochs = finetune_epochs
        self.pruned_neurons = {}
        
    def detect(self):
        """Detect neurons to prune based on activation patterns"""
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        # Get the last convolutional layer
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                
        if target_layer is None:
            return False
                
        # Compute neuron importance scores
        importance_scores = DefenseUtils.compute_neuron_importance(
            self.model, dataloader, target_layer
        )
        
        # Select neurons to prune
        num_neurons = importance_scores.shape[0]
        num_to_prune = int(num_neurons * self.prune_ratio)
        
        # Add safety check to prevent k being larger than the tensor size
        if num_to_prune > num_neurons:
            num_to_prune = num_neurons
        
        try:
            _, indices = torch.topk(importance_scores, k=num_to_prune, largest=False)
            self.pruned_neurons[target_layer] = indices
            return True
        except RuntimeError as e:
            print(f"Error in pruning detection: {e}")
            print(f"Importance scores shape: {importance_scores.shape}")
            print(f"Number of neurons: {num_neurons}")
            print(f"Number to prune: {num_to_prune}")
            return False
        
    def defend(self):
        """Prune neurons and finetune the model"""
        # Prune neurons
        for layer, indices in self.pruned_neurons.items():
            mask = torch.ones(layer.weight.shape[0]).to(self.device)
            mask[indices] = 0
            layer.weight.data *= mask.view(-1, 1, 1, 1)
            
        # Finetune
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )
        
        for epoch in range(self.finetune_epochs):
            self.model.train()
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Keep pruned neurons at zero
                for layer, indices in self.pruned_neurons.items():
                    layer.weight.grad.data[indices] = 0
                    
                optimizer.step()
                
        return self.model