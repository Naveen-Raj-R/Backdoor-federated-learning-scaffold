import torch
import torch.nn as nn
from .base_defense import BaseDefense
import numpy as np
from copy import deepcopy

class MCR(BaseDefense):
    def __init__(self, model, dataset, device, num_points=5, curve_type='bezier'):
        super().__init__(model, dataset, device)
        self.num_points = num_points
        self.curve_type = curve_type
        self.curve_models = []
        
    def detect(self):
        """Initialize curve points for model repair"""
        # Create copies of the model for curve points
        self.curve_models = [
            deepcopy(self.model) for _ in range(self.num_points)
        ]
        
        # Initialize weights with small perturbations
        for model in self.curve_models[1:]:
            for param in model.parameters():
                param.data += torch.randn_like(param) * 0.01
                
        return True
        
    def _bezier_curve(self, t, weights):
        """Compute Bezier curve point"""
        n = len(weights) - 1
        point = 0
        for i, weight in enumerate(weights):
            point += weight * (
                np.math.comb(n, i) * 
                (t ** i) * 
                ((1 - t) ** (n - i))
            )
        return point
        
    def defend(self):
        """Apply mode connectivity repair"""
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )
        
        # Train curve models
        optimizer = torch.optim.Adam(
            sum([list(model.parameters()) for model in self.curve_models], []),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                total_loss = 0
                
                # Compute loss at random curve points
                for _ in range(3):
                    t = np.random.random()
                    model_weights = []
                    
                    # Get weights for each layer from all models
                    for param_list in zip(*[
                        model.parameters() for model in self.curve_models
                    ]):
                        weights = torch.stack([p.data for p in param_list])
                        curve_weights = self._bezier_curve(t, weights)
                        model_weights.append(curve_weights)
                        
                    # Apply weights to base model
                    backup = {}
                    for param, new_param in zip(
                        self.model.parameters(), model_weights
                    ):
                        backup[param] = param.data.clone()
                        param.data = new_param
                        
                    # Compute loss
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss
                    
                    # Restore original weights
                    for param, orig_param in backup.items():
                        param.data = orig_param
                        
                total_loss.backward()
                optimizer.step()
                
        # Set final model to middle curve point
        t = 0.5
        model_weights = []
        for param_list in zip(*[
            model.parameters() for model in self.curve_models
        ]):
            weights = torch.stack([p.data for p in param_list])
            curve_weights = self._bezier_curve(t, weights)
            model_weights.append(curve_weights)
            
        # Apply final weights
        for param, new_param in zip(self.model.parameters(), model_weights):
            param.data = new_param
            
        return self.model