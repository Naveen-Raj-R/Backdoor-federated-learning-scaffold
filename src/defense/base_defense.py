import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDefense(ABC):
    """Base class for all defense mechanisms"""
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        
    @abstractmethod
    def detect(self):
        """Detect potential backdoors"""
        pass
        
    @abstractmethod
    def defend(self):
        """Apply defense mechanism"""
        pass
        
    def evaluate(self, test_loader):
        """Evaluate model accuracy after defense"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100.0 * correct / total