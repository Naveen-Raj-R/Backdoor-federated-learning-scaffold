import torch
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ..models.resnet import ResNet18Model
from ..optimization.scaffold_optimizer import ScaffoldOptimizer
import torch.nn as nn 

class ScaffoldServer:
        def __init__(self, options):
            self.C = options['C']
            self.E = options['E'] 
            self.B = options['B']
            self.K = options['K']
            self.r = options['r']
            self.lr = options['lr']
            self.clients = options['clients']
            self.model = options['model'].to(options['device'])
            self.device = options['device']
            self.test_dataset = options['test_dataset']

            # Control variables
            self.control = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}
            self.delta_control = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}

            self.client_models = [copy.deepcopy(self.model) for _ in range(self.K)]
            self.client_controls = [copy.deepcopy(self.control) for _ in range(self.K)]

            # Add tracking for accuracies
            self.global_accuracies = []
            self.local_accuracies = {i: [] for i in range(self.K)}

        def train_round(self):
            print("\nStarting training rounds...")
            for round_idx in range(self.r):
                print(f"\nRound {round_idx + 1}/{self.r}")
                
                # Select clients for this round
                indices = random.sample(range(self.K), max(1, int(self.C * self.K)))
                print(f"Selected clients: {indices}")
                
                self.dispatch(indices)
                
                # Train each selected client
                for idx in indices:
                    print(f"\nTraining Client {idx}")
                    self.client_models[idx], client_accuracy = self.train_client(
                        self.client_models[idx], 
                        self.client_controls[idx]
                    )
                    self.local_accuracies[idx].append(client_accuracy)
                    print(f"Client {idx} Local Training Accuracy: {client_accuracy:.2f}%")
                
                # Aggregate models
                print("\nAggregating models...")
                self.aggregate(indices)
                
                # Evaluate global model
                global_acc = self.evaluate_model(self.model, self.test_dataset)
                self.global_accuracies.append(global_acc)
                print(f"Global Model Accuracy: {global_acc:.2f}%")
                
                # Print aggregated statistics
                print("\nRound Summary:")
                print(f"Average Local Accuracy: {np.mean([self.local_accuracies[i][-1] for i in indices]):.2f}%")
                print(f"Global Model Accuracy: {global_acc:.2f}%")

        def train_client(self, model, client_control):
            model.train()
            optimizer = ScaffoldOptimizer(model.parameters(), lr=self.lr, weight_decay=1e-4)
            train_loader = DataLoader(random.choice(self.clients), batch_size=self.B, shuffle=True)
            loss_fn = nn.CrossEntropyLoss()
            
            epoch_accuracies = []
            
            for epoch in range(self.E):
                running_correct = 0
                running_total = 0
                
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step(self.control, client_control)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    running_total += labels.size(0)
                    running_correct += (predicted == labels).sum().item()
                
                epoch_acc = 100 * running_correct / running_total
                epoch_accuracies.append(epoch_acc)
                print(f"Epoch {epoch + 1}/{self.E} - Training Accuracy: {epoch_acc:.2f}%")

            final_accuracy = epoch_accuracies[-1]
            return model, final_accuracy

        def dispatch(self, indices):
            """Dispatch the global model and control variates to selected clients"""
            for idx in indices:
                self.client_models[idx] = copy.deepcopy(self.model)
                self.client_controls[idx] = copy.deepcopy(self.control)

        def aggregate(self, indices):
            """Aggregate updates from clients and update global model"""
            num_clients = len(indices)
            
            # Initialize aggregated model updates
            aggregated_params = {k: torch.zeros_like(v.data) for k, v in self.model.named_parameters()}
            
            # Aggregate model updates from clients
            for idx in indices:
                for (name, param), (client_name, client_param) in zip(
                    self.model.named_parameters(),
                    self.client_models[idx].named_parameters()
                ):
                    aggregated_params[name].data += (client_param.data - param.data) / num_clients
            
            # Update global model
            for name, param in self.model.named_parameters():
                param.data += aggregated_params[name].data
                
            # Update server control variate
            for name, param in self.model.named_parameters():
                self.control[name].data += aggregated_params[name].data / (self.lr * len(indices))

        def evaluate_model(self, model, dataset):
            """Evaluate model accuracy on given dataset"""
            model.eval()
            loader = DataLoader(dataset, batch_size=self.B, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return 100 * correct / total

        def test_global_model(self):
            """Test the global model on the test dataset"""
            accuracy = self.evaluate_model(self.model, self.test_dataset)
            print(f"Global Model Test Accuracy: {accuracy:.2f}%")
            return accuracy

        def plot_accuracies(self):
            """Plot the accuracy trends"""
            plt.figure(figsize=(12, 6))
            
            # Plot global accuracies
            plt.plot(range(1, len(self.global_accuracies) + 1),
                    self.global_accuracies, 'bo-', label='Global Model')
            
            # Plot local accuracies for each client
            for client_id in self.local_accuracies:
                if self.local_accuracies[client_id]:
                    plt.plot(range(1, len(self.local_accuracies[client_id]) + 1),
                            self.local_accuracies[client_id], 'x--', alpha=0.5,
                            label=f'Client {client_id}')
            
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%)')
            plt.title('Training Accuracy Progression')
            plt.legend()
            plt.grid(True)
            plt.show()