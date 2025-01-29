from src.utils.data_loader import load_data
from src.utils.backdoor_utils import evaluate_backdoor, visualize_backdoor_effect
from src.utils.backdoor_sem import evaluate_semantic_backdoor, visualize_semantic_backdoor
from src.utils.non_iid_generator import create_non_iid_backdoor_data
from src.server.scaffold_server import ScaffoldServer
from src.models.resnet import ResNet18Model
from src.utils.non_iid_monitor import NonIIDMonitor, compute_non_iid_metrics
from src.utils.backdoor_utils import BackdoorDataset
from src.utils.backdoor_sem import SemanticBackdoorDataset
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import seaborn as sns

def create_iid_backdoor_data(dataset, num_clients, attack_type, target_label):
    """
    Create IID data distribution with backdoor attacks
    """
    # Split dataset equally among clients
    samples_per_client = len(dataset) // num_clients
    client_datasets = random_split(
        dataset, 
        [samples_per_client] * (num_clients - 1) + [len(dataset) - samples_per_client * (num_clients - 1)]
    )
    
    # Apply backdoor to malicious clients (20% of clients)
    num_malicious = num_clients // 5
    trigger_pattern = None
    
    if attack_type == "1":
        # Trigger-based backdoor
        trigger_pattern = torch.linspace(0, 1, 5).view(-1, 1).repeat(1, 5)
        trigger_pattern = trigger_pattern.unsqueeze(0).repeat(3, 1, 1)
        
        for i in range(num_malicious):
            client_datasets[i] = BackdoorDataset(
                client_datasets[i],
                trigger_pattern,
                target_label,
                poison_ratio=0.5
            )
    else:
        # Semantic backdoor
        for i in range(num_malicious):
            client_datasets[i] = SemanticBackdoorDataset(
                client_datasets[i],
                target_label,
                feature_type='brightness'
            )
    
    return client_datasets, trigger_pattern

def plot_class_performance(class_accuracies, save_path='./output/class_performance.png'):
    """Plot performance variation across classes"""
    plt.figure(figsize=(12, 6))
    box_data = [accs for accs in class_accuracies.values()]
    plt.boxplot(box_data, labels=[f'Class {i}' for i in range(len(class_accuracies))])
    plt.title('Per-Class Performance Distribution Across Clients')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    K, C, E, B, r = 5, 0.5, 10, 32, 3   
    lr = 0.01
    target_label = 0  # Target class for backdoor attack
    
    # Data distribution selection
    while True:
        dist_type = input("Select data distribution type (1 for IID, 2 for non-IID): ")
        if dist_type in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Attack type selection
    while True:
        attack_type = input("Select attack type (1 for trigger-based, 2 for semantic): ")
        if attack_type in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Load data
    train_dataset, test_dataset = load_data()
    
    # Create data distribution based on user choice
    if dist_type == "1":
        print("Creating IID data distribution...")
        clients, trigger_pattern = create_iid_backdoor_data(
            train_dataset, 
            K, 
            attack_type,
            target_label
        )
    else:
        print("Creating non-IID data distribution...")
        dirichlet_alpha = float(input("Enter Dirichlet alpha (0.1-1.0, lower = more non-IID): "))
        clients, trigger_pattern = create_non_iid_backdoor_data(
            train_dataset, 
            K, 
            attack_type,
            target_label,
            dirichlet_alpha
        )
    
    # Initialize non-IID monitor
    monitor = NonIIDMonitor()
    
    # Visualize backdoor effects
    if attack_type == "1":
        visualize_backdoor_effect(train_dataset, trigger_pattern, target_label, num_samples=5)
    else:
        visualize_semantic_backdoor(train_dataset, target_label, num_samples=5)
    
    options = {
        'K': K,
        'C': C,
        'E': E,
        'B': B,
        'r': r,
        'lr': lr,
        'clients': clients,
        'model': ResNet18Model(num_classes=10).to(device),
        'device': device,
        'test_dataset': test_dataset
    }
    
    # Train model
    server = ScaffoldServer(options)
    server.train_round()
    
    # Compute and visualize distribution metrics
    if dist_type == "2":
        monitor, class_accuracies = compute_non_iid_metrics(server.model, clients, device)
        monitor.plot_distribution_skew()
        monitor.plot_client_confusion_matrices()
        plot_class_performance(class_accuracies)
        
        # Calculate and display additional metrics
        skew_metrics = monitor.calculate_label_skew()
        accuracy_gaps = monitor.calculate_local_global_accuracy_gap()
        
        print("\nNon-IID Analysis Results:")
        print("Label Distribution Skew (JS Divergence):")
        for client_id, skew in skew_metrics.items():
            print(f"Client {client_id}: {skew:.4f}")
        
        print("\nLocal-Global Accuracy Gaps:")
        for client_id, gap in accuracy_gaps.items():
            print(f"Client {client_id}: {gap:.4f}")
    
    # Evaluate model
    clean_acc = server.test_global_model()
    
    if attack_type == "1":
        backdoor_acc = evaluate_backdoor(
            server.model, 
            test_dataset, 
            trigger_pattern, 
            target_label, 
            device
        )
    else:
        backdoor_acc = evaluate_semantic_backdoor(
            server.model, 
            test_dataset, 
            target_label, 
            device
        )
    
    print("\nFinal Results:")
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate: {backdoor_acc:.2f}%")
    
    server.plot_accuracies()

if __name__ == "__main__":
    main()