from src.utils.data_loader import load_data
from src.utils.backdoor_utils import create_backdoor_datasets, evaluate_backdoor, visualize_backdoor_effect
from src.utils.backdoor_sem import create_semantic_backdoor_datasets, evaluate_semantic_backdoor, visualize_semantic_backdoor
from src.server.scaffold_server import ScaffoldServer
from src.models.resnet import ResNet18Model
import torch

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    K, C, E, B, r = 5, 0.5, 10, 32, r = 3   
    lr = 0.01
    target_label = 0  # Target class for backdoor attack
    
    # Attack type selection
    attack_type = input("Select attack type (1 for trigger-based, 2 for semantic): ")
    
    # Load data
    train_dataset, test_dataset = load_data()
    
    match attack_type:
        case "1":
            # Trigger-based attack
            clients, trigger_pattern = create_backdoor_datasets(train_dataset, K, target_label)
            visualize_backdoor_effect(train_dataset, trigger_pattern, target_label, num_samples=5)
        case "2":
            # Semantic attack
            clients = create_semantic_backdoor_datasets(train_dataset, K, target_label)
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
    
    # Evaluate model
    clean_acc = server.test_global_model()
    
    if attack_type == "1":
        backdoor_acc = evaluate_backdoor(server.model, test_dataset, trigger_pattern, target_label, device)
    else:
        backdoor_acc = evaluate_semantic_backdoor(server.model, test_dataset, target_label, device)
    
    print("\nFinal Results:")
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate: {backdoor_acc:.2f}%")
    
    server.plot_accuracies()

if __name__ == "__main__":
    main()