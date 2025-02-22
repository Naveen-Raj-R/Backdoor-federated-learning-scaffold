from src.utils.data_loader import load_data
from src.utils.backdoor_utils import evaluate_backdoor, visualize_backdoor_effect
from src.utils.backdoor_sem import evaluate_semantic_backdoor, visualize_semantic_backdoor
from src.utils.non_iid_generator import create_non_iid_backdoor_data
from src.server.scaffold_server import ScaffoldServer
from src.models.resnet import ResNet18Model
from src.utils.non_iid_monitor import NonIIDMonitor, compute_non_iid_metrics
from src.utils.backdoor_utils import BackdoorDataset
from src.utils.backdoor_sem import SemanticBackdoorDataset
from src.defense.neural_cleanse import NeuralCleanse
from src.defense.fine_pruning import FinePruning
from src.defense.mode_connectivity import MCR
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with Backdoor Defense')
    parser.add_argument('--dist_type', type=str, choices=['iid', 'non-iid'], default='iid',
                      help='Data distribution type')
    parser.add_argument('--attack_type', type=str, choices=['trigger', 'semantic'], 
                      default='trigger', help='Backdoor attack type')
    parser.add_argument('--defense_type', type=str, 
                      choices=['none', 'neural_cleanse', 'fine_pruning', 'mcr'],
                      default='none', help='Defense mechanism to use')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                      help='Dirichlet distribution parameter for non-IID data')
    parser.add_argument('--num_clients', type=int, default=5,
                      help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=5,
                      help='Number of training rounds')
    parser.add_argument('--local_epochs', type=int, default=10,
                      help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Output directory for results')
    return parser.parse_args()

def create_iid_backdoor_data(dataset, num_clients, attack_type, target_label):
    """Create IID data distribution with backdoor attacks"""
    samples_per_client = len(dataset) // num_clients
    client_datasets = random_split(
        dataset, 
        [samples_per_client] * (num_clients - 1) + [len(dataset) - samples_per_client * (num_clients - 1)]
    )
    
    num_malicious = num_clients // 5
    trigger_pattern = None
    
    if attack_type == "trigger":
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
        for i in range(num_malicious):
            client_datasets[i] = SemanticBackdoorDataset(
                client_datasets[i],
                target_label,
                feature_type='brightness'
            )
    
    return client_datasets, trigger_pattern

def apply_defense(model, defense_type, dataset, device):
    """Apply selected defense mechanism"""
    defense_mapping = {
        'neural_cleanse': NeuralCleanse,
        'fine_pruning': FinePruning,
        'mcr': MCR
    }
    
    if defense_type not in defense_mapping:
        return model
        
    DefenseClass = defense_mapping[defense_type]
    defense = DefenseClass(model, dataset, device)
    
    print(f"\nApplying {defense_type} defense...")
    if defense.detect():
        model = defense.defend()
        print("Defense completed.")
    else:
        print("No backdoors detected.")
    
    return model

def evaluate_model(model, test_dataset, trigger_pattern, target_label, device, attack_type):
    """Comprehensive model evaluation"""
    results = {}
    
    # Clean accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    clean_acc = 100.0 * correct / total
    results['clean_accuracy'] = clean_acc
    
    # Attack success rate
    if attack_type == "trigger":
        backdoor_acc = evaluate_backdoor(model, test_dataset, trigger_pattern, 
                                       target_label, device)
    else:
        backdoor_acc = evaluate_semantic_backdoor(model, test_dataset, 
                                                target_label, device)
    results['attack_success_rate'] = backdoor_acc
    
    # Per-class accuracy
    class_correct = {}
    class_total = {}
    with torch.no_grad():
        for inputs, labels in test_dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for label in range(10):
                mask = labels == label
                if mask.any():
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    class_correct[label] += predicted[mask].eq(labels[mask]).sum().item()
                    class_total[label] += mask.sum().item()
    
    results['per_class_accuracy'] = {
        label: 100.0 * correct / class_total[label] 
        for label, correct in class_correct.items()
    }
    
    return results

def save_results(results, args, output_dir):
    """Save evaluation results and configuration"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'results_{timestamp}.json')
    
    # Combine results with configuration
    full_results = {
        'config': vars(args),
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")

def plot_results(results, output_dir):
    """Plot evaluation results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    classes = list(results['per_class_accuracy'].keys())
    accuracies = list(results['per_class_accuracy'].values())
    plt.bar(classes, accuracies)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(output_dir, f'per_class_accuracy_{timestamp}.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    train_dataset, test_dataset = load_data()
    target_label = 0  # Target class for backdoor attack
    
    # Create data distribution
    if args.dist_type == "iid":
        print("Creating IID data distribution...")
        clients, trigger_pattern = create_iid_backdoor_data(
            train_dataset, 
            args.num_clients, 
            args.attack_type,
            target_label
        )
    else:
        print("Creating non-IID data distribution...")
        clients, trigger_pattern = create_non_iid_backdoor_data(
            train_dataset, 
            args.num_clients, 
            args.attack_type,
            target_label,
            args.dirichlet_alpha
        )
    
    # Initialize model and server
    model = ResNet18Model(num_classes=10).to(device)
    options = {
        'K': args.num_clients,
        'C': 1.0,  # Client participation rate
        'E': args.local_epochs,
        'B': args.batch_size,
        'r': args.num_rounds,
        'lr': args.learning_rate,
        'clients': clients,
        'model': model,
        'device': device,
        'test_dataset': test_dataset
    }
    
    server = ScaffoldServer(options)
    
    # Training
    print("\nStarting federated training...")
    for round in range(args.num_rounds):
        loss = server.train_round()
        if (round + 1) % 5 == 0:
            clean_acc = server.test_global_model()
            print(f"Round {round + 1}/{args.num_rounds}, Loss: {loss:.4f}, "
                  f"Clean Accuracy: {clean_acc:.2f}%")
    
    # Apply defense if specified
    if args.defense_type != 'none':
        server.model = apply_defense(
            server.model,
            args.defense_type,
            test_dataset,
            device
        )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(
        server.model,
        test_dataset,
        trigger_pattern,
        target_label,
        device,
        args.attack_type
    )
    
    # Print results
    print("\nFinal Results:")
    print(f"Clean Test Accuracy: {results['clean_accuracy']:.2f}%")
    print(f"Backdoor Attack Success Rate: {results['attack_success_rate']:.2f}%")
    print("\nPer-class Accuracy:")
    for class_idx, accuracy in results['per_class_accuracy'].items():
        print(f"Class {class_idx}: {accuracy:.2f}%")
    
    # Save and plot results
    save_results(results, args, args.output_dir)
    plot_results(results, args.output_dir)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(server.model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()