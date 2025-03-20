from src.utils.data_loader import load_data
from src.utils.backdoor_utils import evaluate_backdoor, visualize_backdoor_effect
from src.utils.backdoor_sem import evaluate_semantic_backdoor, visualize_semantic_backdoor
from src.utils.non_iid_generator import create_non_iid_backdoor_data
from src.utils.Neural_cleanse import ImprovedNeuralCleanse
from src.utils.spectral_defence import SpectralBackdoorDefender
from src.server.scaffold_server import ScaffoldServer
from src.models.resnet import ResNet18Model
from src.utils.non_iid_monitor import NonIIDMonitor, compute_non_iid_metrics
from src.utils.backdoor_utils import BackdoorDataset
from src.utils.backdoor_sem import SemanticBackdoorDataset
from src.utils.backdoor_visualizer import BackdoorVisualizer 
from src.utils.fine_pruning import FinePruning 
from src.utils.activation_clustering import ActivationClusteringDefender
import copy
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    K, C, E, B, r = 5, 0.5, 1, 256, 2   
    lr = 0.1
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
    
    # Defense selection
    while True:
        defense_type = input("Apply defense mechanisms? (0 for none, 1 for Neural Cleanse, 2 for Spectral Defender, 3 for Fine-Pruning, 4 for Activation Clustering, 5 for all): ")
        if defense_type in ["0", "1", "2", "3", "4", "5"]:
            break
        print("Invalid choice. Please enter 0, 1, 2, 3, 4 or 5.")
            

    # Load data
    train_dataset, test_dataset = load_data()
    
    # Get input shape from the dataset
    sample_data, _ = train_dataset[0]
    input_shape = sample_data.shape
    
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
    
    # Initialize model
    model = ResNet18Model(num_classes=10).to(device)
    
    options = {
        'K': K,
        'C': C,
        'E': E,
        'B': B,
        'r': r,
        'lr': lr,
        'clients': clients,
        'model': model,
        'device': device,
        'test_dataset': test_dataset
    }
    
    # Train model
    server = ScaffoldServer(options)
    print("\nTraining federated model with SCAFFOLD...")
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
    
    print("\nOriginal Model Results:")
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate: {backdoor_acc:.2f}%")
    
    if defense_type in ["1", "5"]:
        print("\nApplying Neural Cleanse defense...")
        cleanse = ImprovedNeuralCleanse(
            model=server.model,
            num_classes=10,
            input_shape=input_shape,
            device=device
        )
        
        print("\nReverse-engineering potential triggers...")
        cleanse.reverse_engineer_triggers(
            max_iter=1000,
            lr=0.01,
            early_stop_threshold=0.99
        )
        
        print("\nAnalyzing model for potential backdoors...")
        backdoor_detected, suspected_classes, l1_norms, optimized_masks, optimized_patterns = cleanse.detect_backdoors(
            threshold_factor=2.0
        )
        
        if backdoor_detected:
            print(f"\nBackdoor detected! Suspected target classes: {suspected_classes}")
            
            print("\nRepairing model...")
            repaired_model = cleanse.repair_model(suspected_classes, optimized_masks, optimized_patterns)
            
            print("\nEvaluating repaired model...")
            clean_acc_after_nc, backdoor_acc_after_nc = cleanse.evaluate_repair(
                repaired_model, 
                test_dataset, 
                trigger_pattern, 
                target_label
            )
            
            server.model = repaired_model
            
            print("\nNeural Cleanse Defense Results:")
            print(f"Clean Test Accuracy: Before={clean_acc:.2f}%, After={clean_acc_after_nc:.2f}%")
            print(f"Backdoor Attack Success Rate: Before={backdoor_acc:.2f}%, After={backdoor_acc_after_nc:.2f}%")
            print(f"Backdoor Success Reduction: {backdoor_acc - backdoor_acc_after_nc:.2f}%")
            
            torch.save(repaired_model.state_dict(), './output/repaired_model.pth')
        else:
            print("\nNo backdoor detected by Neural Cleanse.")
        
    # Apply Spectral Backdoor Defender if selected
    if defense_type in ["2", "5"]:
        print("\nApplying Spectral Backdoor Defense...")
        
        # Create trigger patterns dictionary for the evaluate_repair method
        if attack_type == "1":
            trigger_patterns = {target_label: trigger_pattern}
            target_labels = {target_label: target_label}
        else:
            # For semantic backdoor, we'll pass None as we don't have explicit trigger patterns
            trigger_patterns = None
            target_labels = {target_label: target_label}
        
        # Initialize the Spectral Backdoor Defender
        spectral_defender = SpectralBackdoorDefender(
            model=server.model,
            device=device,
            num_classes=10,
            input_shape=input_shape
        )
        
        # Collect activations from layers
        print("\nCollecting layer activations...")
        spectral_defender.collect_layer_activations()
        
        # Compute eigenspectrum
        print("\nComputing eigenspectrum...")
        spectral_defender.compute_eigenspectrum()
        
        # Visualize eigenspectra
        print("\nVisualizing eigenspectra...")
        spectral_defender.visualize_eigenspectra()
        
        # Detect backdoors
        print("\nAnalyzing model for potential backdoors using spectral analysis...")
        backdoor_detected, compromised_layers, eigenvalue_metrics = spectral_defender.detect_backdoors(threshold_factor=2.0)
        spectral_defender.visualize_eigenspectra()
        spectral_defender.visualize_anomaly_metrics()
        spectral_defender.visualize_backdoor_detection_summary(eigenvalue_metrics)

        # For specific classes identified as suspicious
        for class_idx in suspected_classes:
            spectral_defender.visualize_activation_clusters("layer_name", class_idx)
            spectral_defender.visualize_neuron_activations("layer_name")
        if backdoor_detected:
            print(f"\nBackdoor detected by spectral analysis! Compromised layers: {compromised_layers}")
            
            # We would need to implement the repair method in the SpectralBackdoorDefender class
            # For now, let's assume it's implemented and use it
            
            # After repair, evaluate the repaired model
            if hasattr(spectral_defender, 'repair_model'):
                print("\nRepairing model...")
                repaired_model = spectral_defender.repair_model(compromised_layers=compromised_layers)
                
                # Evaluate the repaired model
                spectral_defender.evaluate_repair(repaired_model, test_dataset, trigger_patterns, target_labels)
                
                # Update server model with repaired model
                server.model = repaired_model

            else:
                print("\nRepair method not implemented in SpectralBackdoorDefender.")
        else:
            print("\nNo backdoor detected by spectral analysis.")
        
    # Apply Fine-Pruning defense if selected
    if defense_type in ["3", "5"]:
        print("\nApplying Fine-Pruning defense...")
        
        # Initialize Fine-Pruning
        fine_pruning = FinePruning(
            model=copy.deepcopy(server.model),
            clean_dataset=train_dataset,  # Using train dataset as clean data
            device=device,
            num_classes=10,
            prune_ratio=0.1  # Start with pruning 10% of neurons
        )
        
        # Apply the complete defense pipeline
        defended_model = fine_pruning.defend(
            trigger_pattern=trigger_pattern if attack_type == "1" else None,
            target_label=target_label,
            attack_type=attack_type,
            fine_tuning_epochs=3  # Adjust as needed
        )
        
        # Evaluate the defended model
        clean_acc_after_fp, backdoor_acc_after_fp = fine_pruning.evaluate_defense(
            defended_model,
            test_dataset,
            trigger_pattern=trigger_pattern if attack_type == "1" else None,
            target_label=target_label,
            attack_type=attack_type
        )
        
        print("\nFine-Pruning Defended Model Results:")
        print(f"Clean Test Accuracy: {clean_acc_after_fp:.2f}%")
        print(f"Backdoor Attack Success Rate: {backdoor_acc_after_fp:.2f}%")
        
        # Update server model with defended model for potential further analysis
        server.model = defended_model
        
        # Plot accuracy trends
        server.plot_accuracies()
    
    if defense_type in ["4", "5"]:
        print("\nApplying Activation Clustering defense...")
        
        # Initialize the Activation Clustering defense
        activation_clustering = ActivationClusteringDefender(
            model=server.model,
            dataset=train_dataset,
            device=device,
            num_classes=10,
            batch_size=64
        )
        
        # Collect activations from the model
        print("\nCollecting activations from the model...")
        sample_indices = activation_clustering.collect_activations()
        
        # Perform clustering analysis
        print("\nPerforming clustering analysis...")
        reduced_activations = activation_clustering.reduce_dimensions(method='pca', n_components=10)
        cluster_labels = activation_clustering.perform_clustering(reduced_activations, method='kmeans', n_clusters=2)
        
        # Detect potential backdoors
        print("\nDetecting potential backdoors...")
        poisoned_classes, poison_indices = activation_clustering.analyze_clusters(reduced_activations, cluster_labels, sample_indices)
        
        if poisoned_classes:
            print(f"\nBackdoor detected! Suspected poisoned classes: {poisoned_classes}")
            
            # Visualize the clustering results
            print("\nVisualizing clustering results...")
            activation_clustering.visualize_clusters(reduced_activations, cluster_labels)
            
            # Repair the model
            print("\nRepairing model using clean data...")
            repaired_model = activation_clustering.repair_model(
                clean_dataset=train_dataset,
                epochs=3,
                learning_rate=0.001
            )
            
            # Evaluate the repaired model
            print("\nEvaluating repaired model...")
            clean_acc_after_ac, backdoor_acc_after_ac = activation_clustering.evaluate_repair(
                repaired_model,
                test_dataset,
                trigger_pattern=trigger_pattern if attack_type == "1" else None,
                target_label=target_label,
                attack_type=attack_type
            )
            
            print("\nActivation Clustering Defense Results:")
            print(f"Clean Test Accuracy: Before={clean_acc:.2f}%, After={clean_acc_after_ac:.2f}%")
            print(f"Backdoor Success Rate: Before={backdoor_acc:.2f}%, After={backdoor_acc_after_ac:.2f}%")
        else:
            print("\nNo backdoor detected using Activation Clustering.")

if __name__ == "__main__":
    main()