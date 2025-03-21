class DefenseConfig:
    """Configuration for defense mechanisms"""
    NEURAL_CLEANSE = {
        'learning_rate': 0.01,
        'optimization_steps': 1000,
        'norm_threshold': 5
    }
    
    FINE_PRUNING = {
        'prune_ratio': 0.1,
        'finetune_epochs': 10,
        'learning_rate': 0.001
    }
    
    MCR = {
        'num_points': 5,
        'curve_type': 'bezier',
        'smoothing_factor': 0.1
    }