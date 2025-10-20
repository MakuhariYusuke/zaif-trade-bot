"""
Configuration for Self-Supervised Pre-training
自己教師あり事前学習の設定

This file contains default configurations for all self-supervised learning
techniques implemented in the pretraining module.

設定内容:
- Masked Price Modeling parameters
- Contrastive Learning parameters
- Anomaly Detection parameters
- Training hyperparameters
"""

from typing import Any, Dict

# Default configuration for Masked Price Modeling
MPM_CONFIG = {
    "hidden_dim": 512,
    "num_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "max_seq_len": 100,
    "mask_prob": 0.15,
    "learning_rate": 1e-4,
}

# Default configuration for Contrastive Learning
CONTRASTIVE_CONFIG = {
    "hidden_dim": 512,
    "projection_dim": 128,
    "temperature": 0.5,
    "learning_rate": 1e-4,
    "augmentation": {
        "shift_prob": 0.5,
        "noise_prob": 0.3,
        "scale_prob": 0.2,
        "max_shift": 5,
        "noise_std": 0.1,
        "scale_range": (0.8, 1.2),
    },
}

# Default configuration for Anomaly Detection
ANOMALY_CONFIG = {
    "hidden_dims": [256, 128, 64],
    "latent_dim": 32,
    "lstm_hidden_dim": 128,
    "lstm_num_layers": 2,
    "seq_len": 100,
    "alpha": 0.5,  # Balance between reconstruction and prediction
    "learning_rate": 1e-4,
}

# Default training configuration
TRAINING_CONFIG = {"epochs": 100, "batch_size": 32, "patience": 10, "save_best": True}

# Complete self-supervised pre-training configuration
SELF_SUPERVISED_CONFIG = {
    "mpm": MPM_CONFIG,
    "mpm_training": TRAINING_CONFIG,
    "contrastive": CONTRASTIVE_CONFIG,
    "contrastive_training": TRAINING_CONFIG,
    "anomaly": ANOMALY_CONFIG,
    "anomaly_training": TRAINING_CONFIG,
    # Global settings
    "input_dim": 156,  # Financial feature dimension
    "device": "cuda",  # or 'cpu'
    "checkpoint_dir": "checkpoints/pretraining",
    "random_seed": 42,
}

# Lightweight configuration for quick testing
LIGHTWEIGHT_CONFIG = {
    "mpm": {
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
        "max_seq_len": 50,
        "mask_prob": 0.15,
        "learning_rate": 5e-4,
    },
    "mpm_training": {"epochs": 50, "batch_size": 16, "patience": 5, "save_best": True},
    "contrastive": {
        "hidden_dim": 256,
        "projection_dim": 64,
        "temperature": 0.5,
        "learning_rate": 5e-4,
        "augmentation": {
            "shift_prob": 0.5,
            "noise_prob": 0.3,
            "scale_prob": 0.2,
            "max_shift": 3,
            "noise_std": 0.05,
            "scale_range": (0.9, 1.1),
        },
    },
    "contrastive_training": {
        "epochs": 50,
        "batch_size": 16,
        "patience": 5,
        "save_best": True,
    },
    "anomaly": {
        "hidden_dims": [128, 64, 32],
        "latent_dim": 16,
        "lstm_hidden_dim": 64,
        "lstm_num_layers": 1,
        "seq_len": 50,
        "alpha": 0.5,
        "learning_rate": 5e-4,
    },
    "anomaly_training": {
        "epochs": 50,
        "batch_size": 16,
        "patience": 5,
        "save_best": True,
    },
    "input_dim": 156,
    "device": "cuda",
    "checkpoint_dir": "checkpoints/pretraining_lightweight",
    "random_seed": 42,
}

# High-performance configuration for production use
PRODUCTION_CONFIG = {
    "mpm": {
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
        "max_seq_len": 200,
        "mask_prob": 0.15,
        "learning_rate": 5e-5,
    },
    "mpm_training": {
        "epochs": 200,
        "batch_size": 64,
        "patience": 20,
        "save_best": True,
    },
    "contrastive": {
        "hidden_dim": 768,
        "projection_dim": 256,
        "temperature": 0.1,
        "learning_rate": 5e-5,
        "augmentation": {
            "shift_prob": 0.6,
            "noise_prob": 0.4,
            "scale_prob": 0.3,
            "max_shift": 10,
            "noise_std": 0.05,
            "scale_range": (0.7, 1.3),
        },
    },
    "contrastive_training": {
        "epochs": 200,
        "batch_size": 64,
        "patience": 20,
        "save_best": True,
    },
    "anomaly": {
        "hidden_dims": [512, 256, 128, 64],
        "latent_dim": 64,
        "lstm_hidden_dim": 256,
        "lstm_num_layers": 3,
        "seq_len": 200,
        "alpha": 0.3,  # Favor reconstruction for anomaly detection
        "learning_rate": 5e-5,
    },
    "anomaly_training": {
        "epochs": 200,
        "batch_size": 64,
        "patience": 20,
        "save_best": True,
    },
    "input_dim": 156,
    "device": "cuda",
    "checkpoint_dir": "checkpoints/pretraining_production",
    "random_seed": 42,
}


def get_config(config_type: str = "default") -> Dict[str, Any]:
    """
    Get configuration by type

    Args:
        config_type: Configuration type ('default', 'lightweight', 'production')

    Returns:
        Configuration dictionary
    """
    if config_type == "lightweight":
        return LIGHTWEIGHT_CONFIG.copy()
    elif config_type == "production":
        return PRODUCTION_CONFIG.copy()
    else:
        return SELF_SUPERVISED_CONFIG.copy()


def update_config(
    base_config: Dict[str, Any], updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update configuration with custom parameters

    Args:
        base_config: Base configuration
        updates: Updates to apply

    Returns:
        Updated configuration
    """
    config = base_config.copy()

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v

    deep_update(config, updates)
    return config
