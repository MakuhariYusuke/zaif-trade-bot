#!/usr/bin/env python3
"""
Configuration classes for Unified Trainer.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, cast, List


class UnifiedAlgorithm(Enum):
    """Supported training algorithms in UnifiedTrainer."""

    PPO = "ppo"
    BASE_ML = "base_ml"
    ITERATIVE = "iterative"
    ENSEMBLE = "ensemble"
    CURRICULUM = "curriculum"
    SELF_SUPERVISED = "self_supervised"
    MULTIMODAL = "multimodal"
    ONLINE_LEARNING = "online_learning"


@dataclass
class UnifiedTrainerConfig:
    """Configuration for UnifiedTrainer."""

    algorithm: UnifiedAlgorithm
    force: bool = False
    dry_run: bool = False
    enable_streaming: bool = False
    stream_batch_size: int = 256
    max_features: Optional[int] = None
    offline_mode: bool = False
    total_timesteps: Optional[int] = None  # Added to fix attribute access error
    
    # Federated Learning Configuration
    enable_federated: bool = False
    num_clients: int = 3
    federated_rounds: int = 10
    privacy_budget: float = 1.0  # Differential privacy budget
    client_fraction: float = 1.0  # Fraction of clients to participate per round
    
    # Mixed Precision Training Configuration
    enable_mixed_precision: bool = False
    precision: str = "fp16"  # "fp16", "bf16", "fp8"
    gradient_scaling: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    
    # Efficient Network Configuration
    use_efficient_network: bool = False
    use_depthwise_conv: bool = True
    use_efficient_attention: bool = True
    use_dynamic_network: bool = True
    attention_method: str = "linformer"  # "linformer", "performer"
    sequence_length: int = 10

    # Multimodal Learning Configuration
    enable_multimodal: bool = False
    price_feature_dim: int = 156
    text_embedding_dim: int = 768
    economic_feature_dim: int = 10
    multimodal_hidden_dim: int = 256
    multimodal_num_heads: int = 8

    # Online Learning Configuration
    enable_online_learning: bool = False
    online_learning_mode: str = "incremental"  # "incremental", "streaming"
    online_batch_size: int = 32
    online_memory_samples: int = 10000
    online_adaptation_threshold: float = 0.1

    # Anomaly Detection Configuration
    enable_anomaly_detection: bool = False
    anomaly_statistical_methods: Optional[List[str]] = None  # ['zscore', 'iqr', 'mad']
    anomaly_ml_methods: Optional[List[str]] = None  # ['isolation_forest', 'elliptic_envelope']
    enable_anomaly_autoencoder: bool = False
    anomaly_voting_threshold: float = 0.5

    # Meta Learning Configuration
    enable_meta_learning: bool = False
    meta_algorithm: str = 'maml'  # 'maml' or 'reptile'
    meta_batch_size: int = 4
    meta_inner_lr: float = 0.01
    meta_outer_lr: float = 0.001
    meta_adaptation_steps: int = 10

    # Enhanced Federated Learning Configuration
    federated_markets: bool = False  # Enable market-based federated learning
    markets: Optional[List[str]] = None  # List of markets for federated learning

    # Continual Learning Configuration
    enable_continual_learning: bool = False
    continual_method: str = 'ewc'  # 'ewc', 'rehearsal', 'progressive'
    continual_ewc_lambda: float = 0.1
    continual_buffer_size: int = 1000
    continual_max_tasks: int = 5


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary or None if loading fails
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = cast(dict[str, Any], json.load(f))
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return None