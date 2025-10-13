#!/usr/bin/env python3
"""
Configuration classes for Unified Trainer.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class UnifiedAlgorithm(Enum):
    """Supported training algorithms in UnifiedTrainer."""

    PPO = "ppo"
    BASE_ML = "base_ml"
    ITERATIVE = "iterative"
    ENSEMBLE = "ensemble"
    CURRICULUM = "curriculum"


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
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return None