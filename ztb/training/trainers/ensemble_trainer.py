#!/usr/bin/env python3
"""
Ensemble Algorithm Trainer for Unified Training.

Handles ensemble training using multiple models.
"""

import json
from pathlib import Path
from typing import Any, Dict

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EnsembleAlgorithmTrainer:
    """
    Handles ensemble algorithm training.
    """

    def __init__(self, config_manager: Any) -> None:
        """
        Initialize ensemble trainer.

        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)

    def train(self, unified_config: Dict[str, Any]) -> Any:
        """
        Execute ensemble training.

        Args:
            unified_config: Unified configuration

        Returns:
            Ensemble system
        """
        from datetime import datetime

        from ztb.training.models.ensemble import EnsembleTradingSystem

        # Get model configurations from config
        model_configs = unified_config.get("ensemble_models", [])
        if not model_configs:
            raise ValueError(
                "No ensemble_models specified in config for ensemble training"
            )

        # Create ensemble system
        ensemble_system = EnsembleTradingSystem(model_configs)

        self.logger.info(
            f"Ensemble system initialized with {len(ensemble_system.ensemble.models)} models"
        )

        # For ensemble, we don't train but validate the setup
        if unified_config.get("dry_run", False):
            self.logger.info("Dry run: ensemble system setup validated")
            return ensemble_system

        # Save ensemble configuration for later use
        ensemble_config_path = (
            Path(unified_config.get("model_dir", "models")) / "ensemble_config.json"
        )
        with open(ensemble_config_path, "w") as f:
            json.dump(
                {
                    "model_configs": model_configs,
                    "created_at": str(datetime.now()),
                    "session_id": unified_config.get("session_id", "ensemble_session"),
                },
                f,
                indent=2,
            )

        self.logger.info(f"Ensemble configuration saved to {ensemble_config_path}")

        return ensemble_system
