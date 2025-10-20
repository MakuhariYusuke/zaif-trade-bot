#!/usr/bin/env python3
"""
Base ML Algorithm Trainer for Unified Training.

Handles base ML reinforcement experiments.
"""

from typing import Any, Dict

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseMLAlgorithmTrainer:
    """
    Handles base ML algorithm training.
    """

    def __init__(self, config_manager: "ConfigManager") -> None:
        """
        Initialize base ML trainer.

        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)

    def train(self, unified_config: Dict[str, Any]) -> Any:
        """
        Execute base ML training.

        Args:
            unified_config: Unified configuration

        Returns:
            Training result
        """
        from ztb.training.entrypoints.base_ml_reinforcement import (
            MLReinforcementExperiment,
        )

        experiment = MLReinforcementExperiment(
            unified_config, total_steps=unified_config.get("total_steps", 1000)
        )
        return experiment.run()
