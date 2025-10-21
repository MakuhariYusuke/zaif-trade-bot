#!/usr/bin/env python3
"""
Base ML Algorithm Trainer for Unified Training.

Handles base ML reinforcement experiments.
"""

from typing import Any, Dict, TYPE_CHECKING

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.training.core.config_manager import ConfigManager

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

        # Local narrowing: prefer passed unified_config (dict), otherwise fall back to
        # self.config if available. This avoids attribute-defined type issues.
        cfg = unified_config if isinstance(unified_config, dict) else getattr(self, "config", {})

        experiment = MLReinforcementExperiment(
            cfg, total_steps=cfg.get("total_steps", 1000)
        )
        return experiment.run()
