#!/usr/bin/env python3
"""
Curriculum Algorithm Trainer for Unified Training.

Handles curriculum learning with staged training.
"""

import os
from typing import Any, Dict, Optional

from ztb.training.core.config_manager import ConfigManager
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

logger = get_logger(__name__)


class CurriculumAlgorithmTrainer:
    """
    Handles curriculum algorithm training.
    """

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize curriculum trainer.

        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = get_logger(__name__)

    def train(self, unified_config: Dict[str, Any]) -> Optional[bool]:
        """
        Execute curriculum learning training.

        Args:
            unified_config: Unified configuration

        Returns:
            Success indicator
        """
        from ztb.training.experiments.curriculum_learning import main as curriculum_main

        # Set up environment for curriculum learning
        self.logger.info("Starting curriculum learning (P0→P2 staged approach)")

        # Validate data path
        data_path = unified_config.get("data_path", "ml-dataset-enhanced.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Curriculum learning uses its own main function
        # We need to temporarily modify the working directory or config
        original_cwd = os.getcwd()

        try:
            # Change to project root for curriculum learning
            project_root = get_project_root()
            os.chdir(project_root)

            # Run curriculum learning
            curriculum_main()

            # Return success indicator
            return True

        except Exception as e:
            self.logger.error(f"Curriculum learning failed: {e}")
            return False

        finally:
            # Restore original working directory
            os.chdir(original_cwd)