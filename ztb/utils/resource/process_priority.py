"""
Process Priority and Resource Control Utilities
プロセス優先度とリソース制御ユーティリティ

This module provides utilities for controlling process priority, CPU affinity,
and other resource management features for parallel training.
"""

import os
import json
import psutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ProcessPriorityManager:
    """Manager for process priority and CPU affinity settings"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize priority manager.

        Args:
            config_path: Path to configuration file. If not provided, will use the path from the
                         ZAIF_TRADE_BOT_CONFIG environment variable if set, otherwise defaults to
                         '<project_root>/config/rl_config.json'.
        """
        if config_path is None:
            env_config_path = os.environ.get("ZAIF_TRADE_BOT_CONFIG")
            if env_config_path:
                self.config_path = Path(env_config_path)
            else:
                self.config_path = Path(__file__).parent.parent.parent / "config" / "rl_config.json"
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config  # type: ignore
        return {}

    def set_process_priority(self, model_type: str) -> Tuple[str, int]:
        """
        Set process priority and CPU affinity for a model type

        Args:
            model_type: Type of model ('generalization' or 'aggressive')

        Returns:
            Tuple of (priority_level, nice_value)
        """
        valid_model_types = {'generalization', 'aggressive'}
        if model_type.lower() not in valid_model_types:
            raise ValueError(f"Invalid model_type '{model_type}'. Accepted values are: {valid_model_types}")

        logger.info(f"Setting resource control for {model_type} model")

        # Get configuration
        parallel_config = self.config.get('parallel_training', {})
        priority = parallel_config.get(f'{model_type.lower()}_priority', 'normal')
        nice_value = parallel_config.get('nice_values', {}).get(priority, 0)
        affinity_enabled = parallel_config.get('cpu_affinity_enabled', False)

        logger.info(f"Priority setting: {priority} (nice value: {nice_value})")
        logger.info(f"CPU affinity: {'enabled' if affinity_enabled else 'disabled'}")

        # Set nice value (Linux only)
        self._set_nice_value(nice_value)

        # Set CPU affinity (Linux only)
        if affinity_enabled:
            self._set_cpu_affinity(model_type)

        return priority, nice_value

    def _set_nice_value(self, nice_value: int) -> None:
        """Set process nice value"""
        try:
            if hasattr(os, 'nice'):
                current_nice = os.nice(0)  # type: ignore
                increment = nice_value - current_nice
                if increment != 0:
                    new_nice = os.nice(increment)  # type: ignore
                    logger.info(f"Nice value changed: {current_nice} -> {new_nice}")
                else:
                    logger.info(f"Nice value unchanged: {current_nice}")
            else:
                logger.info("Nice value setting: not supported on this platform")
        except OSError as e:
            logger.error(f"Nice value setting error: {e}")

    def _set_cpu_affinity(self, model_type: str) -> None:
        """Set CPU affinity for the process"""
        try:
            if hasattr(os, 'sched_setaffinity'):
                # Use physical cores for affinity to improve performance
                cpu_count = psutil.cpu_count(logical=False)
                if cpu_count is None:
                    cpu_count = 4  # Default fallback

                if model_type.lower() == 'generalization':
                    # Generalization model: first half of physical cores
                    cores = list(range(cpu_count // 2))
                else:
                    # Aggressive model: second half of physical cores
                    cores = list(range(cpu_count // 2, cpu_count))

                os.sched_setaffinity(0, cores)  # type: ignore
                logger.info(f"CPU affinity set to physical cores: {cores}")
            else:
                logger.info("CPU affinity setting: not supported on this platform")
        except OSError as e:
            logger.error(f"CPU affinity setting error: {e}")

    def reset_priority(self) -> None:
        """Reset process priority to default"""
        try:
            if hasattr(os, 'nice'):
                current_nice = os.nice(0)
                if current_nice != 0:
                    try:
                        os.nice(-current_nice)  # type: ignore  # type: ignore
                        logger.info("Process priority reset to default")
                    except PermissionError:
                        logger.warning("Insufficient permissions to decrease nice value (increase priority). Run as root/administrator if needed.")
                else:
                    logger.info("Process priority already at default")
            else:
                logger.info("Resetting process priority: not supported on this platform")
        except OSError as e:
            logger.error(f"Failed to reset priority: {e}")

    def get_nice_value(self) -> Optional[int]:
        """Get current process nice value"""
        try:
            if hasattr(os, 'nice'):
                return os.nice(0)  # type: ignore
            else:
                logger.info("Getting nice value: not supported on this platform")
                return None
        except OSError:
            return None
