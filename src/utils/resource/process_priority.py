"""
Process Priority and Resource Control Utilities
プロセス優先度とリソース制御ユーティリティ

This module provides utilities for controlling process priority, CPU affinity,
and other resource management features for parallel training.
"""

import os
import sys
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
        Initialize priority manager

        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            self.config_path = Path(__file__).parent.parent.parent / "config" / "rl_config.json"
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def set_process_priority(self, model_type: str) -> Tuple[str, int]:
        """
        Set process priority and CPU affinity for a model type

        Args:
            model_type: Type of model ('generalization' or 'aggressive')

        Returns:
            Tuple of (priority_level, nice_value)
        """
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

    def _set_nice_value(self, nice_value: int):
        """Set process nice value"""
        try:
            if hasattr(os, 'nice'):
                current_nice = os.nice(0)
                new_nice = os.nice(nice_value)
                logger.info(f"Nice value changed: {current_nice} -> {new_nice}")
            else:
                logger.info("Nice value setting: not supported on Windows")
        except OSError as e:
            logger.error(f"Nice value setting error: {e}")

    def _set_cpu_affinity(self, model_type: str):
        """Set CPU affinity for the process"""
        try:
            if hasattr(os, 'sched_setaffinity'):
                cpu_count = psutil.cpu_count()
                if cpu_count is None:
                    cpu_count = 4  # Default fallback

                if model_type.lower() == 'generalization':
                    # Generalization model: first half of cores
                    cores = list(range(cpu_count // 2))
                else:
                    # Aggressive model: second half of cores
                    cores = list(range(cpu_count // 2, cpu_count))

                os.sched_setaffinity(0, cores)
                logger.info(f"CPU affinity set to cores: {cores}")
            else:
                logger.info("CPU affinity setting: not supported on Windows")
        except OSError as e:
            logger.error(f"CPU affinity setting error: {e}")

    def get_resource_info(self) -> Dict[str, Any]:
        """Get current resource information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'process': {
                'pid': os.getpid(),
                'nice': self._get_current_nice()
            }
        }

    def _get_current_nice(self) -> Optional[int]:
        """Get current process nice value"""
        try:
            if hasattr(os, 'nice'):
                return os.nice(0)
        except OSError:
            pass
        return None

    def reset_priority(self):
        """Reset process priority to default"""
        try:
            if hasattr(os, 'nice'):
                current_nice = os.nice(0)  # type: ignore
                if current_nice != 0:
                    os.nice(-current_nice)  # type: ignore  # Reset to 0
                logger.info("Process priority reset to default")
        except OSError as e:
            logger.error(f"Failed to reset priority: {e}")


def set_model_priority(model_type: str, config_path: Optional[str] = None) -> Tuple[str, int]:
    """
    Convenience function to set priority for a model type

    Args:
        model_type: Type of model ('generalization' or 'aggressive')
        config_path: Path to configuration file

    Returns:
        Tuple of (priority_level, nice_value)
    """
    manager = ProcessPriorityManager(config_path)
    return manager.set_process_priority(model_type)


def get_system_resources() -> Dict[str, Any]:
    """Get system resource information"""
    manager = ProcessPriorityManager()
    return manager.get_resource_info()