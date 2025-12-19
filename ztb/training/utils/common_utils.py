"""
Common utilities for training components.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().isoformat()


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def get_metric_from_logger(model, metric_name: str) -> Optional[float]:
    """Get metric value from model logger."""
    if not hasattr(model, "logger") or model.logger is None:
        return None

    try:
        name_to_value = model.logger.name_to_value

        # Check possible metric name variations
        possible_names = [
            metric_name,
            f"train/{metric_name}",
            f"rollout/{metric_name}",
        ]

        for name in possible_names:
            if name in name_to_value:
                return float(name_to_value[name])

        return None
    except (AttributeError, KeyError, TypeError):
        return None