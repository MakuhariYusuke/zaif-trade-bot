import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_model_name_from_config(cfg_path: Path) -> Optional[str]:
    """Read model_name from a config JSON if present.

    Args:
        cfg_path: Path to a JSON config

    Returns:
        model_name if present, else None
    """
    try:
        p = Path(cfg_path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("training", {}).get("model_name")
    except Exception:
        return None


def load_config_unified(
    config_path: str, required_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Unified config loading with validation.

    Args:
        config_path: Path to config file
        required_keys: List of required keys

    Returns:
        Config dictionary

    Raises:
        ValueError: If required keys are missing
        Exception: If loading fails
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if required_keys:
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Required config key missing: {key}")
        return config
    except Exception as e:
        from ztb.utils.logging_utils import get_logger

        logger = get_logger(__name__)
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise
