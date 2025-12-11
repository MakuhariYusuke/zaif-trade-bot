import json
from pathlib import Path
from typing import Optional


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
