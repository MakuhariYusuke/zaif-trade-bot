from pathlib import Path
from typing import Any, Dict, Optional
import json


def safe_open_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """安全に JSON ファイルを開いて dict を返す。失敗したら None を返す。

    Args:
        path: Path または None

    Returns:
        dict または None
    """
    if path is None:
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return None
    except Exception:
        return None


def ensure_dict(value: Optional[Any]) -> Dict[str, Any]:
    """与えられた値を dict に正規化する。変換できなければ空 dict を返す。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def safe_to_float(value: Any, default: float = 0.0) -> float:
    """値を安全に float に変換する。失敗したら default を返す。"""
    try:
        return float(value)
    except Exception:
        return default
