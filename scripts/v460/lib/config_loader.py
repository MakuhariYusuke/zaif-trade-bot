"""
v460 Config loader — base.yaml + experiment.yaml マージ.

001# §4.2 準拠.

Override 規則:
  1. 実験 YAML は _base で base.yaml を指定
  2. 記載したキーのみ上書き
  3. features.selected と data.train_end_index は必須 (null → エラー)
  4. _gate で対応 Gate を明示
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (base is not mutated)."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key.startswith("_"):
            continue  # skip meta keys (_base, _gate, etc.)
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_config(
    experiment_path: str | Path,
    base_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Load experiment config with base.yaml merge.

    Args:
        experiment_path: Path to experiment YAML (absolute or relative to project root).
        base_path: Override for base.yaml path.  If None, read from experiment's ``_base``.

    Returns:
        Merged configuration dict.

    Raises:
        ValueError: If required fields are null after merge.
    """
    exp_path = Path(experiment_path)
    if not exp_path.is_absolute():
        exp_path = _PROJECT_ROOT / exp_path

    with open(exp_path, "r", encoding="utf-8") as f:
        exp_raw = yaml.safe_load(f) or {}

    # Resolve base
    if base_path is None:
        base_ref = exp_raw.get("_base", "configs/v460/base.yaml")
        base_path = _PROJECT_ROOT / base_ref
    else:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = _PROJECT_ROOT / base_path

    with open(base_path, "r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}

    merged = _deep_merge(base_raw, exp_raw)

    # Preserve meta
    try:
        merged["_base"] = str(base_path.relative_to(_PROJECT_ROOT))
    except ValueError:
        merged["_base"] = str(base_path)
    merged["_gate"] = exp_raw.get("_gate", "unknown")
    try:
        merged["_experiment"] = str(exp_path.relative_to(_PROJECT_ROOT))
    except ValueError:
        merged["_experiment"] = str(exp_path)

    # Validation
    _validate(merged)

    return merged


def _validate(cfg: dict[str, Any]) -> None:
    """Validate required fields are non-null."""
    errors: list[str] = []

    features_selected = cfg.get("features", {}).get("selected")
    if features_selected is None:
        errors.append("features.selected is null — must be specified in experiment YAML")

    train_end = cfg.get("data", {}).get("train_end_index")
    if train_end is None:
        errors.append("data.train_end_index is null — must be specified in experiment YAML")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


def load_gate_thresholds(
    path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Load gate_thresholds.yaml."""
    if path is None:
        path = _PROJECT_ROOT / "configs" / "v460" / "gate_thresholds.yaml"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
