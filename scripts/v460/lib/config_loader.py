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
from functools import lru_cache
from pathlib import Path
from typing import cast

from ztb.io.yaml_io import read_yaml
from ztb.types.common import ConfigSection, is_config_dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_IMMUTABLE_CONFIG_TYPES = (str, int, float, bool, bytes, type(None))


def _coerce_config_section(raw: object) -> ConfigSection:
    """Best-effort coercion for YAML payloads."""
    if isinstance(raw, dict):
        return cast(ConfigSection, raw)
    return {}


def _clone_config_value(value: object) -> object:
    """Clone config payloads while keeping immutable scalars zero-copy."""
    if isinstance(value, _IMMUTABLE_CONFIG_TYPES):
        return value
    if isinstance(value, list):
        return [_clone_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_config_value(item) for item in value)
    if isinstance(value, dict):
        return {
            str(k): _clone_config_value(v)
            for k, v in value.items()
            if isinstance(k, str)
        }
    return copy.deepcopy(value)


def _yaml_file_signature(path: Path) -> tuple[int, int]:
    """Return (mtime_ns, size) for cache invalidation; missing file keeps old behavior."""
    try:
        st = path.stat()
    except OSError:
        return -1, -1
    return st.st_mtime_ns, st.st_size


@lru_cache(maxsize=64)
def _read_config_section_cached(
    path_str: str,
    mtime_ns: int,
    size: int,
) -> ConfigSection:
    del mtime_ns, size  # only used as cache key
    return _coerce_config_section(read_yaml(path_str))


def _read_config_section(path: Path) -> ConfigSection:
    """Read YAML with file-signature cache while keeping caller isolation."""
    mtime_ns, size = _yaml_file_signature(path)
    return cast(
        ConfigSection,
        _clone_config_value(_read_config_section_cached(str(path), mtime_ns, size)),
    )


def _deep_merge(base: ConfigSection, override: ConfigSection) -> ConfigSection:
    """Recursively merge override into base (base is not mutated)."""
    merged = cast(ConfigSection, _clone_config_value(base))
    for key, val in override.items():
        if key.startswith("_"):
            continue  # skip meta keys (_base, _gate, etc.)
        existing = merged.get(key)
        if is_config_dict(val) and isinstance(existing, dict):
            merged[key] = _deep_merge(cast(ConfigSection, existing), val)
        else:
            merged[key] = _clone_config_value(val)
    return merged


def load_config(
    experiment_path: str | Path,
    base_path: str | Path | None = None,
) -> ConfigSection:
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

    exp_raw = _read_config_section(exp_path)

    # Resolve base
    if base_path is None:
        base_ref_value = exp_raw.get("_base")
        base_ref = base_ref_value if isinstance(base_ref_value, str) else "configs/v460/base.yaml"
        base_path = _PROJECT_ROOT / base_ref
    else:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = _PROJECT_ROOT / base_path

    base_raw = _read_config_section(base_path)

    merged = _deep_merge(base_raw, exp_raw)

    # Preserve meta
    try:
        merged["_base"] = str(base_path.relative_to(_PROJECT_ROOT))
    except ValueError:
        merged["_base"] = str(base_path)
    gate_value = exp_raw.get("_gate")
    merged["_gate"] = gate_value if isinstance(gate_value, str) else "unknown"
    # 007# F7: _task も明示復元 (_deep_merge がスキップするため)
    task_value = exp_raw.get("_task")
    merged["_task"] = task_value if isinstance(task_value, str) else "feature_info"
    try:
        merged["_experiment"] = str(exp_path.relative_to(_PROJECT_ROOT))
    except ValueError:
        merged["_experiment"] = str(exp_path)

    # Validation
    _validate(merged)

    return merged


def _validate(cfg: ConfigSection) -> None:
    """Validate required fields are non-null."""
    errors: list[str] = []

    features_value = cfg.get("features")
    features_selected = (
        features_value.get("selected")
        if is_config_dict(features_value)
        else None
    )
    if features_selected is None:
        errors.append("features.selected is null — must be specified in experiment YAML")

    data_value = cfg.get("data")
    train_end = (
        data_value.get("train_end_index")
        if is_config_dict(data_value)
        else None
    )
    if train_end is None:
        errors.append("data.train_end_index is null — must be specified in experiment YAML")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


def load_gate_thresholds(
    path: str | Path | None = None,
) -> ConfigSection:
    """Load gate_thresholds.yaml."""
    if path is None:
        path = _PROJECT_ROOT / "configs" / "v460" / "gate_thresholds.yaml"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

    return _read_config_section(path)


def load_fill_test_config(
    path: str | Path | None = None,
) -> ConfigSection:
    """Load fill_test.yaml — fill test の全設定を YAML から読込.

    v459 反省: 設定がコード内 dict で形骸化 → YAML 一元管理.

    Args:
        path: YAML パス. None なら configs/v460/fill_test.yaml.

    Returns:
        設定 dict. FillTestConfig / AdaptationConfig / LotSizingConfig に
        分配して使用する.
    """
    if path is None:
        path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path

    return _read_config_section(path)
