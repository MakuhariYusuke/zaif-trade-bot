from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any, TypeVar

from ztb.io.json_io import read_json_object

ObjectMap = dict[str, object]
TDefault = TypeVar("TDefault")

def safe_open_json(path: Path | None) -> ObjectMap | None:
    """安全に JSON ファイルを開いて dict を返す。失敗したら None を返す。

    Args:
        path: Path または None

    Returns:
        dict または None
    """
    if path is None:
        return None
    try:
        return read_json_object(path)
    except Exception:
        return None

def ensure_dict(value: object | None) -> ObjectMap:
    """与えられた値を dict に正規化する。変換できなければ空 dict を返す。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    try:
        coerced = dict(value)  # type: ignore[arg-type]
        return {str(k): v for k, v in coerced.items()}
    except Exception:
        return {}

def safe_to_float(value: object, default: float = 0.0) -> float:
    """値を安全に float に変換する。失敗したら default を返す。"""
    try:
        return float(value)
    except Exception:
        return default

def safe_to_finite(value: Any) -> float | None:
    """値を有限 float に安全変換. None / NaN / Inf → None を返す.

    161# DRY: ab_judgment._safe_finite / dashboard._to_finite を統合。
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None

def safe_to_int(value: object, default: int = 0) -> int:
    """値を安全に int に変換する。失敗したら default を返す。"""
    try:
        return int(float(value))  # floatを経由して "1.0" なども扱えるように
    except Exception:
        return default

def safe_to_bool(value: object, default: bool = False) -> bool:
    """値を安全に bool に変換する。失敗したら default を返す。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    try:
        return bool(value)
    except Exception:
        return default

def safe_get_nested_value(
    data: ObjectMap, keys: list[str], default: TDefault | None = None
) -> object | TDefault | None:
    """
    ネストされた辞書から安全に値を取得する。

    Args:
        data: 辞書データ
        keys: キーのリスト (例: ['a', 'b', 'c'] は data['a']['b']['c'])
        default: デフォルト値

    Returns:
        取得した値またはデフォルト値
    """
    current: object = data
    try:
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except Exception:
        return default

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全な除算を行う。ゼロ除算を避ける。

    Args:
        numerator: 分子
        denominator: 分母
        default: 分母が0の場合のデフォルト値

    Returns:
        除算結果またはデフォルト値
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except Exception:
        return default

def safe_list_get(
    lst: list[object], index: int, default: TDefault | None = None
) -> object | TDefault | None:
    """
    リストから安全に要素を取得する。インデックスエラーを避ける。

    Args:
        lst: リスト
        index: インデックス
        default: デフォルト値

    Returns:
        取得した要素またはデフォルト値
    """
    try:
        return lst[index] if 0 <= index < len(lst) else default
    except Exception:
        return default

def validate_range(value: float, min_val: float, max_val: float) -> bool:
    """
    値が指定範囲内にあるかを検証する。

    Args:
        value: 検証する値
        min_val: 最小値
        max_val: 最大値

    Returns:
        範囲内ならTrue
    """
    return min_val <= value <= max_val

def safe_config_get(
    config: ObjectMap,
    key: str,
    default: TDefault | None = None,
    required: bool = False,
) -> object | TDefault | None:
    """
    設定から安全に値を取得する。ネストされたキーにも対応。

    Args:
        config: 設定辞書
        key: キー (ドット区切りでネスト指定可能、例: 'training.learning_rate')
        default: デフォルト値
        required: Trueの場合、値が存在しないと例外を発生

    Returns:
        取得した値またはデフォルト値

    Raises:
        ValueError: required=Trueで値が存在しない場合
    """
    if "." in key:
        keys = key.split(".")
        value = safe_get_nested_value(config, keys, default)
    else:
        value = config.get(key, default)

    if required and value is None:
        raise ValueError(f"Required config key '{key}' not found")

    return value

def safe_config_get_float(
    config: ObjectMap, key: str, default: float = 0.0
) -> float:
    """設定からfloat値を安全に取得"""
    value = safe_config_get(config, key, default)
    return safe_to_float(value, default)

def safe_config_get_int(config: ObjectMap, key: str, default: int = 0) -> int:
    """設定からint値を安全に取得"""
    value = safe_config_get(config, key, default)
    return safe_to_int(value, default)

def safe_config_get_bool(
    config: ObjectMap, key: str, default: bool = False
) -> bool:
    """設定からbool値を安全に取得"""
    value = safe_config_get(config, key, default)
    return safe_to_bool(value, default)

def safe_config_get_str(config: ObjectMap, key: str, default: str = "") -> str:
    """設定からstr値を安全に取得"""
    value = safe_config_get(config, key, default)
    return str(value) if value is not None else default
