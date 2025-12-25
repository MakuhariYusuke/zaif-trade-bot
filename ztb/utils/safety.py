import json
from pathlib import Path
from typing import Any, Dict, Optional, List


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


def safe_to_int(value: Any, default: int = 0) -> int:
    """値を安全に int に変換する。失敗したら default を返す。"""
    try:
        return int(float(value))  # floatを経由して "1.0" なども扱えるように
    except Exception:
        return default


def safe_to_bool(value: Any, default: bool = False) -> bool:
    """値を安全に bool に変換する。失敗したら default を返す。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    try:
        return bool(value)
    except Exception:
        return default


def safe_get_nested_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    ネストされた辞書から安全に値を取得する。

    Args:
        data: 辞書データ
        keys: キーのリスト (例: ['a', 'b', 'c'] は data['a']['b']['c'])
        default: デフォルト値

    Returns:
        取得した値またはデフォルト値
    """
    current = data
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


def safe_list_get(lst: List[Any], index: int, default: Any = None) -> Any:
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


def safe_config_get(config: Dict[str, Any], key: str, default: Any = None, required: bool = False) -> Any:
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
    if '.' in key:
        keys = key.split('.')
        value = safe_get_nested_value(config, keys, default)
    else:
        value = config.get(key, default)

    if required and value is None:
        raise ValueError(f"Required config key '{key}' not found")

    return value


def safe_config_get_float(config: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """設定からfloat値を安全に取得"""
    value = safe_config_get(config, key, default)
    return safe_to_float(value, default)


def safe_config_get_int(config: Dict[str, Any], key: str, default: int = 0) -> int:
    """設定からint値を安全に取得"""
    value = safe_config_get(config, key, default)
    return safe_to_int(value, default)


def safe_config_get_bool(config: Dict[str, Any], key: str, default: bool = False) -> bool:
    """設定からbool値を安全に取得"""
    value = safe_config_get(config, key, default)
    return safe_to_bool(value, default)
