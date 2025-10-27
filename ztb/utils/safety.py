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


def safe_file_operation(file_path: str, operation: str, *args, **kwargs) -> Any:
    """
    ファイル操作を安全に行う。

    Args:
        file_path: ファイルパス
        operation: 操作 ('read', 'write', 'append')
        *args: 操作に渡す追加引数
        **kwargs: 操作に渡す追加キーワード引数

    Returns:
        操作結果またはNone
    """
    try:
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                return f.write(*args, **kwargs)
        elif operation == 'append':
            with open(file_path, 'a', encoding='utf-8') as f:
                return f.write(*args, **kwargs)
        else:
            return None
    except Exception:
        return None
