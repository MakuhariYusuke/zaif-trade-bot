"""
統計ユーティリティモジュール

統計計算に関するユーティリティ関数を提供します。
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from ztb.metrics.metrics import autocorrelation


def _ensure_numpy(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return data.to_numpy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def p_mean_method(p_values: List[float], method: str = "arithmetic") -> float:
    """
    p平均法による総合p値の計算

    p平均法は、複数の独立した統計検定のp値を統合し、
    全体として統計的有意性があるかを評価する手法です。

    Args:
        p_values: p値のリスト（0.0 ~ 1.0の範囲）
        method: 平均化手法
            - 'arithmetic': 算術平均（単純平均）
            - 'geometric': 幾何平均（対数変換後平均）

    Returns:
        総合p値（0.0 ~ 1.0）

    算術平均の特徴:
        - 直感的で理解しやすい
        - 全てのp値に等しい重み付け
        - 極端なp値（0.99など）の影響を受けやすい

    幾何平均の特徴:
        - 極端なp値の影響を緩和
        - 0に非常に近いp値を適切に扱える
        - 統計学的によりロバスト

    使用例:
        # 3つのメトリクスのp値統合
        p_values = [0.03, 0.07, 0.02]  # 個別のt検定結果
        combined_p = p_mean_method(p_values, 'geometric')
        significant = combined_p < 0.05

    注意事項:
        - p値が完全に独立であることを仮定
        - 相関のある検定では結果が保守的になる可能性
        - 解釈時は個別の検定結果も確認すること
    """
    if not p_values:
        return 1.0

    p_array = np.array(p_values)

    if method == "arithmetic":
        # 算術平均
        return float(np.mean(p_array))
    elif method == "geometric":
        # 幾何平均 (0を避けるため小さな値を加算)
        p_array = np.clip(p_array, 1e-10, 1.0)
        return float(np.exp(np.mean(np.log(p_array))))
    else:
        raise ValueError(f"Unknown method: {method}")


def rolling_statistics(
    data: Union[List[float], np.ndarray, pd.Series], window: int
) -> Dict[str, List[float]]:
    """
    Calculate rolling statistics for time series data.

    Args:
        data: Time series data
        window: Rolling window size

    Returns:
        Dictionary with rolling mean, std, min, max
    """
    np_data = _ensure_numpy(data)
    if len(np_data) < window:
        return {"mean": [], "std": [], "min": [], "max": []}

    means = []
    stds = []
    mins = []
    maxs = []

    # Use pandas rolling if available for efficiency, otherwise loop
    if isinstance(data, pd.Series):
        rolling = data.rolling(window=window)
        means = rolling.mean().dropna().tolist()
        stds = rolling.std().dropna().tolist()
        mins = rolling.min().dropna().tolist()
        maxs = rolling.max().dropna().tolist()
    else:
        # Fallback for list/numpy
        for i in range(window - 1, len(np_data)):
            window_data = np_data[i - window + 1 : i + 1]
            means.append(float(np.mean(window_data)))
            # Use ddof=1 for sample standard deviation to match pandas default
            stds.append(float(np.std(window_data, ddof=1)))
            mins.append(float(np.min(window_data)))
            maxs.append(float(np.max(window_data)))

    return {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
    }


def calculate_volatility(
    data: Union[List[float], np.ndarray, pd.Series], window: int = 20
) -> List[float]:
    """
    Calculate rolling volatility (standard deviation).

    Args:
        data: Time series data (typically returns)
        window: Rolling window size

    Returns:
        Rolling volatility values
    """
    stats = rolling_statistics(data, window)
    return stats["std"]


def calculate_autocorrelation(
    data: Union[List[float], np.ndarray, pd.Series], lag: int = 1
) -> float:
    """
    Calculate autocorrelation at specified lag.

    Args:
        data: Time series data
        lag: Lag for autocorrelation

    Returns:
        Autocorrelation coefficient
    """
    # autocorrelation function in metrics.metrics likely expects list or series
    # Let's check metrics.metrics implementation if possible, but for now pass as is
    # assuming it handles it or we convert.
    # Actually, let's convert to Series for consistency if it's not
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    return autocorrelation(data, lag=lag)


def detect_outliers_iqr(
    data: Union[List[float], np.ndarray, pd.Series], multiplier: float = 1.5
) -> List[bool]:
    """
    Detect outliers using IQR method.

    Args:
        data: Data points
        multiplier: IQR multiplier for outlier detection

    Returns:
        Boolean list indicating outliers
    """
    np_data = _ensure_numpy(data)
    if len(np_data) == 0:
        return []

    q1 = np.percentile(np_data, 25)
    q3 = np.percentile(np_data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return [(x < lower_bound or x > upper_bound) for x in np_data]


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: ATR period

    Returns:
        Series of ATR values
    """
    if len(data) == 0:
        return pd.Series(dtype=float)

    if (
        "high" not in data.columns
        or "low" not in data.columns
        or "close" not in data.columns
    ):
        raise ValueError("Data must contain 'high', 'low', and 'close' columns")

    from ztb.features.generators.technical.volatility.atr import compute_atr

    return compute_atr(data, period=period)


def calculate_distribution_stats(
    data: Union[List[float], np.ndarray], decimals: int = 6
) -> Dict[str, Any]:
    """
    Calculate statistics for a distribution (mean, std, 95% CI).

    Args:
        data: Input data
        decimals: Number of decimals to round to

    Returns:
        Dictionary with mean, std, ci95
    """
    np_data = _ensure_numpy(data)

    if len(np_data) == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}

    mean = float(np.mean(np_data))
    std = float(np.std(np_data, ddof=1))

    # 95% Confidence Interval
    ci95_low = float(np.percentile(np_data, 2.5))
    ci95_high = float(np.percentile(np_data, 97.5))

    return {
        "mean": round(mean, decimals),
        "std": round(std, decimals),
        "ci95": [round(ci95_low, decimals), round(ci95_high, decimals)],
    }
