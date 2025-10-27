"""
統計ユーティリティモジュール

統計計算に関するユーティリティ関数を提供します。
"""

from typing import List, Dict

import numpy as np


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


def rolling_statistics(data: List[float], window: int) -> Dict[str, List[float]]:
    """
    Calculate rolling statistics for time series data.

    Args:
        data: Time series data
        window: Rolling window size

    Returns:
        Dictionary with rolling mean, std, min, max
    """
    if len(data) < window:
        return {"mean": [], "std": [], "min": [], "max": []}

    means = []
    stds = []
    mins = []
    maxs = []

    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1 : i + 1]
        means.append(float(np.mean(window_data)))
        stds.append(float(np.std(window_data)))
        mins.append(float(np.min(window_data)))
        maxs.append(float(np.max(window_data)))

    return {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
    }


def calculate_volatility(data: List[float], window: int = 20) -> List[float]:
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


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0.0

    excess_returns = [r - risk_free_rate / 252 for r in returns]  # Daily risk-free rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    if std_excess == 0:
        return 0.0

    return float(mean_excess / std_excess * np.sqrt(252))  # Annualized


def calculate_max_drawdown(data: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown statistics.

    Args:
        data: Time series data (typically portfolio values)

    Returns:
        Dictionary with max drawdown, peak, trough
    """
    if not data:
        return {"max_drawdown": 0.0, "peak": 0.0, "trough": 0.0}

    peak = data[0]
    max_drawdown = 0.0
    peak_idx = 0
    trough_idx = 0

    for i, value in enumerate(data):
        if value > peak:
            peak = value
            peak_idx = i

        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            trough_idx = i

    return {
        "max_drawdown": max_drawdown,
        "peak": peak,
        "trough": data[trough_idx] if data else 0.0,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
    }


def calculate_autocorrelation(data: List[float], lag: int = 1) -> float:
    """
    Calculate autocorrelation at specified lag.

    Args:
        data: Time series data
        lag: Lag for autocorrelation

    Returns:
        Autocorrelation coefficient
    """
    if len(data) <= lag:
        return 0.0

    data_array = np.array(data)
    return float(np.corrcoef(data_array[:-lag], data_array[lag:])[0, 1])


def detect_outliers_iqr(data: List[float], multiplier: float = 1.5) -> List[bool]:
    """
    Detect outliers using IQR method.

    Args:
        data: Data points
        multiplier: IQR multiplier for outlier detection

    Returns:
        Boolean list indicating outliers
    """
    if not data:
        return []

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return [x < lower_bound or x > upper_bound for x in data]
