"""
metrics.py
トレーディング指標計算モジュール
"""

from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def sharpe_ratio(
    returns: Union[List[float], NDArray[np.floating]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratioを計算（堅牢なNaN処理付き）

    Args:
        returns: リターンの配列
        risk_free_rate: 年率無リスク金利（デフォルト0.0）
        periods_per_year: 年間期間数（日次データの場合252）

    Returns:
        Sharpe ratio（年率換算）
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # 超過リターンの計算
    excess = returns - risk_free_rate / periods_per_year  # type: ignore

    mean_return = np.mean(excess)
    std_return = np.std(excess, ddof=1)

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    # 年率換算
    return float((mean_return / std_return) * np.sqrt(periods_per_year))


def sharpe_with_stats(sharpes: List[float]) -> Dict[str, Union[float, List[float]]]:
    """
    Sharpe ratioの統計情報を計算

    Args:
        sharpes: Sharpe ratioのリスト

    Returns:
        統計情報（平均、標準偏差、95%信頼区間）
    """
    if len(sharpes) == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}

    sharpes_array = np.array(sharpes)
    mean = float(np.mean(sharpes_array))
    std = float(np.std(sharpes_array, ddof=1))

    # 95%信頼区間
    ci95_low = float(np.percentile(sharpes_array, 2.5))
    ci95_high = float(np.percentile(sharpes_array, 97.5))

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95": [round(ci95_low, 6), round(ci95_high, 6)],
    }


def calculate_delta_sharpe(
    base_sharpes: List[float],
    with_feature_sharpes: List[float],
    min_samples: int = 10000,
) -> Optional[Dict[str, Union[float, List[float]]]]:
    """
    delta_sharpeを計算（安定化版）

    Args:
        base_sharpes: ベース特徴量でのSharpe ratioリスト
        with_feature_sharpes: 特徴量追加後のSharpe ratioリスト
        min_samples: 最低試行数（満たさない場合はNaN）

    Returns:
        delta_sharpe統計情報、またはNone（試行数不足時）
    """
    total_samples = len(base_sharpes) + len(with_feature_sharpes)

    if total_samples < min_samples:
        return None  # 試行数不足

    if len(base_sharpes) == 0 or len(with_feature_sharpes) == 0:
        return None

    # delta_sharpeの計算
    base_stats = sharpe_with_stats(base_sharpes)
    with_stats = sharpe_with_stats(with_feature_sharpes)

    delta_mean = cast(float, with_stats["mean"]) - cast(float, base_stats["mean"])
    delta_std = np.sqrt(
        cast(float, with_stats["std"]) ** 2 + cast(float, base_stats["std"]) ** 2
    )  # 誤差伝播

    # 95%信頼区間（簡易計算）
    delta_ci95_low = delta_mean - 1.96 * delta_std
    delta_ci95_high = delta_mean + 1.96 * delta_std

    return {
        "mean": round(delta_mean, 6),
        "std": round(delta_std, 6),
        "ci95": [round(delta_ci95_low, 6), round(delta_ci95_high, 6)],
    }


def validate_ablation_results(results: Dict[str, Any]) -> bool:
    """
    アブレーション結果の妥当性を検証

    Args:
        results: アブレーション結果

    Returns:
        妥当性（True: 有効、False: 無効）
    """
    # delta_sharpeがNoneでないことを確認
    if "delta_sharpe" not in results or results["delta_sharpe"] is None:
        return False

    # 統計情報が揃っていることを確認
    ds = results["delta_sharpe"]
    required_keys = ["mean", "std", "ci95"]

    for key in required_keys:
        if key not in ds:
            return False

    if not isinstance(ds["ci95"], list) or len(ds["ci95"]) != 2:
        return False

    return True


def calculate_feature_metrics(
    feature_data: pd.Series, price_data: pd.Series, feature_name: str
) -> Dict[str, Any]:
    """Calculate basic trading metrics for feature evaluation"""
    # Use feature-specific strategies
    if feature_name == "RSI":
        # RSI strategy: buy when RSI < 30, sell when RSI > 70
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < 30] = 1  # Buy signal
        signals[feature_data > 70] = -1  # Sell signal
    elif feature_name == "ROC":
        # ROC strategy: buy when ROC > 5, sell when ROC < -5
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data > 5] = 1
        signals[feature_data < -5] = -1
    elif feature_name == "OBV":
        # OBV strategy: buy when OBV increasing, sell when decreasing
        obv_change = feature_data.diff().astype(float)
        signals = pd.Series(0, index=feature_data.index)
        signals[obv_change > 0] = 1
        signals[obv_change < 0] = -1
    elif feature_name == "ZScore":
        # ZScore strategy: buy when ZScore < -1, sell when ZScore > 1 (mean reversion)
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < -1] = 1
        signals[feature_data > 1] = -1
    else:
        # Default strategy
        signals = (feature_data > 0).astype(int) - (feature_data < 0).astype(int)

    returns = price_data.pct_change().shift(-1)  # Next period returns

    # Calculate metrics
    valid_idx = signals.notna() & returns.notna() & (signals != 0)
    if valid_idx.sum() == 0:
        return {
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sample_count": 0,
        }

    strategy_returns = signals[valid_idx] * returns[valid_idx]
    cumulative = (1 + strategy_returns).cumprod()

    # Win rate
    win_rate = (strategy_returns > 0).mean()

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe ratio
    if strategy_returns.std() > 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Sortino ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = strategy_returns.mean() / downside_returns.std() * np.sqrt(252)
    else:
        sortino_ratio = 0.0

    # Calmar ratio
    if abs(max_drawdown) > 0:
        calmar_ratio = strategy_returns.mean() * 252 / abs(max_drawdown)
    else:
        calmar_ratio = 0.0

    return {
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "sample_count": int(valid_idx.sum()),
    }
