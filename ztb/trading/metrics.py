"""
metrics.py
トレーディング指標計算モジュール
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sharpe ratioを計算

    Args:
        returns: リターンの配列
        risk_free_rate: 年率無リスク金利（デフォルト0.0）
        periods_per_year: 年間期間数（日次データの場合252）

    Returns:
        Sharpe ratio（年率換算）
    """
    if len(returns) == 0:
        return 0.0

    # 超過リターンの計算
    excess = returns - risk_free_rate / periods_per_year

    mean_return = np.mean(excess)
    std_return = np.std(excess, ddof=1)

    if std_return == 0:
        return 0.0

    # 年率換算
    return (mean_return / std_return) * np.sqrt(periods_per_year)


def sharpe_with_stats(sharpes: List[float]) -> Dict[str, float]:
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
        "ci95": [round(ci95_low, 6), round(ci95_high, 6)]
    }


def calculate_delta_sharpe(
    base_sharpes: List[float],
    with_feature_sharpes: List[float],
    min_samples: int = 10000
) -> Optional[Dict[str, float]]:
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

    delta_mean = with_stats["mean"] - base_stats["mean"]
    delta_std = np.sqrt(with_stats["std"]**2 + base_stats["std"]**2)  # 誤差伝播

    # 95%信頼区間（簡易計算）
    delta_ci95_low = delta_mean - 1.96 * delta_std
    delta_ci95_high = delta_mean + 1.96 * delta_std

    return {
        "mean": round(delta_mean, 6),
        "std": round(delta_std, 6),
        "ci95": [round(delta_ci95_low, 6), round(delta_ci95_high, 6)]
    }


def validate_ablation_results(results: Dict) -> bool:
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