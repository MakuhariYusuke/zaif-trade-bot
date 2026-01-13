"""
Common type definitions for evaluation and monitoring
評価と監視の共通型定義
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

try:
    import pandas as pd
except ImportError:
    pd = None


class AlertType(Enum):
    """アラートタイプ"""

    PERFORMANCE = "performance"
    SAFETY = "safety"
    DRIFT = "drift"
    SYSTEM = "system"


@dataclass
class EvaluationMetrics:
    """統合された評価メトリクス"""

    # ML/Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Financial metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Additional metrics
    consistency_score: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class EvaluationResult:
    """評価結果"""

    timestamp: datetime
    performance_metrics: Optional[EvaluationMetrics] = None
    safety_metrics: Optional[Dict[str, Any]] = None
    drift_detected: bool = False


class SummaryStats(TypedDict):
    """サマリー統計の型定義"""

    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    annual_return: float
    annual_volatility: float


class BenchmarkSummaryStats(TypedDict):
    """ベンチマーク比較サマリー統計の型定義"""

    best_benchmark: Optional[str]
    worst_benchmark: Optional[str]
    avg_information_ratio: float
    avg_alpha: float
    benchmark_correlations: Dict[str, float]


class MultiBenchmarkSummary(TypedDict):
    """複数ベンチマークサマリーの型定義"""

    comparisons: List[BenchmarkComparison]
    summary_stats: Dict[str, Any]


@dataclass
class BenchmarkComparison:
    """個別のベンチマーク比較結果"""

    benchmark_name: str
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    excess_returns: pd.Series
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    r_squared: float
    max_drawdown_diff: float
    win_rate_vs_benchmark: float


@dataclass
class RollingComparison:
    """ローリング比較結果"""

    window_size: int
    rolling_alpha: pd.Series
    rolling_beta: pd.Series
    rolling_tracking_error: pd.Series
    rolling_excess_returns: pd.Series


@dataclass
class BenchmarkComparisonResult:
    """包括的なベンチマーク比較結果"""

    strategy_performance: Dict[str, float]
    benchmark_performance: Dict[str, Dict[str, float]]
    comparisons: List[BenchmarkComparison]
    rolling_comparisons: Optional[List[RollingComparison]] = None
    multi_benchmark_summary: Optional[MultiBenchmarkSummary] = None
