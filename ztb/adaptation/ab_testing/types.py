"""
Type definitions for A/B Testing Framework
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class TrafficSplitType(Enum):
    """トラフィック分割タイプ"""
    PROBABILISTIC = "probabilistic"  # 確率的分割
    TIME_BASED = "time_based"       # 時間ベース分割
    CONDITION_BASED = "condition_based"  # 条件ベース分割


class TestStatus(Enum):
    """テストステータス"""
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ROLLED_BACK = "rolled_back"


class StatisticalTest(Enum):
    """統計的検定タイプ"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    CONFIDENCE_INTERVAL = "confidence_interval"


@dataclass
class TestVariant:
    """テストバリアント定義"""
    name: str
    model_path: str
    traffic_percentage: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestMetrics:
    """テストメトリクス"""
    variant_name: str
    total_trades: int
    profitable_trades: int
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    sample_size: int
    timestamp: datetime


@dataclass
class StatisticalResult:
    """統計的検定結果"""
    test_type: StatisticalTest
    p_value: float
    confidence_level: float
    effect_size: float
    is_significant: bool
    interpretation: str


@dataclass
class TestResult:
    """A/Bテスト結果"""
    test_id: str
    status: TestStatus
    winner_variant: Optional[str]
    confidence_level: float
    statistical_results: List[StatisticalResult]
    metrics_comparison: Dict[str, TestMetrics]
    start_time: datetime
    end_time: Optional[datetime]
    rollback_triggered: bool
    rollback_reason: Optional[str]


@dataclass
class RollbackCondition:
    """ロールバック条件"""
    metric_name: str
    threshold: float
    comparison: str  # "less_than", "greater_than", "absolute_change"
    baseline_value: Optional[float]
    consecutive_periods: int


@dataclass
class MultiArmedBanditState:
    """マルチアームドバンディット状態"""
    variant_rewards: Dict[str, float]
    variant_counts: Dict[str, int]
    total_pulls: int
    epsilon: float  # 探索率
    last_updated: datetime