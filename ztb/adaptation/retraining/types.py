"""
Type definitions for Automatic Retraining Triggers
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional


class TriggerType(Enum):
    """トリガータイプ"""

    PERFORMANCE = "performance"  # パフォーマンスベース
    DATA_DISTRIBUTION = "data_distribution"  # データ分布変化
    TIME_BASED = "time_based"  # 時間ベース
    VOLUME_BASED = "volume_based"  # 出来高ベース
    MANUAL = "manual"  # 手動トリガー


class TriggerPriority(Enum):
    """トリガー優先度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerStatus(Enum):
    """トリガーステータス"""

    INACTIVE = "inactive"  # 非アクティブ
    MONITORING = "monitoring"  # 監視中
    TRIGGERED = "triggered"  # トリガー発動
    EXECUTING = "executing"  # 実行中
    COMPLETED = "completed"  # 完了
    FAILED = "failed"  # 失敗


@dataclass
class TriggerCondition:
    """トリガー条件"""

    trigger_type: TriggerType
    metric_name: str
    operator: str  # "gt", "lt", "gte", "lte", "eq", "ne"
    threshold: float
    duration_minutes: int  # 条件を満たす必要のある時間
    cooldown_minutes: int  # トリガー後のクールダウン時間
    priority: TriggerPriority = TriggerPriority.MEDIUM


@dataclass
class MLPerformanceMetrics:
    """パフォーマンス指標"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    timestamp: datetime



@dataclass
class DataDistributionMetrics:
    """データ分布指標"""

    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    feature_skewness: Dict[str, float]
    feature_kurtosis: Dict[str, float]
    sample_size: int
    timestamp: datetime


@dataclass
class RetrainingRequest:
    """再訓練リクエスト"""

    request_id: str
    trigger_type: TriggerType
    trigger_reason: str
    priority: TriggerPriority
    requested_at: datetime
    estimated_duration: Optional[timedelta]
    required_resources: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingResult:
    """再訓練結果"""

    request_id: str
    success: bool
    new_model_path: Optional[str]
    performance_improvement: float
    training_duration: timedelta
    completed_at: datetime
    error_message: Optional[str] = None
    metrics_before: Optional[MLPerformanceMetrics] = None
    metrics_after: Optional[MLPerformanceMetrics] = None


@dataclass
class TriggerState:
    """トリガー状態"""

    trigger_id: str
    condition: TriggerCondition
    status: TriggerStatus
    last_check: datetime
    last_triggered: Optional[datetime]
    consecutive_violations: int
    cooldown_until: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingSchedule:
    """再訓練スケジュール"""

    schedule_id: str
    trigger_type: TriggerType
    cron_expression: Optional[str]  # cron形式のスケジュール
    interval_minutes: Optional[int]  # インターバル（分）
    next_run: datetime
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0


@dataclass
class ResourceUsage:
    """リソース使用量"""

    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float]
    disk_usage_mb: float
    timestamp: datetime


@dataclass
class RetrainingHistory:
    """再訓練履歴"""

    request_id: str
    trigger_type: TriggerType
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    performance_change: float
    resource_usage: Optional[ResourceUsage]
    error_details: Optional[str]
