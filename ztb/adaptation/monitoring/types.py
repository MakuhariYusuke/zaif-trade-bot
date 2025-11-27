"""
Type definitions for Continuous Evaluation and Monitoring
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    # Import alert types from common types to avoid duplication and maintain
    # a single source-of-truth for Alert types. Import may fail early if the
    # central types module is not yet importable; in that case, fall back to
    # Any-based placeholders to avoid hard import-time failures during
    # diagnostics (this preserves runtime behaviour and helps imports continue).
    from ztb.types.alert_types import (
        Alert,
        AlertCondition,
        AlertLevel,
        AlertStatus,
    )
except Exception:
    # Fallback to Any so the module can be imported even if alert_types is
    # temporarily unavailable due to circular imports while other modules are
    # being processed.
    Alert = Any
    AlertCondition = Any
    AlertLevel = Any
    AlertStatus = Any

# Explicitly define the module export list for clarity and to reduce reliance
# on partially initialized modules in import paths.
__all__ = [
    "MetricType",
    "MetricValue",
    "TradingPerformanceMetrics",
    "RiskMetrics",
    "SystemMetrics",
    "MarketMetrics",
    "AdaptationMetrics",
    "DashboardConfig",
    "ReportConfig",
    "Notification",
    "TimeSeriesData",
    "DashboardData",
    "ReportData",
    "SafetyLevel",
    "AnomalyType",
    "FallbackType",
    "AnomalyDetection",
    "SafetyCheck",
    "FallbackAction",
    "SafetyStatus",
    "RecoveryPlan",
    "ScalingStrategy",
    "ResourceType",
    "ScalingDecision",
    "DeploymentStatus",
    "ResourceUsage",
    "ScalingAction",
    "LoadDistribution",
    "DeploymentPlan",
    "CostOptimization",
    "ScalabilityMetrics",
    # Alert types re-exported from central module
    "Alert",
    "AlertCondition",
    "AlertLevel",
    "AlertStatus",
]


class MetricType(Enum):
    """メトリクスタイプ"""

    PERFORMANCE = "performance"  # パフォーマンスメトリクス
    RISK = "risk"  # リスクメトリクス
    SYSTEM = "system"  # システムメトリクス
    MARKET = "market"  # 市場メトリクス
    ADAPTATION = "adaptation"  # 適応メトリクス


@dataclass
class MetricValue:
    """メトリクス値"""

    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingPerformanceMetrics:
    """取引パフォーマンスメトリクス"""

    total_trades: int
    profitable_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percentage: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_percentage: float
    calmar_ratio: float
    alpha: float
    beta: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    timestamp: datetime


@dataclass
class RiskMetrics:
    """リスクメトリクス"""

    value_at_risk_95: float
    expected_shortfall_95: float
    volatility: float
    downside_volatility: float
    beta_to_market: float
    correlation_to_market: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float]
    disk_usage_percent: float
    network_latency_ms: float
    api_response_time_ms: float
    error_rate: float
    uptime_seconds: float
    timestamp: datetime


@dataclass
class MarketMetrics:
    """市場メトリクス"""

    volume_24h: float
    price_volatility_24h: float
    spread_bps: float
    market_depth: float
    order_book_imbalance: float
    timestamp: datetime


@dataclass
class AdaptationMetrics:
    """適応メトリクス"""

    drift_score: float
    adaptation_frequency: int
    model_version: str
    last_adaptation_time: datetime
    performance_impact: float
    stability_score: float
    timestamp: datetime


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""

    refresh_interval_seconds: int
    metrics_to_display: List[str]
    chart_types: Dict[str, str]
    alert_summary_enabled: bool
    historical_period_days: int


@dataclass
class ReportConfig:
    """レポート設定"""

    report_type: str  # "daily", "weekly", "monthly"
    include_metrics: List[str]
    include_alerts: bool
    include_charts: bool
    recipients: List[str]
    storage_path: str


@dataclass
class Notification:
    """通知"""

    notification_id: str
    alert: Alert
    channel: str
    sent_at: datetime
    status: str  # "sent", "failed", "pending"
    error_message: Optional[str] = None


@dataclass
class TimeSeriesData:
    """時系列データ"""

    metric_name: str
    timestamps: List[datetime]
    values: List[float]


@dataclass
class DashboardData:
    """ダッシュボードデータ"""

    timestamp: datetime
    latest_metrics: Dict[str, MetricValue]
    time_series: Dict[str, TimeSeriesData]
    alert_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]
    refresh_interval_seconds: int


@dataclass
class ReportData:
    """レポートデータ"""

    report_id: str
    generated_at: datetime
    period_days: int
    statistics: Dict[str, Dict[str, float]]
    trends: Dict[str, str]
    alert_analysis: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]


# 安全メカニズム関連の型定義


class SafetyLevel(Enum):
    """安全レベル"""

    NORMAL = "normal"  # 正常動作
    WARNING = "warning"  # 警告状態
    CRITICAL = "critical"  # 重大な問題
    EMERGENCY = "emergency"  # 緊急停止


class AnomalyType(Enum):
    """異常タイプ"""

    STATISTICAL = "statistical"  # 統計的異常
    PERFORMANCE = "performance"  # パフォーマンス異常
    SYSTEM = "system"  # システム異常
    MARKET = "market"  # 市場異常
    MODEL = "model"  # モデル異常


class FallbackType(Enum):
    """フォールバックタイプ"""

    GRADUAL = "gradual"  # 段階的ロールバック
    IMMEDIATE = "immediate"  # 即時ロールバック
    CONSERVATIVE = "conservative"  # 保守的モード
    SHUTDOWN = "shutdown"  # シャットダウン


@dataclass
class AnomalyDetection:
    """異常検知結果"""

    anomaly_type: AnomalyType
    metric_name: str
    detected_value: float
    expected_range: Tuple[float, float]
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class SafetyCheck:
    """安全チェック結果"""

    check_name: str
    safety_level: SafetyLevel
    passed: bool
    message: str
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class FallbackAction:
    """フォールバックアクション"""

    action_id: str
    fallback_type: FallbackType
    description: str
    priority: int
    estimated_duration_seconds: int
    rollback_steps: List[str]
    recovery_steps: List[str]


@dataclass
class SafetyStatus:
    """安全ステータス"""

    overall_safety_level: SafetyLevel
    active_anomalies: List[AnomalyDetection]
    recent_checks: List[SafetyCheck]
    active_fallbacks: List[FallbackAction]
    last_updated: datetime
    system_health_score: float  # 0.0-1.0


@dataclass
class RecoveryPlan:
    """回復計画"""

    plan_id: str
    triggered_by: str
    steps: List[str]
    estimated_completion_time: datetime
    success_criteria: List[str]
    rollback_plan: List[str]


# スケーラビリティと運用関連の型定義


class ScalingStrategy(Enum):
    """スケーリング戦略"""

    HORIZONTAL = "horizontal"  # 水平スケーリング（インスタンス追加）
    VERTICAL = "vertical"  # 垂直スケーリング（リソース増加）
    AUTO = "auto"  # 自動スケーリング
    MANUAL = "manual"  # 手動スケーリング


class ResourceType(Enum):
    """リソースタイプ"""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class ScalingDecision(Enum):
    """スケーリング決定"""

    SCALE_UP = "scale_up"  # スケールアップ
    SCALE_DOWN = "scale_down"  # スケールダウン
    NO_CHANGE = "no_change"  # 変更なし
    MAINTENANCE = "maintenance"  # メンテナンス


class DeploymentStatus(Enum):
    """デプロイメントステータス"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ResourceUsage:
    """リソース使用状況"""

    resource_type: ResourceType
    current_usage: float
    max_capacity: float
    utilization_percentage: float
    timestamp: datetime
    instance_id: str


@dataclass
class ScalingAction:
    """スケーリングアクション"""

    action_id: str
    scaling_decision: ScalingDecision
    scaling_strategy: ScalingStrategy
    target_instances: int
    current_instances: int
    reason: str
    estimated_cost_impact: float
    timestamp: datetime
    executed_by: str


@dataclass
class LoadDistribution:
    """負荷分散"""

    instance_id: str
    current_load: float
    max_load: float
    active_connections: int
    queue_length: int
    response_time_ms: float
    timestamp: datetime


@dataclass
class DeploymentPlan:
    """デプロイメント計画"""

    plan_id: str
    version: str
    target_instances: int
    rollout_strategy: str  # "rolling", "blue_green", "canary"
    rollback_plan: List[str]
    estimated_duration_minutes: int
    created_at: datetime
    status: DeploymentStatus


@dataclass
class CostOptimization:
    """コスト最適化"""

    optimization_id: str
    resource_type: ResourceType
    current_cost: float
    optimized_cost: float
    savings_percentage: float
    recommendations: List[str]
    implementation_status: str
    timestamp: datetime


@dataclass
class ScalabilityMetrics:
    """スケーラビリティメトリクス"""

    total_instances: int
    active_instances: int
    average_load: float
    peak_load: float
    scaling_events_count: int
    average_response_time_ms: float
    cost_per_hour: float
    uptime_percentage: float
    timestamp: datetime


@dataclass
class OperationsStatus:
    """運用ステータス"""

    system_status: str
    last_backup: datetime
    next_maintenance: datetime
    active_alerts: int
    pending_updates: int
    resource_utilization: Dict[str, float]
    performance_score: float
    timestamp: datetime
