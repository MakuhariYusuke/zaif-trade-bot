"""
Type definitions for Scalability and Operations
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ScalingDirection(Enum):
    """スケーリング方向"""

    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingStrategy(Enum):
    """スケーリング戦略"""

    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"


class ResourceType(Enum):
    """リソースタイプ"""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


class ScalingEventType(Enum):
    """スケーリングイベントタイプ"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    AUTO_SCALE = "auto_scale"
    MANUAL_SCALE = "manual_scale"
    FAILURE = "failure"


@dataclass
class ResourceMetrics:
    """リソース使用量メトリクス"""

    resource_type: ResourceType
    current_usage: float
    capacity: float
    utilization_percent: float
    timestamp: datetime
    instance_id: str


@dataclass
class ScalingThreshold:
    """スケーリング閾値"""

    resource_type: ResourceType
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period_seconds: int


@dataclass
class ScalingDecision:
    """スケーリング決定"""

    direction: ScalingDirection
    reason: str
    target_instances: int
    current_instances: int
    confidence_score: float
    estimated_cost_impact: float
    timestamp: datetime


@dataclass
class ScalingEvent:
    """スケーリングイベント"""

    event_id: str
    event_type: ScalingEventType
    decision: ScalingDecision
    success: bool
    execution_time_seconds: float
    error_message: Optional[str]
    timestamp: datetime


@dataclass
class InstanceInfo:
    """インスタンス情報"""

    instance_id: str
    instance_type: str
    region: str
    availability_zone: str
    launch_time: datetime
    state: str
    cost_per_hour: float


@dataclass
class LoadBalancerConfig:
    """ロードバランサー設定"""

    algorithm: str  # "round_robin", "least_connections", "ip_hash"
    health_check_interval: int
    health_check_timeout: int
    unhealthy_threshold: int
    healthy_threshold: int


@dataclass
class CostOptimizationRule:
    """コスト最適化ルール"""

    rule_name: str
    condition: str
    action: str
    priority: int
    enabled: bool
    last_applied: Optional[datetime]


@dataclass
class OperationalMetrics:
    """運用メトリクス"""

    total_instances: int
    active_instances: int
    average_cpu_utilization: float
    average_memory_utilization: float
    total_cost_per_hour: float
    scaling_events_last_24h: int
    failed_scaling_events: int
    uptime_percentage: float
    timestamp: datetime


@dataclass
class BackupConfig:
    """バックアップ設定"""

    backup_interval_hours: int
    retention_period_days: int
    backup_storage_path: str
    compression_enabled: bool
    encryption_enabled: bool


@dataclass
class MaintenanceWindow:
    """メンテナンスウィンドウ"""

    window_id: str
    start_time: datetime
    duration_hours: int
    description: str
    allowed_operations: List[str]
    notification_sent: bool


# 統合運用管理用の型定義


class SystemHealth(Enum):
    """システムヘルスステータス"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class IntegrationStatus(Enum):
    """統合ステータス"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class IntegrationStatus:
    """統合ステータス"""

    monitoring_active: bool
    safety_active: bool
    scalability_active: bool
    online_learning_active: bool
    last_integration_check: datetime


@dataclass
class OperationalMetrics:
    """運用メトリクス"""

    uptime_seconds: float
    total_requests: int
    error_rate: float
    average_response_time: float
    resource_utilization: Dict[str, float]
    last_updated: datetime


@dataclass
class AlertSummary:
    """アラート概要"""

    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    top_alerts: List[Any]  # Alertオブジェクトのリスト
    last_updated: datetime


@dataclass
class RecoveryAction:
    """回復アクション"""

    action_type: str
    reason: str
    timestamp: datetime
    system_state: Dict[str, Any]
    recommended_actions: List[str]
