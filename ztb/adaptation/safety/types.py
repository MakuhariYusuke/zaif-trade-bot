"""
Type definitions for Safety Mechanisms and Fallback
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class SafetyLevel(Enum):
    """安全レベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    WARNING = "warning"


class FallbackStrategy(Enum):
    """フォールバック戦略"""

    ROLLBACK_TO_PREVIOUS = "rollback_to_previous"
    SWITCH_TO_BASELINE = "switch_to_baseline"
    REDUCE_POSITION_SIZE = "reduce_position_size"
    STOP_TRADING = "stop_trading"
    MANUAL_INTERVENTION = "manual_intervention"


class CircuitBreakerState(Enum):
    """サーキットブレーカー状態"""

    CLOSED = "closed"  # 正常動作
    OPEN = "open"  # 遮断中
    HALF_OPEN = "half_open"  # テスト中


class DeploymentPhase(Enum):
    """デプロイメントフェーズ"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


@dataclass
class SafetyThreshold:
    """安全閾値"""

    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "gte", "lte"
    safety_level: SafetyLevel
    description: str


@dataclass
class CircuitBreakerConfig:
    """サーキットブレーカー設定"""

    failure_threshold: int
    recovery_timeout_seconds: int
    monitoring_window_seconds: int
    success_threshold: int
    name: str


@dataclass
class CircuitBreaker:
    """サーキットブレーカー"""

    config: CircuitBreakerConfig
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_state_change: datetime


@dataclass
class FallbackAction:
    """フォールバックアクション"""

    strategy: FallbackStrategy
    trigger_conditions: List[SafetyThreshold]
    execution_order: int
    requires_approval: bool
    description: str
    rollback_model_path: Optional[str]


class FallbackStatus(Enum):
    """フォールバック状態"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIGGERED = "triggered"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentStage:
    """デプロイメントステージ"""

    phase: DeploymentPhase
    traffic_percentage: float
    duration_hours: int
    success_criteria: List[SafetyThreshold]
    rollback_triggers: List[SafetyThreshold]


@dataclass
class SafetyIncident:
    """安全インシデント"""

    id: str
    timestamp: datetime
    safety_level: SafetyLevel
    description: str
    triggered_by: str  # メトリクス名
    current_value: float
    threshold_value: float
    actions_taken: List[str]
    resolved: bool
    resolution_time: Optional[datetime]


@dataclass
class RiskAssessment:
    """リスク評価"""

    overall_risk_score: float
    risk_factors: Dict[str, float]
    mitigation_measures: List[str]
    acceptable_risk_threshold: float
    assessment_time: datetime
    recommended_actions: List[str]


@dataclass
class AnomalyPattern:
    """異常パターン"""

    pattern_id: str
    description: str
    detection_method: str
    severity_score: float
    false_positive_rate: float
    affected_metrics: List[str]
    mitigation_strategy: str


class AnomalyType(Enum):
    """異常タイプ"""

    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_CHANGE = "trend_change"
    DISTRIBUTION_SHIFT = "distribution_shift"
    PERFORMANCE_DROP = "performance_drop"
    SYSTEM_ERROR = "system_error"
    MARKET_ANOMALY = "market_anomaly"


@dataclass
class AnomalyDetection:
    """異常検知結果"""

    anomaly_type: AnomalyType
    confidence: float
    severity: SafetyLevel
    description: str
    affected_metrics: List[str]
    timestamp: datetime
    detection_method: str


class RecoveryStatus(Enum):
    """リカバリー状態"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class RecoveryStrategy(Enum):
    """リカバリ戦略"""

    GRADUAL_RECOVERY = "gradual_recovery"
    IMMEDIATE_RECOVERY = "immediate_recovery"
    CONSERVATIVE_RECOVERY = "conservative_recovery"
    AGGRESSIVE_RECOVERY = "aggressive_recovery"


class SafetyEvent(Enum):
    """安全イベント"""

    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    SYSTEM_DEGRADED = "system_degraded"
    RECOVERY_INITIATED = "recovery_initiated"
    FALLBACK_ACTIVATED = "fallback_activated"


class SafetyAction(Enum):
    """安全アクション"""

    MONITOR = "monitor"
    ALERT = "alert"
    REDUCE_RISK = "reduce_risk"
    STOP_TRADING = "stop_trading"
    ROLLBACK = "rollback"
    RECOVERY = "recovery"


class FallbackMode(Enum):
    """フォールバックモード"""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    GRADUAL = "gradual"
    IMMEDIATE = "immediate"


@dataclass
class SafetyMetrics:
    """安全メトリクス"""

    circuit_breakers_active: int
    incidents_last_24h: int
    fallback_activations: int
    manual_interventions: int
    system_uptime_percent: float
    risk_score_current: float
    timestamp: datetime
