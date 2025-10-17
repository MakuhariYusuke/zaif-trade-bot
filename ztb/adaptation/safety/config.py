"""
Configuration management for Safety Mechanisms and Fallback
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .types import SafetyLevel, FallbackStrategy, SafetyThreshold, CircuitBreakerConfig, FallbackAction, DeploymentStage, DeploymentPhase


@dataclass
class SafetyConfig:
    """安全設定"""

    # 基本設定
    safety_enabled: bool = True
    safety_level: SafetyLevel = SafetyLevel.MEDIUM
    emergency_stop_enabled: bool = True

    # 安全閾値
    safety_thresholds: List[SafetyThreshold] = field(default_factory=list)

    # サーキットブレーカー設定
    circuit_breakers: List[CircuitBreakerConfig] = field(default_factory=list)

    # フォールバック設定
    fallback_actions: List[FallbackAction] = field(default_factory=list)
    max_concurrent_fallbacks: int = 1

    # 段階的デプロイメント設定
    deployment_stages: List[DeploymentStage] = field(default_factory=list)
    auto_progression_enabled: bool = False
    progression_delay_hours: int = 24

    # リスク管理設定
    risk_assessment_interval_hours: int = 6
    acceptable_risk_threshold: float = 0.7
    risk_factors_weights: Dict[str, float] = field(default_factory=lambda: {
        "market_volatility": 0.3,
        "model_performance": 0.4,
        "system_stability": 0.3
    })

    # 異常検知設定
    anomaly_detection_enabled: bool = True
    anomaly_detection_window: int = 1000
    anomaly_threshold_sigma: float = 3.0
    false_positive_tolerance: float = 0.05

    # 監視設定
    monitoring_interval_seconds: int = 30
    safety_check_interval_seconds: int = 60
    incident_reporting_enabled: bool = True
    incident_storage_path: str = "logs/safety_incidents"

    def __post_init__(self):
        """設定の検証と初期化"""
        if self.acceptable_risk_threshold <= 0 or self.acceptable_risk_threshold > 1:
            raise ValueError("acceptable_risk_threshold must be between 0 and 1")

        # デフォルトの安全閾値を設定
        if not self.safety_thresholds:
            self.safety_thresholds = [
                SafetyThreshold(
                    metric_name="max_drawdown",
                    threshold_value=0.20,
                    comparison="gte",
                    safety_level=SafetyLevel.CRITICAL,
                    description="Maximum drawdown exceeded 20%"
                ),
                SafetyThreshold(
                    metric_name="daily_pnl",
                    threshold_value=-0.15,
                    comparison="lte",
                    safety_level=SafetyLevel.HIGH,
                    description="Daily P&L dropped below -15%"
                ),
                SafetyThreshold(
                    metric_name="win_rate",
                    threshold_value=0.35,
                    comparison="lte",
                    safety_level=SafetyLevel.MEDIUM,
                    description="Win rate dropped below 35%"
                ),
                SafetyThreshold(
                    metric_name="api_error_rate",
                    threshold_value=0.10,
                    comparison="gte",
                    safety_level=SafetyLevel.HIGH,
                    description="API error rate exceeded 10%"
                )
            ]

        # デフォルトのサーキットブレーカーを設定
        if not self.circuit_breakers:
            self.circuit_breakers = [
                CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout_seconds=300,
                    monitoring_window_seconds=600,
                    success_threshold=3,
                    name="trading_circuit_breaker"
                ),
                CircuitBreakerConfig(
                    failure_threshold=10,
                    recovery_timeout_seconds=600,
                    monitoring_window_seconds=1800,
                    success_threshold=5,
                    name="api_circuit_breaker"
                )
            ]

        # デフォルトのフォールバックアクションを設定
        if not self.fallback_actions:
            self.fallback_actions = [
                FallbackAction(
                    strategy=FallbackStrategy.REDUCE_POSITION_SIZE,
                    trigger_conditions=[
                        SafetyThreshold("daily_pnl", -0.10, "lte", SafetyLevel.MEDIUM, "Daily P&L warning")
                    ],
                    execution_order=1,
                    requires_approval=False,
                    description="Reduce position size by 50% when daily P&L drops below -10%",
                    rollback_model_path=None
                ),
                FallbackAction(
                    strategy=FallbackStrategy.ROLLBACK_TO_PREVIOUS,
                    trigger_conditions=[
                        SafetyThreshold("max_drawdown", 0.15, "gte", SafetyLevel.HIGH, "High drawdown")
                    ],
                    execution_order=2,
                    requires_approval=True,
                    description="Rollback to previous model version when drawdown exceeds 15%",
                    rollback_model_path="models/previous_version"
                ),
                FallbackAction(
                    strategy=FallbackStrategy.STOP_TRADING,
                    trigger_conditions=[
                        SafetyThreshold("max_drawdown", 0.25, "gte", SafetyLevel.CRITICAL, "Critical drawdown")
                    ],
                    execution_order=3,
                    requires_approval=True,
                    description="Stop all trading when drawdown exceeds 25%",
                    rollback_model_path=None
                )
            ]

        # デフォルトのデプロイメントステージを設定
        if not self.deployment_stages:
            self.deployment_stages = [
                DeploymentStage(
                    phase=DeploymentPhase.STAGING,
                    traffic_percentage=0.1,
                    duration_hours=24,
                    success_criteria=[
                        SafetyThreshold("win_rate", 0.40, "gte", SafetyLevel.MEDIUM, "Minimum win rate")
                    ],
                    rollback_triggers=[
                        SafetyThreshold("max_drawdown", 0.05, "gte", SafetyLevel.HIGH, "Staging drawdown limit")
                    ]
                ),
                DeploymentStage(
                    phase=DeploymentPhase.PRODUCTION,
                    traffic_percentage=1.0,
                    duration_hours=168,  # 1週間
                    success_criteria=[
                        SafetyThreshold("sharpe_ratio", 1.0, "gte", SafetyLevel.MEDIUM, "Minimum Sharpe ratio")
                    ],
                    rollback_triggers=[
                        SafetyThreshold("max_drawdown", 0.10, "gte", SafetyLevel.CRITICAL, "Production drawdown limit")
                    ]
                )
            ]


@dataclass
class EmergencyProtocol:
    """緊急時プロトコル"""

    protocol_name: str
    trigger_conditions: List[SafetyThreshold]
    actions: List[str]
    notification_channels: List[str]
    requires_immediate_action: bool
    documentation_url: Optional[str]

    def is_triggered(self, metrics: Dict[str, float]) -> bool:
        """プロトコルがトリガーされるか判定"""
        for condition in self.trigger_conditions:
            value = metrics.get(condition.metric_name)
            if value is None:
                continue

            if condition.comparison == "gt" and value > condition.threshold_value:
                return True
            elif condition.comparison == "lt" and value < condition.threshold_value:
                return True
            elif condition.comparison == "gte" and value >= condition.threshold_value:
                return True
            elif condition.comparison == "lte" and value <= condition.threshold_value:
                return True

        return False