"""
Configuration management for Scalability and Operations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .types import (
    BackupConfig,
    CostOptimizationRule,
    LoadBalancerConfig,
    ScalingDecision,
    ScalingStrategy,
    ScalingThreshold,
)


@dataclass
class IntegratedOperationsConfig:
    """統合運用管理設定"""

    # 統合運用管理の有効化
    integrated_operations_enabled: bool = True

    # 各コンポーネントの設定
    monitoring_enabled: bool = True
    safety_enabled: bool = True
    scalability_enabled: bool = True
    online_learning_enabled: bool = True

    # 統合監視設定
    health_check_interval_seconds: int = 30
    system_status_update_interval_seconds: int = 60
    metrics_collection_interval_seconds: int = 15

    # アラート設定
    alert_check_interval_seconds: int = 10
    max_alerts_per_hour: int = 100
    alert_cooldown_seconds: int = 300

    # 緊急停止設定
    emergency_shutdown_enabled: bool = True
    emergency_shutdown_timeout_seconds: int = 30
    critical_error_threshold: int = 5

    # コンポーネント連携設定
    component_sync_interval_seconds: int = 60
    max_component_failures: int = 3
    component_restart_delay_seconds: int = 10

    # パフォーマンス監視設定
    performance_monitoring_enabled: bool = True
    performance_alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "cpu_usage_percent": 85.0,
            "memory_usage_percent": 90.0,
            "response_time_ms": 5000.0,
            "error_rate_percent": 5.0,
        }
    )

    # 自動回復設定
    auto_recovery_enabled: bool = True
    recovery_attempt_limit: int = 3
    recovery_cooldown_seconds: int = 300

    def __post_init__(self):
        """設定の検証"""
        if self.health_check_interval_seconds <= 0:
            raise ValueError("health_check_interval_seconds must be positive")

        if self.critical_error_threshold <= 0:
            raise ValueError("critical_error_threshold must be positive")


@dataclass
class OperationsConfig:
    """運用設定"""

    # 基本設定
    operations_enabled: bool = True
    monitoring_interval_seconds: int = 60
    max_instances: int = 10
    min_instances: int = 1

    # 統合運用管理設定
    integrated_config: IntegratedOperationsConfig = field(
        default_factory=IntegratedOperationsConfig
    )

    # 基本設定
    operations_enabled: bool = True
    monitoring_interval_seconds: int = 60
    max_instances: int = 10
    min_instances: int = 1

    # スケーリング設定
    scaling_strategy: ScalingStrategy = ScalingStrategy.CPU_BASED
    scaling_thresholds: List[ScalingThreshold] = field(default_factory=list)
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600

    # 予測スケーリング設定
    predictive_scaling_enabled: bool = False
    prediction_horizon_hours: int = 1
    prediction_interval_minutes: int = 15

    # ロードバランサー設定
    load_balancer_config: LoadBalancerConfig = field(
        default_factory=lambda: LoadBalancerConfig(
            algorithm="least_connections",
            health_check_interval=30,
            health_check_timeout=5,
            unhealthy_threshold=2,
            healthy_threshold=2,
        )
    )

    # コスト最適化設定
    cost_optimization_enabled: bool = True
    cost_optimization_rules: List[CostOptimizationRule] = field(default_factory=list)
    target_utilization_percent: float = 70.0

    # バックアップ設定
    backup_config: BackupConfig = field(
        default_factory=lambda: BackupConfig(
            backup_interval_hours=24,
            retention_period_days=30,
            backup_storage_path="backups/operations",
            compression_enabled=True,
            encryption_enabled=True,
        )
    )

    # メンテナンス設定
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    emergency_maintenance_allowed: bool = False

    # リソース制限
    resource_limits: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "cpu": {"max_percent": 80.0, "min_percent": 20.0},
            "memory": {"max_percent": 85.0, "min_percent": 30.0},
            "gpu": {"max_percent": 90.0, "min_percent": 10.0},
        }
    )

    # 通知設定
    notification_channels: List[str] = field(default_factory=lambda: ["log", "email"])
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "scaling_failure_rate": 0.1,
            "cost_overrun_percent": 20.0,
            "instance_downtime_percent": 5.0,
        }
    )

    def __post_init__(self):
        """設定の検証と初期化"""
        if self.max_instances < self.min_instances:
            raise ValueError(
                "max_instances must be greater than or equal to min_instances"
            )

        if (
            self.target_utilization_percent <= 0
            or self.target_utilization_percent > 100
        ):
            raise ValueError("target_utilization_percent must be between 0 and 100")

        # デフォルトのスケーリング閾値を設定
        if not self.scaling_thresholds:
            self.scaling_thresholds = [
                ScalingThreshold(
                    resource_type="cpu",
                    scale_up_threshold=75.0,
                    scale_down_threshold=30.0,
                    cooldown_period_seconds=300,
                ),
                ScalingThreshold(
                    resource_type="memory",
                    scale_up_threshold=80.0,
                    scale_down_threshold=40.0,
                    cooldown_period_seconds=300,
                ),
            ]

        # デフォルトのコスト最適化ルールを設定
        if not self.cost_optimization_rules:
            self.cost_optimization_rules = [
                CostOptimizationRule(
                    rule_name="off_peak_scaling",
                    condition="hour between 22 and 6",
                    action="scale_down_to_minimum",
                    priority=1,
                    enabled=True,
                    last_applied=None,
                ),
                CostOptimizationRule(
                    rule_name="high_cost_alert",
                    condition="hourly_cost > budget_limit * 1.2",
                    action="send_alert_and_scale_down",
                    priority=2,
                    enabled=True,
                    last_applied=None,
                ),
            ]

        # デフォルトのメンテナンスウィンドウを設定
        if not self.maintenance_windows:
            self.maintenance_windows = [
                {
                    "name": "weekly_maintenance",
                    "day_of_week": "sunday",
                    "start_hour": 2,
                    "duration_hours": 4,
                    "description": "Weekly system maintenance and updates",
                }
            ]


@dataclass
class ScalingPolicy:
    """スケーリングポリシー"""

    policy_name: str
    scaling_strategy: ScalingStrategy
    target_resource: str
    target_value: float
    cooldown_seconds: int
    enabled: bool

    def evaluate(self, current_metrics: Dict[str, float]) -> ScalingDecision:
        """ポリシーに基づいてスケーリング決定を評価"""
        # 実際の実装ではメトリクスに基づいてスケーリング決定を行う
        # ここではダミーの実装
        import datetime

        from .types import ScalingDecision, ScalingDirection

        return ScalingDecision(
            direction=ScalingDirection.NONE,
            reason="Policy evaluation placeholder",
            target_instances=1,
            current_instances=1,
            confidence_score=0.5,
            estimated_cost_impact=0.0,
            timestamp=datetime.datetime.now(),
        )
