"""
Configuration management for Continuous Evaluation and Monitoring
継続的評価と監視の設定管理
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import AlertCondition, AlertLevel, DashboardConfig, MetricType, ReportConfig


class EvaluationMode(Enum):
    """評価モード"""

    CONTINUOUS = "continuous"
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"


class AlertThreshold(Enum):
    """アラート閾値レベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """監視設定"""

    # 基本設定
    monitoring_enabled: bool = True
    collection_interval_seconds: int = 60
    retention_period_days: int = 30

    # メトリクス設定
    enabled_metric_types: List[MetricType] = field(
        default_factory=lambda: [
            MetricType.PERFORMANCE,
            MetricType.RISK,
            MetricType.SYSTEM,
        ]
    )

    # アラート設定
    alert_conditions: List[AlertCondition] = field(default_factory=list)
    alert_cooldown_seconds: int = 300
    max_concurrent_alerts: int = 10

    # 通知設定
    notification_channels: List[str] = field(default_factory=lambda: ["log", "email"])
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook_url: Optional[str] = None

    # ダッシュボード設定
    dashboard_config: DashboardConfig = field(
        default_factory=lambda: DashboardConfig(
            refresh_interval_seconds=300,
            metrics_to_display=[
                "win_rate",
                "total_pnl",
                "sharpe_ratio",
                "max_drawdown",
            ],
            chart_types={
                "win_rate": "line",
                "total_pnl": "area",
                "sharpe_ratio": "bar",
                "max_drawdown": "line",
            },
            alert_summary_enabled=True,
            historical_period_days=7,
        )
    )

    # レポート設定
    report_configs: List[ReportConfig] = field(default_factory=list)

    # 異常検知設定
    enable_anomaly_detection: bool = True
    anomaly_detection_window: int = 1000  # データポイント数
    anomaly_threshold_sigma: float = 3.0

    # パフォーマンス閾値
    performance_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "win_rate": {"warning": 0.45, "critical": 0.40},
            "sharpe_ratio": {"warning": 1.0, "critical": 0.5},
            "max_drawdown": {"warning": 0.10, "critical": 0.20},
            "daily_pnl": {"warning": -0.05, "critical": -0.10},
        }
    )

    # システム閾値
    system_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "cpu_usage_percent": {"warning": 80.0, "critical": 95.0},
            "memory_usage_mb": {"warning": 3072, "critical": 4096},
            "error_rate": {"warning": 0.05, "critical": 0.10},
            "api_response_time_ms": {"warning": 1000, "critical": 5000},
        }
    )

    def __post_init__(self):
        """設定の検証と初期化"""
        if self.collection_interval_seconds < 10:
            raise ValueError("collection_interval_seconds must be at least 10")

        if self.retention_period_days < 1:
            raise ValueError("retention_period_days must be at least 1")

        # デフォルトのアラート条件を設定
        if not self.alert_conditions:
            self.alert_conditions = [
                AlertCondition(
                    metric_name="win_rate",
                    operator="lt",
                    threshold=0.45,
                    duration_seconds=300,
                    cooldown_seconds=1800,
                    alert_level=AlertLevel.WARNING,
                    description="Win rate dropped below 45%",
                ),
                AlertCondition(
                    metric_name="max_drawdown",
                    operator="gt",
                    threshold=0.15,
                    duration_seconds=300,
                    cooldown_seconds=3600,
                    alert_level=AlertLevel.CRITICAL,
                    description="Max drawdown exceeded 15%",
                ),
                AlertCondition(
                    metric_name="cpu_usage_percent",
                    operator="gt",
                    threshold=90.0,
                    duration_seconds=300,
                    cooldown_seconds=1800,
                    alert_level=AlertLevel.WARNING,
                    description="CPU usage exceeded 90%",
                ),
            ]

        # デフォルトのレポート設定
        if not self.report_configs:
            self.report_configs = [
                ReportConfig(
                    report_type="daily",
                    include_metrics=["win_rate", "total_pnl", "sharpe_ratio"],
                    include_alerts=True,
                    include_charts=True,
                    recipients=self.email_recipients,
                    storage_path="reports/daily",
                )
            ]


@dataclass
class AlertEscalationRule:
    """アラートエスカレーションルール"""

    alert_level: AlertLevel
    escalation_delay_minutes: int
    escalate_to_level: AlertLevel
    additional_recipients: List[str]
    escalation_message: str

    def should_escalate(self, alert_age_minutes: int) -> bool:
        """エスカレーションが必要か判定"""
        return alert_age_minutes >= self.escalation_delay_minutes


@dataclass
class ScalabilityConfig:
    """スケーラビリティ設定"""

    # 自動スケーリング設定
    auto_scaling_enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8  # CPU使用率80%以上でスケールアップ
    scale_down_threshold: float = 0.3  # CPU使用率30%以下でスケールダウン
    cooldown_period_seconds: int = 300  # スケーリング後のクールダウン期間

    # 負荷分散設定
    load_balancing_enabled: bool = True
    load_distribution_algorithm: str = (
        "round_robin"  # round_robin, least_connections, weighted
    )
    max_connections_per_instance: int = 1000
    connection_timeout_seconds: int = 30

    # リソース最適化設定
    resource_optimization_enabled: bool = True
    cost_optimization_target: float = 0.1  # コスト削減目標（10%）
    resource_overprovisioning_limit: float = 0.2  # リソース過剰プロビジョニング制限
    optimization_check_interval_minutes: int = 60

    # デプロイメント設定
    deployment_strategy: str = "rolling"
    max_concurrent_deployments: int = 3
    deployment_timeout_minutes: int = 30
    rollback_on_failure: bool = True

    # 運用設定
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    monitoring_retention_days: int = 30
    maintenance_window_start: str = "02:00"  # HH:MM形式
    maintenance_window_duration_hours: int = 4

    def __post_init__(self):
        """設定検証"""
        if self.min_instances < 1:
            raise ValueError("min_instances must be at least 1")
        if self.max_instances < self.min_instances:
            raise ValueError(
                "max_instances must be greater than or equal to min_instances"
            )
        if not (0.0 < self.scale_up_threshold <= 1.0):
            raise ValueError("scale_up_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.scale_down_threshold < self.scale_up_threshold):
            raise ValueError(
                "scale_down_threshold must be less than scale_up_threshold"
            )


@dataclass
class EvaluationConfig:
    """評価設定"""

    # 評価間隔（秒）
    evaluation_interval_seconds: int = 60

    # アラートチェック間隔（秒）
    alert_check_interval_seconds: int = 30

    # メトリクス収集間隔（秒）
    metrics_collection_interval_seconds: int = 60

    # 評価履歴保持期間（時間）
    evaluation_history_retention_hours: int = 168  # 1週間

    # システムメトリクス保持期間（時間）
    system_metrics_retention_hours: int = 24

    # 評価モード
    evaluation_mode: EvaluationMode = EvaluationMode.CONTINUOUS

    # パフォーマンス閾値
    performance_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "min_accuracy": 0.4,
            "max_drawdown_threshold": 0.25,
            "min_sharpe_ratio": 1.0,
            "min_win_rate": 0.45,
        }
    )

    # 安全閾値
    safety_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "max_anomalies": 5,
            "min_safety_score": 0.6,
            "max_response_time_ms": 1000,
        }
    )

    # ドリフト閾値
    drift_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "drift_severity_threshold": 3.0,
            "drift_detection_confidence": 0.8,
        }
    )

    # アラート設定
    alert_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_email_alerts": False,
            "enable_slack_alerts": False,
            "alert_cooldown_minutes": 15,
            "max_alerts_per_hour": 10,
        }
    )

    # 評価スコアの重み付け
    evaluation_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "performance_weight": 0.4,
            "safety_weight": 0.3,
            "drift_weight": 0.3,
        }
    )

    # レポート設定
    report_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "generate_daily_reports": True,
            "generate_weekly_reports": True,
            "include_system_metrics": True,
            "include_detailed_analysis": False,
        }
    )

    # 異常検知設定
    anomaly_detection: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_statistical_anomaly": True,
            "enable_ml_anomaly": False,
            "anomaly_sensitivity": 0.8,
            "baseline_window_hours": 24,
        }
    )

    # 自動対応設定
    auto_response: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_auto_restart": False,
            "enable_auto_rollback": False,
            "auto_response_threshold": AlertThreshold.CRITICAL,
            "max_auto_actions_per_hour": 3,
        }
    )


@dataclass
class AlertConfig:
    """アラート設定"""

    # アラートレベルごとの設定
    alert_levels: Dict[AlertThreshold, Dict[str, Any]] = field(
        default_factory=lambda: {
            AlertThreshold.LOW: {
                "enabled": True,
                "notification_channels": ["log"],
                "escalation_time_minutes": 60,
                "auto_resolve": True,
            },
            AlertThreshold.MEDIUM: {
                "enabled": True,
                "notification_channels": ["log", "console"],
                "escalation_time_minutes": 30,
                "auto_resolve": False,
            },
            AlertThreshold.HIGH: {
                "enabled": True,
                "notification_channels": ["log", "console", "email"],
                "escalation_time_minutes": 15,
                "auto_resolve": False,
            },
            AlertThreshold.CRITICAL: {
                "enabled": True,
                "notification_channels": ["log", "console", "email", "slack"],
                "escalation_time_minutes": 5,
                "auto_resolve": False,
            },
        }
    )

    # アラートタイプごとの設定
    alert_types: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "performance": {
                "enabled": True,
                "aggregation_window_minutes": 10,
                "min_occurrences_for_alert": 3,
            },
            "safety": {
                "enabled": True,
                "aggregation_window_minutes": 5,
                "min_occurrences_for_alert": 1,
            },
            "drift": {
                "enabled": True,
                "aggregation_window_minutes": 15,
                "min_occurrences_for_alert": 2,
            },
            "system": {
                "enabled": True,
                "aggregation_window_minutes": 5,
                "min_occurrences_for_alert": 1,
            },
        }
    )


@dataclass
class ContinuousMonitoringConfig:
    """継続的監視設定"""

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # グローバル設定
    enable_monitoring: bool = True
    log_level: str = "INFO"
    enable_debug_mode: bool = False

    # データ保持設定
    data_retention: Dict[str, int] = field(
        default_factory=lambda: {
            "evaluation_history_days": 30,
            "alert_history_days": 90,
            "metrics_history_days": 7,
            "logs_days": 30,
        }
    )

    # 外部統合設定
    integrations: Dict[str, Any] = field(
        default_factory=lambda: {
            "prometheus_enabled": False,
            "grafana_enabled": False,
            "datadog_enabled": False,
            "slack_webhook_url": None,
            "email_smtp_config": None,
        }
    )


# デフォルト設定
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_ALERT_CONFIG = AlertConfig()
DEFAULT_CONTINUOUS_MONITORING_CONFIG = ContinuousMonitoringConfig()
