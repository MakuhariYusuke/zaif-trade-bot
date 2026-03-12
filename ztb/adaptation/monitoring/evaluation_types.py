"""
Type definitions for Continuous Evaluation and Monitoring
継続的評価と監視の型定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ztb.types.common import AlertLevel
from ztb.types.evaluation_types import AlertType, EvaluationResult
from ztb.types.evaluation_types import EvaluationMetrics

@dataclass
class MonitoringEvaluationResult(EvaluationResult):
    """監視拡張評価結果"""

    drift_severity: int | None = None
    online_learning_metrics: dict[str, Any] | None = None
    overall_score: float | None = None
    recommendations: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: str | None = None

@dataclass
class MonitoringAlert:
    """監視アラート"""

    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: datetime | None = None
    acknowledged: bool = False
    acknowledged_timestamp: datetime | None = None

@dataclass
class SystemMetrics:
    """システムメトリクス"""

    timestamp: datetime
    cpu_usage: float  # パーセント
    memory_usage: float  # パーセント
    disk_usage: float  # パーセント
    network_connections: int
    active_threads: int

    # オプションの追加メトリクス
    gpu_usage: float | None = None
    network_io_bytes: int | None = None
    disk_io_bytes: int | None = None

@dataclass
class AlertAggregation:
    """アラート集計"""

    alert_type: AlertType
    time_window_minutes: int
    alert_count: int
    first_alert_timestamp: datetime
    last_alert_timestamp: datetime
    average_severity: float
    affected_components: list[str]

@dataclass
class EvaluationSummary:
    """評価サマリー"""

    period_start: datetime
    period_end: datetime
    total_evaluations: int
    average_score: float
    score_trend: str  # "improving", "declining", "stable"
    drift_detection_rate: float
    alert_count: int
    top_recommendations: list[str]
    system_health_score: float

@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""

    report_id: str
    generated_at: datetime
    period_days: int
    overall_performance_score: float
    key_metrics: dict[str, float]
    alerts_summary: dict[str, int]
    recommendations: list[str]
    charts_data: dict[str, Any] = field(default_factory=dict)
    export_formats: list[str] = field(default_factory=lambda: ["pdf", "html", "json"])

@dataclass
class AnomalyReport:
    """異常レポート"""

    report_id: str
    detected_at: datetime
    anomaly_type: str
    severity_score: float
    affected_metrics: list[str]
    description: str
    recommended_actions: list[str]
    confidence_score: float
    false_positive_probability: float

@dataclass
class DriftAnalysisReport:
    """ドリフト分析レポート"""

    report_id: str
    analysis_timestamp: datetime
    drift_detected: bool
    drift_severity: int
    affected_features: list[str]
    drift_direction: str  # "positive", "negative", "neutral"
    confidence_interval: dict[str, float]
    recommended_actions: list[str]
    mitigation_suggestions: list[str]

@dataclass
class SystemHealthReport:
    """システム健全性レポート"""

    report_id: str
    assessment_timestamp: datetime
    overall_health_score: float
    component_health_scores: dict[str, float]
    critical_issues: list[str]
    warning_issues: list[str]
    recommendations: list[str]
    next_assessment_due: datetime

# コールバック型定義
AlertCallback = Callable[[MonitoringAlert], None]
EvaluationCallback = Callable[[EvaluationResult], None]
MetricsCallback = Callable[[SystemMetrics], None]
ReportCallback = Callable[[PerformanceReport], None]
