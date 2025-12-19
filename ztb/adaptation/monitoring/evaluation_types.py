"""
Type definitions for Continuous Evaluation and Monitoring
継続的評価と監視の型定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ztb.types.common import AlertLevel
from ztb.types.evaluation_types import AlertType, EvaluationResult
from ztb.types.evaluation_types import EvaluationMetrics


@dataclass
class MonitoringEvaluationResult(EvaluationResult):
    """監視拡張評価結果"""

    drift_severity: Optional[int] = None
    online_learning_metrics: Optional[Dict[str, Any]] = None
    overall_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class MonitoringAlert:
    """監視アラート"""

    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_timestamp: Optional[datetime] = None


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
    gpu_usage: Optional[float] = None
    network_io_bytes: Optional[int] = None
    disk_io_bytes: Optional[int] = None


@dataclass
class AlertAggregation:
    """アラート集計"""

    alert_type: AlertType
    time_window_minutes: int
    alert_count: int
    first_alert_timestamp: datetime
    last_alert_timestamp: datetime
    average_severity: float
    affected_components: List[str]


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
    top_recommendations: List[str]
    system_health_score: float


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""

    report_id: str
    generated_at: datetime
    period_days: int
    overall_performance_score: float
    key_metrics: Dict[str, float]
    alerts_summary: Dict[str, int]
    recommendations: List[str]
    charts_data: Dict[str, Any] = field(default_factory=dict)
    export_formats: List[str] = field(default_factory=lambda: ["pdf", "html", "json"])


@dataclass
class AnomalyReport:
    """異常レポート"""

    report_id: str
    detected_at: datetime
    anomaly_type: str
    severity_score: float
    affected_metrics: List[str]
    description: str
    recommended_actions: List[str]
    confidence_score: float
    false_positive_probability: float


@dataclass
class DriftAnalysisReport:
    """ドリフト分析レポート"""

    report_id: str
    analysis_timestamp: datetime
    drift_detected: bool
    drift_severity: int
    affected_features: List[str]
    drift_direction: str  # "positive", "negative", "neutral"
    confidence_interval: Dict[str, float]
    recommended_actions: List[str]
    mitigation_suggestions: List[str]


@dataclass
class SystemHealthReport:
    """システム健全性レポート"""

    report_id: str
    assessment_timestamp: datetime
    overall_health_score: float
    component_health_scores: Dict[str, float]
    critical_issues: List[str]
    warning_issues: List[str]
    recommendations: List[str]
    next_assessment_due: datetime


# コールバック型定義
AlertCallback = Callable[[MonitoringAlert], None]
EvaluationCallback = Callable[[EvaluationResult], None]
MetricsCallback = Callable[[SystemMetrics], None]
ReportCallback = Callable[[PerformanceReport], None]
