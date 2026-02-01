"""
Monitoring report generator.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from ztb.types.alert_types import AlertLevel

# Avoid importing adaptation.monitoring at module import time to prevent
# circular imports (monitor imports ReportGenerator and reporting imports
# monitoring types). Use TYPE_CHECKING for type hints only.
if TYPE_CHECKING:
    from ztb.adaptation.monitoring.types import MetricValue, ReportData, AlertManager
else:
    MetricValue = Any
    ReportData = Any
    AlertManager = Any


class ReportGenerator:
    """Generate monitoring reports from collected metrics and alerts."""

    def __init__(self, config: Any):
        self.config = config

    def generate_report(
        self,
        metrics_collector: Any,
        alert_manager: Any,
        period_days: int = 7,
    ) -> ReportData:
        all_metrics = {}
        for metric_name in metrics_collector.metrics_buffer.keys():
            history = metrics_collector.get_metric_history(
                metric_name, hours=period_days * 24
            )
            if history:
                all_metrics[metric_name] = history

        statistics = self._calculate_statistics(all_metrics)
        trends = self._analyze_trends(all_metrics)
        alert_analysis = self._analyze_alerts(alert_manager, period_days)
        performance_analysis = self._analyze_performance(all_metrics)

        return ReportData(
            report_id=f"report_{int(time.time())}",
            generated_at=datetime.now(),
            period_days=period_days,
            statistics=statistics,
            trends=trends,
            alert_analysis=alert_analysis,
            performance_analysis=performance_analysis,
            recommendations=self._generate_recommendations(trends, alert_analysis),
        )

    def _calculate_statistics(
        self, metrics: Dict[str, List[MetricValue]]
    ) -> Dict[str, Dict[str, float]]:
        statistics: Dict[str, Dict[str, float]] = {}

        for metric_name, values in metrics.items():
            if not values:
                continue

            metric_values = [v.value for v in values]
            statistics[metric_name] = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values)),
                "median": float(np.median(metric_values)),
                "count": len(metric_values),
            }

        return statistics

    def _analyze_trends(self, metrics: Dict[str, List[MetricValue]]) -> Dict[str, str]:
        trends: Dict[str, str] = {}

        for metric_name, values in metrics.items():
            if len(values) < 2:
                trends[metric_name] = "insufficient_data"
                continue

            x = np.arange(len(values))
            y = np.array([v.value for v in values])

            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                if abs(slope) < 1e-6:
                    trends[metric_name] = "stable"
                elif slope > 0:
                    trends[metric_name] = "increasing"
                else:
                    trends[metric_name] = "decreasing"
            else:
                trends[metric_name] = "stable"

        return trends

    def _analyze_alerts(
        self, alert_manager: AlertManager, period_days: int
    ) -> Dict[str, Any]:
        alerts = alert_manager.get_alert_history(hours=period_days * 24)

        return {
            "total_alerts": len(alerts),
            "by_level": {level.value: 0 for level in AlertLevel},
            "by_metric": {},
            "most_frequent_alerts": [],
        }

    def _analyze_performance(
        self, metrics: Dict[str, List[MetricValue]]
    ) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {}

        if "win_rate" in metrics:
            win_rates = [v.value for v in metrics["win_rate"]]
            analysis["win_rate_trend"] = (
                "improving" if win_rates[-1] > win_rates[0] else "declining"
            )

        if "total_pnl" in metrics:
            pnl_values = [v.value for v in metrics["total_pnl"]]
            analysis["pnl_trend"] = (
                "profitable" if pnl_values[-1] > 0 else "unprofitable"
            )

        return analysis

    def _generate_recommendations(
        self, trends: Dict[str, str], alert_analysis: Dict[str, Any]
    ) -> List[str]:
        recommendations: List[str] = []

        if trends.get("win_rate") == "decreasing":
            recommendations.append(
                "Consider reviewing trading strategy - win rate is declining"
            )

        if trends.get("max_drawdown") == "increasing":
            recommendations.append(
                "Implement additional risk controls - drawdown is increasing"
            )

        if alert_analysis.get("total_alerts", 0) > 10:
            recommendations.append(
                "High alert frequency detected - review system configuration"
            )

        return recommendations
