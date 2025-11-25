"""
Continuous Evaluation and Monitoring System
リアルタイムパフォーマンス監視とアラートシステム
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ztb.types.alert_types import AlertCondition, AlertLevel, AlertStatus

from .config import MonitoringConfig
from .types import (
    Alert,
    DashboardData,
    MetricType,
    MetricValue,
    ReportData,
    TimeSeriesData,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """メトリクス収集器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.collection_thread: Optional[threading.Thread] = None
        self.is_collecting = False
        self.custom_collectors: Dict[str, Callable] = {}

    def start_collection(self) -> None:
        """収集開始"""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_worker, daemon=True
        )
        self.collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self) -> None:
        """収集停止"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")

    def _collection_worker(self) -> None:
        """収集ワーカー"""
        while self.is_collecting:
            try:
                # 標準メトリクス収集
                self._collect_standard_metrics()

                # カスタムメトリクス収集
                self._collect_custom_metrics()

                # 古いメトリクス削除
                self._cleanup_old_metrics()

                time.sleep(self.config.collection_interval_seconds)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(5)  # エラー時は短い待機

    def _collect_standard_metrics(self) -> None:
        """標準メトリクス収集"""
        timestamp = datetime.now()

        # パフォーマンスメトリクス
        if MetricType.PERFORMANCE in self.config.enabled_metric_types:
            perf_metrics = self._collect_performance_metrics()
            for name, value in perf_metrics.items():
                self._store_metric(name, value, MetricType.PERFORMANCE, timestamp)

        # リスクメトリクス
        if MetricType.RISK in self.config.enabled_metric_types:
            risk_metrics = self._collect_risk_metrics()
            for name, value in risk_metrics.items():
                self._store_metric(name, value, MetricType.RISK, timestamp)

        # システムメトリクス
        if MetricType.SYSTEM in self.config.enabled_metric_types:
            system_metrics = self._collect_system_metrics()
            for name, value in system_metrics.items():
                self._store_metric(name, value, MetricType.SYSTEM, timestamp)

        # 市場メトリクス
        if MetricType.MARKET in self.config.enabled_metric_types:
            market_metrics = self._collect_market_metrics()
            for name, value in market_metrics.items():
                self._store_metric(name, value, MetricType.MARKET, timestamp)

        # 適応メトリクス
        if MetricType.ADAPTATION in self.config.enabled_metric_types:
            adaptation_metrics = self._collect_adaptation_metrics()
            for name, value in adaptation_metrics.items():
                self._store_metric(name, value, MetricType.ADAPTATION, timestamp)

    def _collect_performance_metrics(self) -> Dict[str, float]:
        """パフォーマンスメトリクス収集"""
        # 実際の実装では取引システムからデータを取得
        # ここではサンプルデータを返す
        return {
            "win_rate": 0.55,
            "total_pnl": 1250.75,
            "sharpe_ratio": 1.23,
            "max_drawdown": -150.25,
            "avg_trade_duration": 45.5,
            "total_trades": 1250,
            "profit_factor": 1.45,
        }

    def _collect_risk_metrics(self) -> Dict[str, float]:
        """リスクメトリクス収集"""
        return {
            "value_at_risk": -250.50,
            "expected_shortfall": -375.75,
            "volatility": 0.15,
            "beta": 0.85,
            "correlation_coefficient": 0.65,
            "stress_test_loss": -500.00,
        }

    def _collect_system_metrics(self) -> Dict[str, float]:
        """システムメトリクス収集"""

        import psutil

        return {
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "network_connections": len(psutil.net_connections()),
            "process_count": len(psutil.pids()),
            "uptime_seconds": time.time() - psutil.boot_time(),
        }

    def _collect_market_metrics(self) -> Dict[str, float]:
        """市場メトリクス収集"""
        # 実際の実装では市場データソースから取得
        return {
            "market_volatility": 0.18,
            "liquidity_index": 0.75,
            "spread_average": 0.02,
            "volume_24h": 1000000.0,
            "price_change_24h": 2.5,
        }

    def _collect_adaptation_metrics(self) -> Dict[str, float]:
        """適応メトリクス収集"""
        return {
            "model_accuracy": 0.89,
            "drift_score": 0.12,
            "adaptation_frequency": 15.5,
            "learning_rate_current": 0.001,
            "memory_usage_mb": 512.5,
            "checkpoint_count": 25,
        }

    def _collect_custom_metrics(self) -> None:
        """カスタムメトリクス収集"""
        for name, collector_func in self.custom_collectors.items():
            try:
                value = collector_func()
                if isinstance(value, (int, float)):
                    self._store_metric(
                        name, float(value), MetricType.PERFORMANCE, datetime.now()
                    )
            except Exception as e:
                logger.error(f"Custom metric collection error for {name}: {e}")

    def _store_metric(
        self, name: str, value: float, metric_type: MetricType, timestamp: datetime
    ) -> None:
        """メトリクス保存"""
        metric_value = MetricValue(
            name=name, value=value, timestamp=timestamp, metric_type=metric_type
        )
        self.metrics_buffer[name].append(metric_value)

    def _cleanup_old_metrics(self) -> None:
        """古いメトリクス削除"""
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_period_days)

        for metric_name, buffer in self.metrics_buffer.items():
            # 古いメトリクスを削除
            while buffer and buffer[0].timestamp < cutoff_time:
                buffer.popleft()

    def get_metric_history(
        self, metric_name: str, hours: int = 24
    ) -> List[MetricValue]:
        """メトリクス履歴取得"""
        if metric_name not in self.metrics_buffer:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.metrics_buffer[metric_name] if m.timestamp >= cutoff_time
        ]

    def get_latest_metrics(self) -> Dict[str, MetricValue]:
        """最新メトリクス取得"""
        latest = {}
        for metric_name, buffer in self.metrics_buffer.items():
            if buffer:
                latest[metric_name] = buffer[-1]
        return latest

    def add_custom_collector(self, name: str, collector_func: Callable) -> None:
        """カスタム収集器追加"""
        self.custom_collectors[name] = collector_func


class AlertManager:
    """アラートマネージャー"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldowns: Dict[str, datetime] = {}
        self.notification_handlers: Dict[str, Callable] = {}

    def check_alerts(self, metrics: Dict[str, MetricValue]) -> List[Alert]:
        """アラートチェック"""
        new_alerts = []

        for condition in self.config.alert_conditions:
            alert_key = f"{condition.metric_name}_{condition.operator}"

            # クールダウンチェック
            if alert_key in self.cooldowns:
                if datetime.now() < self.cooldowns[alert_key]:
                    continue
                else:
                    del self.cooldowns[alert_key]

            # メトリクス値取得
            if condition.metric_name not in metrics:
                continue

            metric_value = metrics[condition.metric_name]
            triggered = self._evaluate_condition(condition, metric_value.value)

            if triggered:
                # アラート作成
                alert = Alert(
                    id=f"{alert_key}_{int(time.time())}",
                    condition=condition,
                    current_value=metric_value.value,
                    threshold=condition.threshold,
                    level=condition.alert_level,
                    status=AlertStatus.ACTIVE,
                    triggered_at=datetime.now(),
                    resolved_at=None,
                    acknowledged_at=None,
                    description=self._generate_alert_message(
                        condition, metric_value.value
                    ),
                    context={
                        "metric_value": metric_value.value,
                        "threshold": condition.threshold,
                    },
                )

                self.active_alerts[alert.id] = alert
                new_alerts.append(alert)

                # クールダウン設定
                self.cooldowns[alert_key] = datetime.now() + timedelta(
                    seconds=self.config.alert_cooldown_seconds
                )

                # 通知送信
                self._send_notifications(alert)

        return new_alerts

    def _evaluate_condition(self, condition: AlertCondition, value: float) -> bool:
        """条件評価"""
        if condition.operator == "gt":
            return value > condition.threshold
        elif condition.operator == "lt":
            return value < condition.threshold
        elif condition.operator == "eq":
            return abs(value - condition.threshold) < 1e-6
        elif condition.operator == "ne":
            return abs(value - condition.threshold) >= 1e-6
        elif condition.operator == "gte":
            return value >= condition.threshold
        elif condition.operator == "lte":
            return value <= condition.threshold
        else:
            return False

    def _generate_alert_message(self, condition: AlertCondition, value: float) -> str:
        """アラートメッセージ生成"""
        return f"Alert: {condition.metric_name} {condition.operator} {condition.threshold} (current: {value:.4f})"

    def _send_notifications(self, alert: Alert) -> None:
        """通知送信"""
        for channel in self.config.notification_channels:
            if channel in self.notification_handlers:
                try:
                    self.notification_handlers[channel](alert)
                except Exception as e:
                    logger.error(f"Notification error for channel {channel}: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """アラート承認"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged_at = datetime.now()
            self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """アラート履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history if alert.triggered_at >= cutoff_time
        ]

    def add_notification_handler(self, channel: str, handler: Callable) -> None:
        """通知ハンドラー追加"""
        self.notification_handlers[channel] = handler


class DashboardGenerator:
    """ダッシュボード生成器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config

    def generate_dashboard_data(
        self, metrics_collector: MetricsCollector, alert_manager: AlertManager
    ) -> DashboardData:
        """ダッシュボードデータ生成"""
        # 最新メトリクス取得
        latest_metrics = metrics_collector.get_latest_metrics()

        # 時系列データ取得
        time_series_data = {}
        for metric_name in self.config.dashboard_config.metrics_to_display:
            history = metrics_collector.get_metric_history(metric_name, hours=24)
            if history:
                time_series_data[metric_name] = TimeSeriesData(
                    metric_name=metric_name,
                    timestamps=[m.timestamp for m in history],
                    values=[m.value for m in history],
                )

        # アラートサマリー
        active_alerts = alert_manager.get_active_alerts()
        alert_summary = {
            "total_active": len(active_alerts),
            "by_level": defaultdict(int),
            "recent_alerts": [],
        }

        for alert in active_alerts:
            alert_summary["by_level"][alert.level.value] += 1

        # 最近のアラート（最新5件）
        recent_alerts = alert_manager.get_alert_history(hours=1)[-5:]
        alert_summary["recent_alerts"] = [
            {
                "id": alert.id,
                "level": alert.level.value,
                "description": alert.description,
                "timestamp": alert.triggered_at.isoformat(),
            }
            for alert in recent_alerts
        ]

        # パフォーマンスサマリー
        performance_summary = self._generate_performance_summary(latest_metrics)

        return DashboardData(
            timestamp=datetime.now(),
            latest_metrics=latest_metrics,
            time_series=time_series_data,
            alert_summary=alert_summary,
            performance_summary=performance_summary,
            refresh_interval_seconds=self.config.dashboard_config.refresh_interval_seconds,
        )

    def _generate_performance_summary(
        self, latest_metrics: Dict[str, MetricValue]
    ) -> Dict[str, Any]:
        """パフォーマンスサマリー生成"""
        summary = {}

        # 主要メトリクスのステータス判定
        if "win_rate" in latest_metrics:
            win_rate = latest_metrics["win_rate"].value
            summary["win_rate_status"] = "good" if win_rate > 0.5 else "poor"

        if "sharpe_ratio" in latest_metrics:
            sharpe = latest_metrics["sharpe_ratio"].value
            summary["sharpe_status"] = (
                "excellent" if sharpe > 2.0 else "good" if sharpe > 1.0 else "poor"
            )

        if "max_drawdown" in latest_metrics:
            drawdown = latest_metrics["max_drawdown"].value
            summary["drawdown_status"] = (
                "good"
                if drawdown > -0.1
                else "warning"
                if drawdown > -0.2
                else "critical"
            )

        return summary


class ReportGenerator:
    """レポート生成器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config

    def generate_report(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        period_days: int = 7,
    ) -> ReportData:
        """レポート生成"""
        # 期間内のメトリクス取得
        all_metrics = {}
        for metric_name in metrics_collector.metrics_buffer.keys():
            history = metrics_collector.get_metric_history(
                metric_name, hours=period_days * 24
            )
            if history:
                all_metrics[metric_name] = history

        # 統計計算
        statistics = self._calculate_statistics(all_metrics)

        # トレンド分析
        trends = self._analyze_trends(all_metrics)

        # アラート分析
        alert_analysis = self._analyze_alerts(alert_manager, period_days)

        # パフォーマンス分析
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
        """統計計算"""
        statistics = {}

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
        """トレンド分析"""
        trends = {}

        for metric_name, values in metrics.items():
            if len(values) < 2:
                trends[metric_name] = "insufficient_data"
                continue

            # 単純な線形トレンド
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
        """アラート分析"""
        alerts = alert_manager.get_alert_history(hours=period_days * 24)

        return {
            "total_alerts": len(alerts),
            "by_level": defaultdict(int, {level.value: 0 for level in AlertLevel}),
            "by_metric": defaultdict(int),
            "most_frequent_alerts": [],
        }

    def _analyze_performance(
        self, metrics: Dict[str, List[MetricValue]]
    ) -> Dict[str, Any]:
        """パフォーマンス分析"""
        # 主要パフォーマンスメトリクスの分析
        analysis = {}

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
        """推奨事項生成"""
        recommendations = []

        # トレンドベースの推奨
        if trends.get("win_rate") == "decreasing":
            recommendations.append(
                "Consider reviewing trading strategy - win rate is declining"
            )

        if trends.get("max_drawdown") == "increasing":
            recommendations.append(
                "Implement additional risk controls - drawdown is increasing"
            )

        # アラートベースの推奨
        if alert_analysis.get("total_alerts", 0) > 10:
            recommendations.append(
                "High alert frequency detected - review system configuration"
            )

        return recommendations


class PerformanceMonitor:
    """パフォーマンス監視システム"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        self.dashboard_generator = DashboardGenerator(config)
        self.report_generator = ReportGenerator(config)

        # デフォルトのアラート条件設定
        self._setup_default_alerts()

        # デフォルトの通知ハンドラー設定
        self._setup_default_notifications()

    def start_monitoring(self) -> None:
        """監視開始"""
        self.metrics_collector.start_collection()
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """監視停止"""
        self.metrics_collector.stop_collection()
        logger.info("Performance monitoring stopped")

    def _setup_default_alerts(self) -> None:
        """デフォルトアラート設定"""
        default_conditions = [
            AlertCondition(
                metric_name="win_rate",
                operator="lt",
                threshold=0.4,
                duration_seconds=60,
                cooldown_seconds=300,
                alert_level=AlertLevel.WARNING,
                description="Win rate below acceptable threshold",
            ),
            AlertCondition(
                metric_name="max_drawdown",
                operator="lt",
                threshold=-0.25,
                duration_seconds=60,
                cooldown_seconds=300,
                alert_level=AlertLevel.CRITICAL,
                description="Maximum drawdown exceeded limit",
            ),
            AlertCondition(
                metric_name="cpu_usage_percent",
                operator="gt",
                threshold=90.0,
                duration_seconds=60,
                cooldown_seconds=300,
                alert_level=AlertLevel.WARNING,
                description="High CPU usage detected",
            ),
            AlertCondition(
                metric_name="memory_usage_percent",
                operator="gt",
                threshold=85.0,
                duration_seconds=60,
                cooldown_seconds=300,
                alert_level=AlertLevel.CRITICAL,
                description="High memory usage detected",
            ),
        ]

        self.config.alert_conditions.extend(default_conditions)

    def _setup_default_notifications(self) -> None:
        """デフォルト通知設定"""
        # ログ通知
        self.alert_manager.add_notification_handler("log", self._log_notification)

        # コンソール通知（開発用）
        self.alert_manager.add_notification_handler(
            "console", self._console_notification
        )

    def _log_notification(self, alert: Alert) -> None:
        """ログ通知"""
        logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.description}")

    def _console_notification(self, alert: Alert) -> None:
        """コンソール通知"""
        print(f"🚨 ALERT: {alert.description}")

    def check_and_alert(self) -> List[Alert]:
        """チェックとアラート"""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        return self.alert_manager.check_alerts(latest_metrics)

    def get_dashboard_data(self) -> DashboardData:
        """ダッシュボードデータ取得"""
        return self.dashboard_generator.generate_dashboard_data(
            self.metrics_collector, self.alert_manager
        )

    def generate_report(self, period_days: int = 7) -> ReportData:
        """レポート生成"""
        return self.report_generator.generate_report(
            self.metrics_collector, self.alert_manager, period_days
        )

    def add_custom_metric_collector(self, name: str, collector_func: Callable) -> None:
        """カスタムメトリクス収集器追加"""
        self.metrics_collector.add_custom_collector(name, collector_func)

    def add_alert_condition(self, condition: AlertCondition) -> None:
        """アラート条件追加"""
        self.config.alert_conditions.append(condition)

    def add_notification_channel(self, channel: str, handler: Callable) -> None:
        """通知チャンネル追加"""
        self.alert_manager.add_notification_handler(channel, handler)

    def get_metrics_history(
        self, metric_name: str, hours: int = 24
    ) -> List[MetricValue]:
        """メトリクス履歴取得"""
        return self.metrics_collector.get_metric_history(metric_name, hours)

    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""
        return self.alert_manager.get_active_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """アラート承認"""
        return self.alert_manager.acknowledge_alert(alert_id)

    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        return self.alert_manager.resolve_alert(alert_id)
