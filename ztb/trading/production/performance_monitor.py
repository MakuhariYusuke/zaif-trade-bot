"""
V433 Phase 5: Gradual Rollout Layer - Performance Monitor

運用中のパフォーマンスを継続監視し、問題を早期検知する。
"""

import asyncio
import json
import logging
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional


class MonitorFrequency(Enum):
    """監視頻度"""

    REALTIME = "realtime"  # リアルタイム
    HIGH = "high"  # 高頻度（1分）
    MEDIUM = "medium"  # 中頻度（5分）
    LOW = "low"  # 低頻度（15分）


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceThreshold:
    """パフォーマンス閾値"""

    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison: str  # '>', '<', '>=', '<='


@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    system_id: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """パフォーマンススナップショット"""

    snapshot_id: str
    timestamp: datetime
    system_id: str
    metrics: Dict[str, float]
    alerts: List[PerformanceAlert] = field(default_factory=list)
    overall_health: str = "healthy"


@dataclass
class HealthCheck:
    """ヘルスチェック"""

    check_id: str
    timestamp: datetime
    system_id: str
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    error_message: Optional[str] = None


class PerformanceMonitor:
    """
    パフォーマンスモニター

    運用中のシステムパフォーマンスを継続監視し、
    問題を早期検知してアラートを発行する。
    """

    def __init__(
        self,
        monitor_frequency: MonitorFrequency = MonitorFrequency.MEDIUM,
        alert_cooldown_minutes: int = 5,
        max_alerts_history: int = 1000,
    ):
        """
        初期化

        Args:
            monitor_frequency: 監視頻度
            alert_cooldown_minutes: アラートクールダウン時間（分）
            max_alerts_history: アラート履歴最大数
        """
        self.monitor_frequency = monitor_frequency
        self.alert_cooldown_minutes = alert_cooldown_minutes
        self.max_alerts_history = max_alerts_history

        # パフォーマンス閾値
        self.performance_thresholds: Dict[str, PerformanceThreshold] = {}
        self._setup_default_thresholds()

        # 監視対象システム
        self.monitored_systems: set[str] = set()

        # パフォーマンスデータ
        self.performance_history: Dict[str, List[PerformanceSnapshot]] = {}
        self.alerts_history: List[PerformanceAlert] = []
        self.health_checks: List[HealthCheck] = []

        # アクティブアラート
        self.active_alerts: Dict[str, PerformanceAlert] = {}

        # アラート抑制
        self.alert_cooldowns: Dict[str, datetime] = {}

        # モニタリング制御
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_monitoring = datetime.now()

        # コールバック
        self.alert_callbacks: List[Callable[[PerformanceAlert], Awaitable[None]]] = []
        self.snapshot_callbacks: List[
            Callable[[PerformanceSnapshot], Awaitable[None]]
        ] = []
        self.health_callbacks: List[Callable[[HealthCheck], Awaitable[None]]] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Performance Monitor initialized")

    def _setup_default_thresholds(self) -> None:
        """デフォルト閾値設定"""
        default_thresholds = [
            PerformanceThreshold("cpu_usage", 70.0, 85.0, 95.0, ">"),
            PerformanceThreshold("memory_usage", 75.0, 90.0, 95.0, ">"),
            PerformanceThreshold("response_time_ms", 1000.0, 5000.0, 10000.0, ">"),
            PerformanceThreshold("error_rate", 0.05, 0.10, 0.20, ">"),
            PerformanceThreshold("win_rate", 0.45, 0.35, 0.25, "<"),
            PerformanceThreshold("sharpe_ratio", 0.2, 0.1, 0.0, "<"),
            PerformanceThreshold("max_drawdown", 0.10, 0.15, 0.20, ">"),
            PerformanceThreshold("execution_latency_ms", 500.0, 2000.0, 5000.0, ">"),
        ]

        for threshold in default_thresholds:
            self.performance_thresholds[threshold.metric_name] = threshold

    def add_system_to_monitor(self, system_id: str) -> None:
        """
        監視対象システム追加

        Args:
            system_id: システムID
        """
        self.monitored_systems.add(system_id)
        self.performance_history[system_id] = []

        self.logger.info(f"System added to monitoring: {system_id}")

    def remove_system_from_monitor(self, system_id: str) -> None:
        """
        監視対象システム削除

        Args:
            system_id: システムID
        """
        if system_id in self.monitored_systems:
            self.monitored_systems.remove(system_id)

        self.logger.info(f"System removed from monitoring: {system_id}")

    def add_performance_threshold(self, threshold: PerformanceThreshold) -> None:
        """
        パフォーマンス閾値追加

        Args:
            threshold: パフォーマンス閾値
        """
        self.performance_thresholds[threshold.metric_name] = threshold
        self.logger.info(f"Performance threshold added: {threshold.metric_name}")

    def update_performance_metrics(
        self, system_id: str, metrics: Dict[str, float]
    ) -> None:
        """
        パフォーマンス指標更新

        Args:
            system_id: システムID
            metrics: パフォーマンス指標
        """
        if system_id not in self.monitored_systems:
            return

        # スナップショット作成
        snapshot = PerformanceSnapshot(
            snapshot_id=f"SNAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            system_id=system_id,
            metrics=metrics.copy(),
        )

        # アラートチェック
        alerts = self._check_thresholds(system_id, metrics)
        snapshot.alerts = alerts

        # 全体ヘルス評価
        snapshot.overall_health = self._assess_overall_health(alerts)

        # 履歴保存
        self.performance_history[system_id].append(snapshot)

        # 履歴サイズ制限（最新1000件）
        if len(self.performance_history[system_id]) > 1000:
            self.performance_history[system_id] = self.performance_history[system_id][
                -1000:
            ]

        # アラート処理
        for alert in alerts:
            self._process_alert(alert)

        # コールバック実行
        for callback in self.snapshot_callbacks:
            try:
                asyncio.create_task(callback(snapshot))
            except Exception as e:
                self.logger.error(f"Snapshot callback error: {e}")

    def _check_thresholds(
        self, system_id: str, metrics: Dict[str, float]
    ) -> List[PerformanceAlert]:
        """
        閾値チェック

        Args:
            system_id: システムID
            metrics: パフォーマンス指標

        Returns:
            List[PerformanceAlert]: アラートリスト
        """
        alerts = []

        for metric_name, value in metrics.items():
            if metric_name not in self.performance_thresholds:
                continue

            threshold = self.performance_thresholds[metric_name]

            # クールダウンチェック
            alert_key = f"{system_id}:{metric_name}"
            if alert_key in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[alert_key]:
                    continue

            # 閾値比較
            severity = self._evaluate_threshold(threshold, value)
            if severity:
                alert = PerformanceAlert(
                    alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    timestamp=datetime.now(),
                    severity=severity,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=self._get_threshold_value(threshold, severity),
                    message=self._generate_alert_message(threshold, value, severity),
                    system_id=system_id,
                )

                alerts.append(alert)

                # クールダウン設定
                self.alert_cooldowns[alert_key] = datetime.now() + timedelta(
                    minutes=self.alert_cooldown_minutes
                )

        return alerts

    def _evaluate_threshold(
        self, threshold: PerformanceThreshold, value: float
    ) -> Optional[AlertSeverity]:
        """
        閾値評価

        Args:
            threshold: パフォーマンス閾値
            value: 現在の値

        Returns:
            Optional[AlertSeverity]: アラート重要度
        """
        if threshold.comparison == ">":
            if value >= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= threshold.error_threshold:
                return AlertSeverity.ERROR
            elif value >= threshold.warning_threshold:
                return AlertSeverity.WARNING
        elif threshold.comparison == "<":
            if value <= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= threshold.error_threshold:
                return AlertSeverity.ERROR
            elif value <= threshold.warning_threshold:
                return AlertSeverity.WARNING
        elif threshold.comparison == ">=":
            if value >= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= threshold.error_threshold:
                return AlertSeverity.ERROR
            elif value >= threshold.warning_threshold:
                return AlertSeverity.WARNING
        elif threshold.comparison == "<=":
            if value <= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= threshold.error_threshold:
                return AlertSeverity.ERROR
            elif value <= threshold.warning_threshold:
                return AlertSeverity.WARNING

        return None

    def _get_threshold_value(
        self, threshold: PerformanceThreshold, severity: AlertSeverity
    ) -> float:
        """
        閾値取得

        Args:
            threshold: パフォーマンス閾値
            severity: アラート重要度

        Returns:
            float: 閾値
        """
        if severity == AlertSeverity.CRITICAL:
            return threshold.critical_threshold
        elif severity == AlertSeverity.ERROR:
            return threshold.error_threshold
        else:
            return threshold.warning_threshold

    def _generate_alert_message(
        self, threshold: PerformanceThreshold, value: float, severity: AlertSeverity
    ) -> str:
        """
        アラートメッセージ生成

        Args:
            threshold: パフォーマンス閾値
            value: 現在の値
            severity: アラート重要度

        Returns:
            str: アラートメッセージ
        """
        threshold_value = self._get_threshold_value(threshold, severity)
        comparison = "above" if threshold.comparison in [">", ">="] else "below"

        return f"{threshold.metric_name} is {comparison} threshold: {value:.2f} {threshold.comparison} {threshold_value:.2f}"

    def _assess_overall_health(self, alerts: List[PerformanceAlert]) -> str:
        """
        全体ヘルス評価

        Args:
            alerts: アラートリスト

        Returns:
            str: 全体ヘルス状態
        """
        if any(alert.severity == AlertSeverity.CRITICAL for alert in alerts):
            return "critical"
        elif any(alert.severity == AlertSeverity.ERROR for alert in alerts):
            return "unhealthy"
        elif any(alert.severity == AlertSeverity.WARNING for alert in alerts):
            return "degraded"
        else:
            return "healthy"

    def _process_alert(self, alert: PerformanceAlert) -> None:
        """
        アラート処理

        Args:
            alert: パフォーマンスアラート
        """
        # アラート履歴保存
        self.alerts_history.append(alert)

        # 履歴サイズ制限
        if len(self.alerts_history) > self.max_alerts_history:
            self.alerts_history = self.alerts_history[-self.max_alerts_history :]

        # アクティブアラート管理
        alert_key = f"{alert.system_id}:{alert.metric_name}"
        self.active_alerts[alert_key] = alert

        # コールバック実行
        for callback in self.alert_callbacks:
            try:
                asyncio.create_task(callback(alert))
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

        self.logger.warning(
            f"Performance alert: {alert.message} (severity: {alert.severity.value})"
        )

    def resolve_alert(self, alert_id: str) -> bool:
        """
        アラート解決

        Args:
            alert_id: アラートID

        Returns:
            bool: 解決成功フラグ
        """
        for alert in self.alerts_history:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()

                # アクティブアラートから削除
                alert_key = f"{alert.system_id}:{alert.metric_name}"
                if alert_key in self.active_alerts:
                    del self.active_alerts[alert_key]

                self.logger.info(f"Alert resolved: {alert_id}")
                return True

        return False

    def perform_health_check(
        self,
        system_id: str,
        component: str,
        response_time_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """
        ヘルスチェック実行

        Args:
            system_id: システムID
            component: コンポーネント名
            response_time_ms: 応答時間
            error_message: エラーメッセージ
        """
        status = "healthy"
        if error_message or response_time_ms > 5000:  # 5秒以上
            status = "unhealthy"
        elif response_time_ms > 2000:  # 2秒以上
            status = "degraded"

        health_check = HealthCheck(
            check_id=f"HC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            system_id=system_id,
            component=component,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
        )

        self.health_checks.append(health_check)

        # 履歴サイズ制限（最新500件）
        if len(self.health_checks) > 500:
            self.health_checks = self.health_checks[-500:]

        # コールバック実行
        for callback in self.health_callbacks:
            try:
                asyncio.create_task(callback(health_check))
            except Exception as e:
                self.logger.error(f"Health callback error: {e}")

    def get_performance_summary(
        self, system_id: str, hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        パフォーマンス要約取得

        Args:
            system_id: システムID
            hours: 集計期間（時間）

        Returns:
            Optional[Dict[str, Any]]: パフォーマンス要約
        """
        if system_id not in self.performance_history:
            return None

        period_start = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [
            s
            for s in self.performance_history[system_id]
            if s.timestamp >= period_start
        ]

        if not recent_snapshots:
            return None

        # 指標別統計
        metrics_summary = {}
        all_metrics = set()
        for snapshot in recent_snapshots:
            all_metrics.update(snapshot.metrics.keys())

        for metric_name in all_metrics:
            values = [
                s.metrics.get(metric_name, 0)
                for s in recent_snapshots
                if metric_name in s.metrics
            ]
            if values:
                metrics_summary[metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # アラート統計
        recent_alerts = [
            alert for snapshot in recent_snapshots for alert in snapshot.alerts
        ]

        alert_stats = {
            "total": len(recent_alerts),
            "by_severity": {
                "info": len(
                    [a for a in recent_alerts if a.severity == AlertSeverity.INFO]
                ),
                "warning": len(
                    [a for a in recent_alerts if a.severity == AlertSeverity.WARNING]
                ),
                "error": len(
                    [a for a in recent_alerts if a.severity == AlertSeverity.ERROR]
                ),
                "critical": len(
                    [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
                ),
            },
        }

        # ヘルス統計
        recent_health = [
            hc
            for hc in self.health_checks
            if hc.system_id == system_id and hc.timestamp >= period_start
        ]

        health_stats = {
            "total_checks": len(recent_health),
            "healthy": len([hc for hc in recent_health if hc.status == "healthy"]),
            "degraded": len([hc for hc in recent_health if hc.status == "degraded"]),
            "unhealthy": len([hc for hc in recent_health if hc.status == "unhealthy"]),
        }

        return {
            "system_id": system_id,
            "period_hours": hours,
            "total_snapshots": len(recent_snapshots),
            "metrics_summary": metrics_summary,
            "alert_stats": alert_stats,
            "health_stats": health_stats,
            "overall_health": recent_snapshots[-1].overall_health
            if recent_snapshots
            else "unknown",
        }

    def get_active_alerts(
        self, system_id: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """
        アクティブアラート取得

        Args:
            system_id: システムID（指定なしの場合は全システム）

        Returns:
            List[PerformanceAlert]: アクティブアラート
        """
        alerts = list(self.active_alerts.values())

        if system_id:
            alerts = [a for a in alerts if a.system_id == system_id]

        return alerts

    def get_alert_history(
        self, system_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[PerformanceAlert]:
        """
        アラート履歴取得

        Args:
            system_id: システムID（指定なしの場合は全システム）
            limit: 取得件数制限

        Returns:
            List[PerformanceAlert]: アラート履歴
        """
        alerts = self.alerts_history

        if system_id:
            alerts = [a for a in alerts if a.system_id == system_id]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        interval_seconds = self._get_monitoring_interval()

        while self.monitoring_active:
            try:
                # 定期的なヘルスチェック
                for system_id in self.monitored_systems:
                    self._perform_system_health_check(system_id)

                # アラートクリーンアップ
                self._cleanup_resolved_alerts()

                time.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def _get_monitoring_interval(self) -> int:
        """
        モニタリング間隔取得

        Returns:
            int: 間隔（秒）
        """
        intervals = {
            MonitorFrequency.REALTIME: 10,  # 10秒
            MonitorFrequency.HIGH: 60,  # 1分
            MonitorFrequency.MEDIUM: 300,  # 5分
            MonitorFrequency.LOW: 900,  # 15分
        }

        return intervals.get(self.monitor_frequency, 300)

    def _perform_system_health_check(self, system_id: str) -> None:
        """
        システムヘルスチェック

        Args:
            system_id: システムID
        """
        # 簡易的なヘルスチェック
        # 実際の実装では各システムのヘルスエンドポイントを呼び出す
        start_time = time.time()

        try:
            # シミュレーション：ランダムでレスポンスタイムを生成
            import random

            response_time = random.uniform(50, 2000)  # 50ms - 2秒
            error_message = None

            # 稀にエラーを発生
            if random.random() < 0.05:  # 5%の確率
                error_message = "Simulated connection error"

        except Exception as e:
            response_time = 9999
            error_message = str(e)

        response_time = (time.time() - start_time) * 1000

        self.perform_health_check(
            system_id, "system_core", response_time, error_message
        )

    def _cleanup_resolved_alerts(self) -> None:
        """解決済みアラートクリーンアップ"""
        # 24時間以上経過した解決済みアラートを削除
        cutoff_time = datetime.now() - timedelta(hours=24)

        self.alerts_history = [
            alert
            for alert in self.alerts_history
            if not (
                alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
            )
        ]

    def add_alert_callback(
        self, callback: Callable[[PerformanceAlert], Awaitable[None]]
    ) -> None:
        """
        アラートコールバック追加

        Args:
            callback: コールバック関数
        """
        self.alert_callbacks.append(callback)

    def add_snapshot_callback(
        self, callback: Callable[[PerformanceSnapshot], Awaitable[None]]
    ) -> None:
        """
        スナップショットコールバック追加

        Args:
            callback: コールバック関数
        """
        self.snapshot_callbacks.append(callback)

    def add_health_callback(
        self, callback: Callable[[HealthCheck], Awaitable[None]]
    ) -> None:
        """
        ヘルスチェックコールバック追加

        Args:
            callback: コールバック関数
        """
        self.health_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "monitor_frequency": self.monitor_frequency.value,
            "alert_cooldown_minutes": self.alert_cooldown_minutes,
            "max_alerts_history": self.max_alerts_history,
            "monitored_systems": list(self.monitored_systems),
            "performance_thresholds": [
                {
                    "metric_name": t.metric_name,
                    "warning_threshold": t.warning_threshold,
                    "error_threshold": t.error_threshold,
                    "critical_threshold": t.critical_threshold,
                    "comparison": t.comparison,
                }
                for t in self.performance_thresholds.values()
            ],
            "alerts_history": [
                {
                    "alert_id": a.alert_id,
                    "timestamp": a.timestamp.isoformat(),
                    "severity": a.severity.value,
                    "metric_name": a.metric_name,
                    "current_value": a.current_value,
                    "threshold_value": a.threshold_value,
                    "message": a.message,
                    "system_id": a.system_id,
                    "resolved": a.resolved,
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                }
                for a in self.alerts_history[-200:]  # 最新200件
            ],
            "last_monitoring": self.last_monitoring.isoformat(),
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Monitor state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.monitor_frequency = MonitorFrequency(state["monitor_frequency"])
            self.alert_cooldown_minutes = state["alert_cooldown_minutes"]
            self.max_alerts_history = state["max_alerts_history"]
            self.monitored_systems = set(state["monitored_systems"])
            self.last_monitoring = datetime.fromisoformat(state["last_monitoring"])

            # 閾値復元
            self.performance_thresholds = {}
            for t_data in state.get("performance_thresholds", []):
                threshold = PerformanceThreshold(
                    metric_name=t_data["metric_name"],
                    warning_threshold=t_data["warning_threshold"],
                    error_threshold=t_data["error_threshold"],
                    critical_threshold=t_data["critical_threshold"],
                    comparison=t_data["comparison"],
                )
                self.performance_thresholds[threshold.metric_name] = threshold

            # アラート履歴復元
            self.alerts_history = []
            for a_data in state.get("alerts_history", []):
                alert = PerformanceAlert(
                    alert_id=a_data["alert_id"],
                    timestamp=datetime.fromisoformat(a_data["timestamp"]),
                    severity=AlertSeverity(a_data["severity"]),
                    metric_name=a_data["metric_name"],
                    current_value=a_data["current_value"],
                    threshold_value=a_data["threshold_value"],
                    message=a_data["message"],
                    system_id=a_data["system_id"],
                    resolved=a_data["resolved"],
                    resolved_at=datetime.fromisoformat(a_data["resolved_at"])
                    if a_data["resolved_at"]
                    else None,
                )
                self.alerts_history.append(alert)

            # パフォーマンス履歴初期化（再構築が必要）
            self.performance_history = {
                system_id: [] for system_id in self.monitored_systems
            }

            self.logger.info(f"Monitor state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load monitor state: {e}")
            return False
