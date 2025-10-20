"""
Integrated Operations Management System
オンライン学習、監視、安全機構、スケーラビリティを統合した運用システム
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..monitoring.monitor import PerformanceMonitor
from ..monitoring.safety import SafetyManager
from ..monitoring.scalability import AutoScaler, LoadBalancer

# from ..config import SACConfig  # 循環インポートを避けるためコメントアウト
from .config import IntegratedOperationsConfig
from .types import (
    AlertSummary,
    IntegrationStatus,
    OperationalMetrics,
    RecoveryAction,
    SystemHealth,
)

logger = logging.getLogger(__name__)


class IntegratedOperationsManager:
    """統合運用マネージャー"""

    def __init__(
        self,
        config: Any,
        operations_config: Optional[IntegratedOperationsConfig] = None,
    ):
        self.config = config
        self.operations_config = operations_config or IntegratedOperationsConfig()
        self.is_running = False
        self.threads: List[threading.Thread] = []
        self.start_time = datetime.now()

        # 各コンポーネントの初期化
        self.monitor = PerformanceMonitor(config.monitoring)
        self.safety_manager = SafetyManager(config.safety)
        self.load_balancer = LoadBalancer(config.scalability)
        self.auto_scaler = AutoScaler(config.scalability, self.load_balancer)
        self.online_learning = None  # モデルが提供されたら初期化

        # 統合ステータス
        self.system_health = SystemHealth.HEALTHY
        self.last_health_check = datetime.now()
        self.integration_status = IntegrationStatus(
            monitoring_active=False,
            safety_active=False,
            scalability_active=False,
            online_learning_active=False,
            last_integration_check=datetime.now(),
        )

        # 運用メトリクス
        self.operational_metrics = OperationalMetrics(
            uptime_seconds=0,
            total_requests=0,
            error_rate=0.0,
            average_response_time=0.0,
            resource_utilization={},
            last_updated=datetime.now(),
        )

        logger.info("Integrated Operations Manager initialized")

    def start_all_systems(self) -> bool:
        """全システム起動"""
        try:
            logger.info("Starting all integrated systems...")

            # 設定に基づいて各システムを起動
            if self.operations_config.monitoring_enabled:
                self.monitor.start_monitoring()
                self.integration_status.monitoring_active = True

            if self.operations_config.safety_enabled:
                # SafetyManagerの起動メソッドを確認して呼び出し
                if hasattr(self.safety_manager, "start_safety_monitoring"):
                    self.safety_manager.start_safety_monitoring()
                self.integration_status.safety_active = True

            if self.operations_config.scalability_enabled:
                # AutoScalerの起動メソッドを確認（起動不要の場合がある）
                self.integration_status.scalability_active = True

            # 統合監視スレッド起動
            if self.operations_config.integrated_operations_enabled:
                integration_thread = threading.Thread(
                    target=self._integration_monitor_worker, daemon=True
                )
                integration_thread.start()
                self.threads.append(integration_thread)

                # ヘルスチェックスレッド起動
                health_thread = threading.Thread(
                    target=self._health_check_worker, daemon=True
                )
                health_thread.start()
                self.threads.append(health_thread)

            self.is_running = True
            logger.info("All integrated systems started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start integrated systems: {e}")
            self.stop_all_systems()
            return False

    def stop_all_systems(self) -> None:
        """全システム停止"""
        logger.info("Stopping all integrated systems...")

        self.is_running = False

        # 設定に基づいて各システム停止
        if self.operations_config.monitoring_enabled:
            try:
                self.monitor.stop_monitoring()
                self.integration_status.monitoring_active = False
            except Exception as e:
                logger.error(f"Error stopping monitoring: {e}")

        if self.operations_config.safety_enabled:
            try:
                if hasattr(self.safety_manager, "stop_safety_monitoring"):
                    self.safety_manager.stop_safety_monitoring()
                self.integration_status.safety_active = False
            except Exception as e:
                logger.error(f"Error stopping safety: {e}")

        if self.operations_config.scalability_enabled:
            try:
                # AutoScalerの停止メソッドを確認（停止不要の場合がある）
                self.integration_status.scalability_active = False
            except Exception as e:
                logger.error(f"Error stopping scalability: {e}")

        # スレッド停止待ち
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        self.threads.clear()
        logger.info("All integrated systems stopped")

    def initialize_online_learning(self, model: Any) -> bool:
        """オンライン学習初期化"""
        try:
            from ..online_learning.config import OnlineLearningConfig
            from ..online_learning.pipeline import OnlineLearningPipeline

            online_config = OnlineLearningConfig()
            self.online_learning = OnlineLearningPipeline(online_config, model)
            self.online_learning.start_learning()
            self.integration_status.online_learning_active = True

            logger.info("Online learning initialized and started")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize online learning: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """システム全体のステータス取得"""
        return {
            "system_health": self.system_health.value,
            "integration_status": {
                "monitoring": self.integration_status.monitoring_active,
                "safety": self.integration_status.safety_active,
                "scalability": self.integration_status.scalability_active,
                "online_learning": self.integration_status.online_learning_active,
                "last_check": self.integration_status.last_integration_check.isoformat(),
            },
            "operational_metrics": {
                "uptime_seconds": self.operational_metrics.uptime_seconds,
                "total_requests": self.operational_metrics.total_requests,
                "error_rate": self.operational_metrics.error_rate,
                "avg_response_time": self.operational_metrics.average_response_time,
                "last_updated": self.operational_metrics.last_updated.isoformat(),
            },
            "component_status": {
                "monitor": self.monitor.get_status()
                if hasattr(self.monitor, "get_status")
                else {"status": "active"},
                "safety": self.safety_manager.get_safety_status()
                if hasattr(self.safety_manager, "get_safety_status")
                else {"status": "active"},
                "scaler": self.auto_scaler.get_scaling_history()
                if hasattr(self.auto_scaler, "get_scaling_history")
                else {"status": "active"},
                "online_learning": self.online_learning.get_status()
                if self.online_learning and hasattr(self.online_learning, "get_status")
                else None,
            },
            "last_health_check": self.last_health_check.isoformat(),
        }

    def get_alerts_summary(self) -> AlertSummary:
        """アラート概要取得"""
        alerts = self.monitor.get_active_alerts()
        critical_count = sum(1 for alert in alerts if alert.level.value >= 4)
        warning_count = sum(1 for alert in alerts if alert.level.value == 3)
        info_count = sum(1 for alert in alerts if alert.level.value <= 2)

        return AlertSummary(
            total_alerts=len(alerts),
            critical_alerts=critical_count,
            warning_alerts=warning_count,
            info_alerts=info_count,
            top_alerts=alerts[:5],  # 最新5件
            last_updated=datetime.now(),
        )

    def trigger_emergency_shutdown(self, reason: str) -> bool:
        """緊急シャットダウン"""
        try:
            logger.critical(f"Emergency shutdown triggered: {reason}")

            # 安全停止を実行
            self.safety_manager.execute_emergency_shutdown(reason)

            # 全システム停止
            self.stop_all_systems()

            # 回復アクション記録
            recovery_action = RecoveryAction(
                action_type="emergency_shutdown",
                reason=reason,
                timestamp=datetime.now(),
                system_state=self.get_system_status(),
                recommended_actions=[
                    "Investigate the cause of emergency shutdown",
                    "Check system logs for error details",
                    "Verify data integrity",
                    "Perform manual system restart after investigation",
                ],
            )

            logger.critical(f"Emergency shutdown completed: {recovery_action}")
            return True

        except Exception as e:
            logger.error(f"Failed to execute emergency shutdown: {e}")
            return False

    def _integration_monitor_worker(self) -> None:
        """統合監視ワーカー"""
        while self.is_running:
            try:
                # 各コンポーネントの統合チェック
                self._check_component_integration()

                # 運用メトリクス更新
                self._update_operational_metrics()

                time.sleep(self.operations_config.component_sync_interval_seconds)

            except Exception as e:
                logger.error(f"Error in integration monitor: {e}")
                time.sleep(10)

    def _health_check_worker(self) -> None:
        """ヘルスチェックワーカー"""
        while self.is_running:
            try:
                # システムヘルスチェック
                health_status = self._perform_health_check()

                # ヘルスステータス更新
                self.system_health = health_status
                self.last_health_check = datetime.now()

                # 異常検知時の処理
                if health_status != SystemHealth.HEALTHY:
                    self._handle_health_issue(health_status)

                time.sleep(self.operations_config.health_check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in health check: {e}")
                time.sleep(30)

    def _check_component_integration(self) -> None:
        """コンポーネント統合チェック"""
        try:
            # 監視システムチェック
            if self.integration_status.monitoring_active:
                monitor_status = self.monitor.get_status()
                if not monitor_status.get("active", False):
                    logger.warning("Monitoring system integration check failed")
                    self.integration_status.monitoring_active = False

            # 安全システムチェック
            if self.integration_status.safety_active:
                safety_status = self.safety_manager.get_status()
                if not safety_status.get("active", False):
                    logger.warning("Safety system integration check failed")
                    self.integration_status.safety_active = False

            # スケーラビリティシステムチェック
            if self.integration_status.scalability_active:
                scaler_status = self.auto_scaler.get_status()
                if not scaler_status.get("active", False):
                    logger.warning("Scalability system integration check failed")
                    self.integration_status.scalability_active = False

            # オンライン学習チェック
            if self.integration_status.online_learning_active and self.online_learning:
                learning_status = self.online_learning.get_status()
                if not learning_status.get("active", False):
                    logger.warning("Online learning integration check failed")
                    self.integration_status.online_learning_active = False

            self.integration_status.last_integration_check = datetime.now()

        except Exception as e:
            logger.error(f"Error in component integration check: {e}")

    def _update_operational_metrics(self) -> None:
        """運用メトリクス更新"""
        try:
            # 稼働時間更新
            if hasattr(self, "start_time"):
                self.operational_metrics.uptime_seconds = (
                    datetime.now() - self.start_time
                ).total_seconds()
            else:
                self.start_time = datetime.now()

            # リクエスト数とエラー率（モニターから取得）
            monitor_metrics = self.monitor.get_recent_metrics()
            self.operational_metrics.total_requests = monitor_metrics.get(
                "total_requests", 0
            )
            self.operational_metrics.error_rate = monitor_metrics.get("error_rate", 0.0)
            self.operational_metrics.average_response_time = monitor_metrics.get(
                "avg_response_time", 0.0
            )

            # リソース使用率
            self.operational_metrics.resource_utilization = {
                "cpu_percent": monitor_metrics.get("cpu_percent", 0.0),
                "memory_percent": monitor_metrics.get("memory_percent", 0.0),
                "disk_usage": monitor_metrics.get("disk_usage", 0.0),
            }

            self.operational_metrics.last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Error updating operational metrics: {e}")

    def _perform_health_check(self) -> SystemHealth:
        """ヘルスチェック実行"""
        try:
            issues = []

            # 各コンポーネントのヘルスチェック
            if not self.integration_status.monitoring_active:
                issues.append("Monitoring system inactive")

            if not self.integration_status.safety_active:
                issues.append("Safety system inactive")

            if not self.integration_status.scalability_active:
                issues.append("Scalability system inactive")

            # アラートチェック
            alerts = self.monitor.get_active_alerts()
            critical_alerts = [a for a in alerts if a.level.value >= 4]
            if critical_alerts:
                issues.append(f"{len(critical_alerts)} critical alerts")

            # リソース使用率チェック
            cpu_usage = self.operational_metrics.resource_utilization.get(
                "cpu_percent", 0.0
            )
            memory_usage = self.operational_metrics.resource_utilization.get(
                "memory_percent", 0.0
            )

            if cpu_usage > 95.0:
                issues.append("CPU usage critically high")
            elif cpu_usage > 80.0:
                issues.append("CPU usage high")

            if memory_usage > 95.0:
                issues.append("Memory usage critically high")
            elif memory_usage > 80.0:
                issues.append("Memory usage high")

            # ヘルスステータス決定
            if (
                any("critically" in issue.lower() for issue in issues)
                or len(critical_alerts) > 0
            ):
                return SystemHealth.CRITICAL
            elif issues:
                return SystemHealth.WARNING
            else:
                return SystemHealth.HEALTHY

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return SystemHealth.UNKNOWN

    def _handle_health_issue(self, health_status: SystemHealth) -> None:
        """ヘルス問題処理"""
        try:
            if health_status == SystemHealth.WARNING:
                logger.warning(f"System health warning: {health_status.value}")
                # 警告時はログ出力のみ

            elif health_status == SystemHealth.CRITICAL:
                logger.critical(f"System health critical: {health_status.value}")
                # クリティカル時は安全停止を検討
                self.safety_manager.handle_critical_health_issue()

            elif health_status == SystemHealth.UNKNOWN:
                logger.error(f"System health unknown: {health_status.value}")
                # 不明時は詳細チェック

        except Exception as e:
            logger.error(f"Error handling health issue: {e}")
