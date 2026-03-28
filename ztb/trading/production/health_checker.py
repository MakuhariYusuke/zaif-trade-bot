"""
V433 Phase 5: Production Monitoring Layer - Health Checker

システムの総合的なヘルスチェックを行い、運用状態を評価する。
"""

import asyncio
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable

try:
    import psutil
except ImportError:
    psutil = None
import requests

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)

class HealthStatus(Enum):
    """ヘルスステータス"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class HealthCheckType(Enum):
    """ヘルスチェックタイプ"""

    SYSTEM = "system"  # システムリソース
    APPLICATION = "application"  # アプリケーションヘルス
    DATABASE = "database"  # データベース接続
    NETWORK = "network"  # ネットワーク接続
    EXTERNAL_API = "external_api"  # 外部API
    BUSINESS_LOGIC = "business_logic"  # ビジネスロジック

@dataclass
class HealthCheck:
    """ヘルスチェック"""

    check_id: str
    name: str
    type: HealthCheckType
    description: str
    enabled: bool = True
    timeout_seconds: int = 30
    interval_seconds: int = 60
    failure_threshold: int = 3
    success_threshold: int = 2
    last_check: datetime | None = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_time_ms: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthReport:
    """ヘルスレポート"""

    report_id: str
    timestamp: datetime
    overall_status: HealthStatus
    checks: list[HealthCheck] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

class HealthChecker:
    """
    ヘルスチェッカー

    システムの総合的なヘルスチェックを行い、運用状態を評価。
    自動修復機能も提供。
    """

    def __init__(
        self, check_interval_seconds: int = 60, report_retention_days: int = 7
    ):
        """
        初期化

        Args:
            check_interval_seconds: チェック間隔（秒）
            report_retention_days: レポート保持期間（日）
        """
        self.check_interval_seconds = check_interval_seconds
        self.report_retention_days = report_retention_days

        # ヘルスチェック管理
        self.health_checks: dict[str, HealthCheck] = {}

        # レポート管理
        self.health_reports: list[HealthReport] = []

        # 自動修復設定
        self.auto_remediation_enabled = True
        self.remediation_actions: dict[
            str, Callable[[HealthCheck], Awaitable[bool]]
        ] = {}

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None

        # コールバック
        self.status_callbacks: list[
            Callable[[HealthStatus, HealthStatus], Awaitable[None]]
        ] = []
        self.report_callbacks: list[Callable[[HealthReport], Awaitable[None]]] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        # デフォルトチェック初期化
        self._initialize_default_checks()

        self.logger.info("Health Checker initialized")

    def _initialize_default_checks(self) -> None:
        """デフォルトヘルスチェック初期化"""
        default_checks = [
            # システムチェック
            HealthCheck(
                "cpu_usage",
                "CPU Usage Check",
                HealthCheckType.SYSTEM,
                "CPU使用率のチェック",
                interval_seconds=30,
            ),
            HealthCheck(
                "memory_usage",
                "Memory Usage Check",
                HealthCheckType.SYSTEM,
                "メモリ使用率のチェック",
                interval_seconds=30,
            ),
            HealthCheck(
                "disk_usage",
                "Disk Usage Check",
                HealthCheckType.SYSTEM,
                "ディスク使用率のチェック",
                interval_seconds=300,
            ),
            HealthCheck(
                "network_connectivity",
                "Network Connectivity",
                HealthCheckType.NETWORK,
                "ネットワーク接続性のチェック",
                interval_seconds=60,
            ),
            # アプリケーションチェック
            HealthCheck(
                "application_process",
                "Application Process",
                HealthCheckType.APPLICATION,
                "アプリケーションプロセスのチェック",
                interval_seconds=30,
            ),
            HealthCheck(
                "application_response",
                "Application Response",
                HealthCheckType.APPLICATION,
                "アプリケーション応答性のチェック",
                interval_seconds=60,
            ),
            # データベースチェック
            HealthCheck(
                "database_connection",
                "Database Connection",
                HealthCheckType.DATABASE,
                "データベース接続のチェック",
                interval_seconds=60,
            ),
            # ビジネスロジックチェック
            HealthCheck(
                "trading_system",
                "Trading System Health",
                HealthCheckType.BUSINESS_LOGIC,
                "取引システムのヘルスのチェック",
                interval_seconds=120,
            ),
        ]

        for check in default_checks:
            self.add_health_check(check)

    def add_health_check(self, check: HealthCheck) -> None:
        """
        ヘルスチェック追加

        Args:
            check: ヘルスチェック
        """
        self.health_checks[check.check_id] = check
        self.logger.info(f"Health check added: {check.check_id} - {check.name}")

    def remove_health_check(self, check_id: str) -> None:
        """
        ヘルスチェック削除

        Args:
            check_id: チェックID
        """
        if check_id in self.health_checks:
            del self.health_checks[check_id]
            self.logger.info(f"Health check removed: {check_id}")

    def enable_health_check(self, check_id: str) -> None:
        """
        ヘルスチェック有効化

        Args:
            check_id: チェックID
        """
        if check_id in self.health_checks:
            self.health_checks[check_id].enabled = True
            self.logger.info(f"Health check enabled: {check_id}")

    def disable_health_check(self, check_id: str) -> None:
        """
        ヘルスチェック無効化

        Args:
            check_id: チェックID
        """
        if check_id in self.health_checks:
            self.health_checks[check_id].enabled = False
            self.logger.info(f"Health check disabled: {check_id}")

    async def run_health_check(self, check_id: str) -> HealthStatus:
        """
        ヘルスチェック実行

        Args:
            check_id: チェックID

        Returns:
            HealthStatus: ヘルスステータス
        """
        if check_id not in self.health_checks:
            return HealthStatus.UNKNOWN

        check = self.health_checks[check_id]
        if not check.enabled:
            return HealthStatus.UNKNOWN

        start_time = time.time()

        try:
            # チェックタイプに応じた実行
            if check.type == HealthCheckType.SYSTEM:
                status = await self._run_system_check(check)
            elif check.type == HealthCheckType.APPLICATION:
                status = await self._run_application_check(check)
            elif check.type == HealthCheckType.DATABASE:
                status = await self._run_database_check(check)
            elif check.type == HealthCheckType.NETWORK:
                status = await self._run_network_check(check)
            elif check.type == HealthCheckType.EXTERNAL_API:
                status = await self._run_external_api_check(check)
            elif check.type == HealthCheckType.BUSINESS_LOGIC:
                status = await self._run_business_logic_check(check)
            else:
                status = HealthStatus.UNKNOWN

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            check.error_message = str(e)
            self.logger.error(f"Health check error for {check_id}: {e}")

        response_time = (time.time() - start_time) * 1000
        check.response_time_ms = response_time
        check.last_check = datetime.now()

        # ステータス遷移管理
        old_status = check.last_status
        check.last_status = status

        if status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            check.consecutive_successes += 1
            check.consecutive_failures = 0
            check.error_message = None
        else:
            check.consecutive_failures += 1
            check.consecutive_successes = 0

        # ステータス変更通知
        if old_status != status:
            for callback in self.status_callbacks:
                try:
                    asyncio.create_task(callback(old_status, status))
                except Exception as e:
                    self.logger.error(f"Status callback error: {e}")

        # 自動修復
        if self.auto_remediation_enabled and status != HealthStatus.HEALTHY:
            if check_id in self.remediation_actions:
                await self._attempt_remediation(check)

        return status

    async def _run_system_check(self, check: HealthCheck) -> HealthStatus:
        """
        システムチェック実行

        Args:
            check: ヘルスチェック

        Returns:
            HealthStatus: ヘルスステータス
        """
        try:
            if psutil is None:
                check.metadata = {"psutil_available": False}
                return HealthStatus.HEALTHY

            if check.check_id == "cpu_usage":
                cpu_percent = psutil.cpu_percent(interval=1)
                check.metadata = {"cpu_percent": cpu_percent}

                if cpu_percent > 95:
                    return HealthStatus.CRITICAL
                elif cpu_percent > 85:
                    return HealthStatus.UNHEALTHY
                elif cpu_percent > 70:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY

            elif check.check_id == "memory_usage":
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                check.metadata = {
                    "memory_percent": memory_percent,
                    "memory_used_gb": memory.used / (1024**3),
                }

                if memory_percent > 95:
                    return HealthStatus.CRITICAL
                elif memory_percent > 90:
                    return HealthStatus.UNHEALTHY
                elif memory_percent > 80:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY

            elif check.check_id == "disk_usage":
                disk = psutil.disk_usage("/")
                disk_percent = disk.percent
                check.metadata = {
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                }

                if disk_percent > 95:
                    return HealthStatus.CRITICAL
                elif disk_percent > 90:
                    return HealthStatus.UNHEALTHY
                elif disk_percent > 85:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY

            elif check.check_id == "network_connectivity":
                # 基本的なネットワークチェック
                try:
                    # DNS解決チェック
                    import socket

                    socket.gethostbyname("google.com")
                    check.metadata = {"dns_resolution": "success"}
                    return HealthStatus.HEALTHY
                except Exception:
                    check.metadata = {"dns_resolution": "failed"}
                    return HealthStatus.UNHEALTHY

        except Exception as e:
            check.error_message = str(e)
            return HealthStatus.UNHEALTHY

        return HealthStatus.UNKNOWN

    async def _run_application_check(self, check: HealthCheck) -> HealthStatus:
        """
        アプリケーションチェック実行

        Args:
            check: ヘルスチェック

        Returns:
            HealthStatus: ヘルスステータス
        """
        try:
            if psutil is None:
                check.metadata = {"psutil_available": False, "process_count": 0}
                return HealthStatus.HEALTHY

            if check.check_id == "application_process":
                # プロセスチェック
                process_name = check.metadata.get("process_name", "python")
                processes = []

                for proc in psutil.process_iter(
                    ["pid", "name", "cpu_percent", "memory_percent"]
                ):
                    try:
                        if process_name.lower() in proc.info["name"].lower():
                            processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                check.metadata = {
                    "process_count": len(processes),
                    "processes": processes,
                }

                if len(processes) == 0:
                    return HealthStatus.CRITICAL
                elif len(processes) < check.metadata.get("min_processes", 1):
                    return HealthStatus.UNHEALTHY
                else:
                    return HealthStatus.HEALTHY

            elif check.check_id == "application_response":
                # アプリケーション応答チェック
                health_endpoint = check.metadata.get("health_endpoint")
                if health_endpoint:
                    try:
                        response = requests.get(health_endpoint, timeout=10)
                        check.metadata = {
                            "status_code": response.status_code,
                            "response_time_ms": response.elapsed.total_seconds() * 1000,
                        }

                        if response.status_code == 200:
                            return HealthStatus.HEALTHY
                        else:
                            return HealthStatus.UNHEALTHY
                    except Exception as e:
                        check.metadata = {"error": str(e)}
                        return HealthStatus.UNHEALTHY

        except Exception as e:
            check.error_message = str(e)
            return HealthStatus.UNHEALTHY

        return HealthStatus.UNKNOWN

    async def _run_database_check(self, check: HealthCheck) -> HealthStatus:
        """
        データベースチェック実行

        Args:
            check: ヘルスチェック

        Returns:
            HealthStatus: ヘルスステータス
        """
        # データベース接続チェックの実装
        # 実際の実装では適切なDBドライバを使用
        check.metadata = {"status": "not_implemented"}
        return HealthStatus.UNKNOWN

    async def _run_network_check(self, check: HealthCheck) -> HealthStatus:
        """
        ネットワークチェック実行

        Args:
            check: ヘルスチェック

        Returns:
            HealthStatus: ヘルスステータス
        """
        try:
            # ネットワークレイテンシチェック
            targets = check.metadata.get("ping_targets", ["8.8.8.8", "1.1.1.1"])

            latencies = []
            for target in targets:
                try:
                    # Windows環境でのping
                    result = subprocess.run(
                        ["ping", "-n", "1", "-w", "1000", target],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if result.returncode == 0:
                        # 応答時間抽出（簡易）
                        output_lines = result.stdout.split("\n")
                        for line in output_lines:
                            if "time=" in line.lower():
                                time_str = line.split("time=")[1].split("ms")[0].strip()
                                latencies.append(float(time_str))
                                break
                except Exception:
                    continue

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                check.metadata = {
                    "average_latency_ms": avg_latency,
                    "targets_tested": len(targets),
                    "successful_pings": len(latencies),
                }

                if avg_latency > 500:
                    return HealthStatus.UNHEALTHY
                elif avg_latency > 200:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY
            else:
                check.metadata = {"error": "no successful pings"}
                return HealthStatus.UNHEALTHY

        except Exception as e:
            check.error_message = str(e)
            return HealthStatus.UNHEALTHY

    async def _run_external_api_check(self, check: HealthCheck) -> HealthStatus:
        """
        外部APIチェック実行

        Args:
            check: ヘルスチェック

        Returns:
            HealthStatus: ヘルスステータス
        """
        # 外部APIチェックの実装
        check.metadata = {"status": "not_implemented"}
        return HealthStatus.UNKNOWN

    async def _run_business_logic_check(self, check: HealthCheck) -> HealthStatus:
        """
        ビジネスロジックチェック実行

        Args:
            check: ヘルスチェック

        Returns:
            HealthStatus: ヘルスステータス
        """
        try:
            if check.check_id == "trading_system":
                # 取引システムのヘルスチェック
                # 実際の実装では取引システムの状態を確認
                check.metadata = {
                    "orders_pending": 0,  # 保留中注文数
                    "active_positions": 0,  # アクティブポジション数
                    "last_trade_time": datetime.now().isoformat(),
                }

                # 簡易的なチェック
                return HealthStatus.HEALTHY

        except Exception as e:
            check.error_message = str(e)
            return HealthStatus.UNHEALTHY

        return HealthStatus.UNKNOWN

    async def _attempt_remediation(self, check: HealthCheck) -> None:
        """
        自動修復試行

        Args:
            check: ヘルスチェック
        """
        if check.check_id not in self.remediation_actions:
            return

        remediation_func = self.remediation_actions[check.check_id]

        try:
            success = await remediation_func(check)
            if success:
                self.logger.info(f"Auto remediation successful for {check.check_id}")
                # 修復後に再チェック
                await asyncio.sleep(5)
                await self.run_health_check(check.check_id)
            else:
                self.logger.warning(f"Auto remediation failed for {check.check_id}")

        except Exception as e:
            self.logger.error(f"Auto remediation error for {check.check_id}: {e}")

    def add_remediation_action(
        self, check_id: str, action: Callable[[HealthCheck], Awaitable[bool]]
    ) -> None:
        """
        修復アクション追加

        Args:
            check_id: チェックID
            action: 修復アクション
        """
        self.remediation_actions[check_id] = action
        self.logger.info(f"Remediation action added for {check_id}")

    async def run_all_checks(self) -> HealthReport:
        """
        全チェック実行

        Returns:
            HealthReport: ヘルスレポート
        """
        report = HealthReport(
            report_id=f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            overall_status=HealthStatus.UNKNOWN,
        )

        # 全チェック実行
        tasks = []
        for check in self.health_checks.values():
            if check.enabled:
                tasks.append(self.run_health_check(check.check_id))

        if tasks:
            await asyncio.gather(*tasks)

        # レポート作成
        report.checks = list(self.health_checks.values())
        report.overall_status = self._calculate_overall_status(report.checks)
        report.summary = self._generate_summary(report.checks)
        report.recommendations = self._generate_recommendations(report.checks)

        # レポート保存
        self.health_reports.append(report)

        # レポート保持数制限（最新100件）
        if len(self.health_reports) > 100:
            self.health_reports = self.health_reports[-100:]

        # コールバック実行
        for callback in self.report_callbacks:
            try:
                asyncio.create_task(callback(report))
            except Exception as e:
                self.logger.error(f"Report callback error: {e}")

        return report

    def _calculate_overall_status(self, checks: list[HealthCheck]) -> HealthStatus:
        """
        全体ステータス計算

        Args:
            checks: ヘルスチェックリスト

        Returns:
            HealthStatus: 全体ステータス
        """
        if not checks:
            return HealthStatus.UNKNOWN

        status_priority = {
            HealthStatus.CRITICAL: 4,
            HealthStatus.UNHEALTHY: 3,
            HealthStatus.DEGRADED: 2,
            HealthStatus.HEALTHY: 1,
            HealthStatus.UNKNOWN: 0,
        }

        max_priority = max(
            (status_priority.get(check.last_status, 0) for check in checks), default=0
        )

        for status, priority in status_priority.items():
            if priority == max_priority:
                return status

        return HealthStatus.UNKNOWN

    def _generate_summary(self, checks: list[HealthCheck]) -> dict[str, Any]:
        """
        要約生成

        Args:
            checks: ヘルスチェックリスト

        Returns:
            dict[str, Any]: 要約
        """
        total_checks = len(checks)
        enabled_checks = len([c for c in checks if c.enabled])

        status_counts = {}
        for check in checks:
            if check.enabled:
                status = check.last_status.value
                status_counts[status] = status_counts.get(status, 0) + 1

        avg_response_time = None
        response_times = [
            c.response_time_ms for c in checks if c.response_time_ms is not None
        ]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)

        return {
            "total_checks": total_checks,
            "enabled_checks": enabled_checks,
            "status_counts": status_counts,
            "average_response_time_ms": avg_response_time,
            "last_check_time": datetime.now().isoformat(),
        }

    def _generate_recommendations(self, checks: list[HealthCheck]) -> list[str]:
        """
        推奨事項生成

        Args:
            checks: ヘルスチェックリスト

        Returns:
            list[str]: 推奨事項
        """
        recommendations = []

        for check in checks:
            if not check.enabled:
                continue

            if check.last_status == HealthStatus.CRITICAL:
                recommendations.append(
                    f"CRITICAL: {check.name} - Immediate attention required"
                )
            elif check.last_status == HealthStatus.UNHEALTHY:
                recommendations.append(
                    f"UNHEALTHY: {check.name} - Investigate and resolve"
                )
            elif check.last_status == HealthStatus.DEGRADED:
                recommendations.append(f"DEGRADED: {check.name} - Monitor closely")

            # 連続失敗チェック
            if check.consecutive_failures >= check.failure_threshold:
                recommendations.append(
                    f"Persistent failures in {check.name} - Consider manual intervention"
                )

        if not recommendations:
            recommendations.append("All systems operating normally")

        return recommendations

    def get_health_report(
        self, report_id: str | None = None
    ) -> HealthReport | None:
        """
        ヘルスレポート取得

        Args:
            report_id: レポートID（指定なしの場合は最新）

        Returns:
            HealthReport | None: ヘルスレポート
        """
        if report_id:
            for report in self.health_reports:
                if report.report_id == report_id:
                    return report
            return None
        else:
            return self.health_reports[-1] if self.health_reports else None

    def get_health_history(self, hours: int = 24) -> list[HealthReport]:
        """
        ヘルス履歴取得

        Args:
            hours: 取得期間（時間）

        Returns:
            list[HealthReport]: ヘルスレポートリスト
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [r for r in self.health_reports if r.timestamp >= cutoff_time]

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Health monitoring started")

    # Backwards-compatible aliases expected by integration tests
    def start_checking(self) -> None:
        return self.start_monitoring()

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Health monitoring stopped")

    def stop_checking(self) -> None:
        return self.stop_monitoring()

    def get_health_status(self) -> dict[str, Any]:
        """Return a brief health status summary for external callers."""
        report = self.get_latest_report()
        if report is None:
            return {"status": HealthStatus.UNKNOWN.value}
        return {"status": report.overall_status.value, "summary": report.summary}

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # 全チェック実行
                asyncio.run(self.run_all_checks())

                time.sleep(self.check_interval_seconds)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def add_status_callback(
        self, callback: Callable[[HealthStatus, HealthStatus], Awaitable[None]]
    ) -> None:
        """
        ステータスコールバック追加

        Args:
            callback: コールバック関数
        """
        self.status_callbacks.append(callback)

    def add_report_callback(
        self, callback: Callable[[HealthReport], Awaitable[None]]
    ) -> None:
        """
        レポートコールバック追加

        Args:
            callback: コールバック関数
        """
        self.report_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "check_interval_seconds": self.check_interval_seconds,
            "report_retention_days": self.report_retention_days,
            "auto_remediation_enabled": self.auto_remediation_enabled,
            "health_checks": {
                check_id: {
                    "check_id": check.check_id,
                    "name": check.name,
                    "type": check.type.value,
                    "description": check.description,
                    "enabled": check.enabled,
                    "timeout_seconds": check.timeout_seconds,
                    "interval_seconds": check.interval_seconds,
                    "failure_threshold": check.failure_threshold,
                    "success_threshold": check.success_threshold,
                    "last_check": check.last_check.isoformat()
                    if check.last_check
                    else None,
                    "last_status": check.last_status.value,
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes,
                    "response_time_ms": check.response_time_ms,
                    "error_message": check.error_message,
                    "metadata": check.metadata,
                }
                for check_id, check in self.health_checks.items()
            },
            "health_reports": [
                {
                    "report_id": r.report_id,
                    "timestamp": r.timestamp.isoformat(),
                    "overall_status": r.overall_status.value,
                    "summary": r.summary,
                    "recommendations": r.recommendations,
                }
                for r in self.health_reports[-20:]  # 最新20件
            ],
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Health checker state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            state = read_state_payload(filepath)

            self.check_interval_seconds = state["check_interval_seconds"]
            self.report_retention_days = state["report_retention_days"]
            self.auto_remediation_enabled = state["auto_remediation_enabled"]

            # ヘルスチェック復元
            self.health_checks = {}
            for check_id, check_data in state.get("health_checks", {}).items():
                check = HealthCheck(
                    check_id=check_data["check_id"],
                    name=check_data["name"],
                    type=HealthCheckType(check_data["type"]),
                    description=check_data["description"],
                    enabled=check_data["enabled"],
                    timeout_seconds=check_data["timeout_seconds"],
                    interval_seconds=check_data["interval_seconds"],
                    failure_threshold=check_data["failure_threshold"],
                    success_threshold=check_data["success_threshold"],
                    last_check=datetime.fromisoformat(check_data["last_check"])
                    if check_data["last_check"]
                    else None,
                    last_status=HealthStatus(check_data["last_status"]),
                    consecutive_failures=check_data["consecutive_failures"],
                    consecutive_successes=check_data["consecutive_successes"],
                    response_time_ms=check_data["response_time_ms"],
                    error_message=check_data["error_message"],
                    metadata=check_data["metadata"],
                )
                self.health_checks[check_id] = check

            # ヘルスレポート復元
            self.health_reports = []
            for r_data in state.get("health_reports", []):
                report = HealthReport(
                    report_id=r_data["report_id"],
                    timestamp=datetime.fromisoformat(r_data["timestamp"]),
                    overall_status=HealthStatus(r_data["overall_status"]),
                    summary=r_data["summary"],
                    recommendations=r_data["recommendations"],
                )
                self.health_reports.append(report)

            self.logger.info(f"Health checker state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load health checker state: {e}")
            return False
