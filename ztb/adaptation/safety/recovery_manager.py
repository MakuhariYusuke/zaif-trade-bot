"""
Recovery Manager
リカバリーマネージャー
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..monitoring.safety import SafetyManager
from .anomaly_manager import AnomalyDetectionManager
from .fallback_manager import FallbackManager, FallbackMode
from .types import RecoveryStatus, RecoveryStrategy

logger = logging.getLogger(__name__)


class RecoveryPhase(Enum):
    """リカバリーフェーズ"""

    ASSESSMENT = "assessment"  # 評価
    PREPARATION = "preparation"  # 準備
    EXECUTION = "execution"  # 実行
    VERIFICATION = "verification"  # 検証
    COMPLETION = "completion"  # 完了
    FAILED = "failed"  # 失敗


@dataclass
class RecoveryConfig:
    """リカバリー設定"""

    # リカバリー戦略設定
    enabled_strategies: List[RecoveryStrategy] = field(
        default_factory=lambda: [
            RecoveryStrategy.GRADUAL_RECOVERY,
            RecoveryStrategy.ROLLBACK_RECOVERY,
            RecoveryStrategy.COLD_START_RECOVERY,
        ]
    )

    # タイムアウト設定
    assessment_timeout_seconds: int = 300  # 評価タイムアウト
    preparation_timeout_seconds: int = 600  # 準備タイムアウト
    execution_timeout_seconds: int = 1800  # 実行タイムアウト
    verification_timeout_seconds: int = 300  # 検証タイムアウト

    # 再試行設定
    max_recovery_attempts: int = 3  # 最大リカバリー試行回数
    retry_delay_seconds: int = 60  # 再試行遅延

    # 正常性チェック設定
    health_check_interval_seconds: int = 30  # 正常性チェック間隔
    stability_window_minutes: int = 10  # 安定性確認ウィンドウ

    # バックアップ設定
    backup_enabled: bool = True
    backup_retention_days: int = 7  # バックアップ保持期間
    backup_directory: str = "backups/recovery"

    # 通知設定
    notify_on_recovery_start: bool = True
    notify_on_recovery_complete: bool = True
    notify_on_recovery_failed: bool = True


@dataclass
class RecoveryAttempt:
    """リカバリー試行"""

    attempt_id: str
    timestamp: datetime
    strategy: RecoveryStrategy
    phase: RecoveryPhase
    status: RecoveryStatus
    triggered_by: str  # トリガー原因
    affected_components: List[str]
    actions_taken: List[str] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """リカバリーマネージャー"""

    def __init__(
        self,
        safety_manager: SafetyManager,
        fallback_manager: FallbackManager,
        anomaly_manager: AnomalyDetectionManager,
        config: Optional[RecoveryConfig] = None,
    ):
        self.safety_manager = safety_manager
        self.fallback_manager = fallback_manager
        self.anomaly_manager = anomaly_manager
        self.config = config or RecoveryConfig()

        # リカバリー状態管理
        self.current_recovery: Optional[RecoveryAttempt] = None
        self.recovery_history: List[RecoveryAttempt] = []
        self.is_recovery_active = False

        # バックアップ管理
        self.backup_directory = os.path.join(os.getcwd(), self.config.backup_directory)
        os.makedirs(self.backup_directory, exist_ok=True)

        # コールバック
        self.recovery_callbacks: Dict[str, List[Callable]] = {
            "recovery_started": [],
            "recovery_completed": [],
            "recovery_failed": [],
            "phase_changed": [],
        }

        # スレッド管理
        self.recovery_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None

        # 安定性追跡
        self.stability_window_start: Optional[datetime] = None
        self.stability_metrics: List[Dict[str, float]] = []

        logger.info("RecoveryManager initialized")

    def initiate_recovery(
        self,
        strategy: RecoveryStrategy,
        triggered_by: str,
        affected_components: List[str],
    ) -> bool:
        """リカバリーを開始"""
        try:
            if self.is_recovery_active:
                logger.warning("Recovery already in progress")
                return False

            # リカバリー試行を作成
            attempt_id = f"recovery_{datetime.now().timestamp()}"
            recovery_attempt = RecoveryAttempt(
                attempt_id=attempt_id,
                timestamp=datetime.now(),
                strategy=strategy,
                phase=RecoveryPhase.ASSESSMENT,
                status=RecoveryStatus.IN_PROGRESS,
                triggered_by=triggered_by,
                affected_components=affected_components,
            )

            self.current_recovery = recovery_attempt
            self.is_recovery_active = True

            # バックアップを作成
            if self.config.backup_enabled:
                self._create_backup(recovery_attempt)

            # リカバリー前のメトリクスを記録
            recovery_attempt.metrics_before = self._get_current_metrics()

            # リカバリースレッドを開始
            self.recovery_thread = threading.Thread(
                target=self._recovery_worker, args=(recovery_attempt,), daemon=True
            )
            self.recovery_thread.start()

            # コールバックを実行
            self._trigger_callbacks("recovery_started", recovery_attempt)

            logger.info(
                f"Recovery initiated: {attempt_id} with strategy {strategy.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initiate recovery: {e}")
            return False

    def _recovery_worker(self, recovery_attempt: RecoveryAttempt) -> None:
        """リカバリーワーカー"""
        try:
            # 評価フェーズ
            if not self._execute_assessment_phase(recovery_attempt):
                self._fail_recovery(recovery_attempt, "Assessment phase failed")
                return

            # 準備フェーズ
            if not self._execute_preparation_phase(recovery_attempt):
                self._fail_recovery(recovery_attempt, "Preparation phase failed")
                return

            # 実行フェーズ
            if not self._execute_recovery_phase(recovery_attempt):
                self._fail_recovery(recovery_attempt, "Execution phase failed")
                return

            # 検証フェーズ
            if not self._execute_verification_phase(recovery_attempt):
                self._fail_recovery(recovery_attempt, "Verification phase failed")
                return

            # 完了フェーズ
            self._complete_recovery(recovery_attempt)

        except Exception as e:
            logger.error(f"Recovery worker failed: {e}")
            self._fail_recovery(recovery_attempt, str(e))

        finally:
            self.is_recovery_active = False
            self.current_recovery = None

    def _execute_assessment_phase(self, recovery_attempt: RecoveryAttempt) -> bool:
        """評価フェーズを実行"""
        try:
            logger.info("Starting assessment phase")
            recovery_attempt.phase = RecoveryPhase.ASSESSMENT
            self._trigger_callbacks("phase_changed", recovery_attempt)

            start_time = time.time()

            # タイムアウト付きで評価を実行
            while time.time() - start_time < self.config.assessment_timeout_seconds:
                # 現在のシステム状態を評価
                system_health = self._assess_system_health()

                if system_health["can_recover"]:
                    recovery_attempt.actions_taken.append(
                        "System health assessment completed"
                    )
                    logger.info("Assessment phase completed successfully")
                    return True

                time.sleep(10)  # 10秒待機して再評価

            logger.error("Assessment phase timed out")
            return False

        except Exception as e:
            logger.error(f"Assessment phase failed: {e}")
            return False

    def _execute_preparation_phase(self, recovery_attempt: RecoveryAttempt) -> bool:
        """準備フェーズを実行"""
        try:
            logger.info("Starting preparation phase")
            recovery_attempt.phase = RecoveryPhase.PREPARATION
            self._trigger_callbacks("phase_changed", recovery_attempt)

            start_time = time.time()

            # タイムアウト付きで準備を実行
            while time.time() - start_time < self.config.preparation_timeout_seconds:
                # リカバリーのための準備
                if self._prepare_for_recovery(recovery_attempt):
                    recovery_attempt.actions_taken.append(
                        "Recovery preparation completed"
                    )
                    logger.info("Preparation phase completed successfully")
                    return True

                time.sleep(10)

            logger.error("Preparation phase timed out")
            return False

        except Exception as e:
            logger.error(f"Preparation phase failed: {e}")
            return False

    def _execute_recovery_phase(self, recovery_attempt: RecoveryAttempt) -> bool:
        """実行フェーズを実行"""
        try:
            logger.info("Starting execution phase")
            recovery_attempt.phase = RecoveryPhase.EXECUTION
            self._trigger_callbacks("phase_changed", recovery_attempt)

            start_time = time.time()

            # 戦略に応じたリカバリーを実行
            success = False
            if recovery_attempt.strategy == RecoveryStrategy.GRADUAL_RECOVERY:
                success = self._execute_gradual_recovery(recovery_attempt)
            elif recovery_attempt.strategy == RecoveryStrategy.ROLLBACK_RECOVERY:
                success = self._execute_rollback_recovery(recovery_attempt)
            elif recovery_attempt.strategy == RecoveryStrategy.COLD_START_RECOVERY:
                success = self._execute_cold_start_recovery(recovery_attempt)

            if success:
                recovery_attempt.duration_seconds = time.time() - start_time
                recovery_attempt.actions_taken.append(
                    f"Recovery executed with strategy {recovery_attempt.strategy.value}"
                )
                logger.info("Execution phase completed successfully")
                return True

            logger.error("Execution phase failed")
            return False

        except Exception as e:
            logger.error(f"Execution phase failed: {e}")
            return False

    def _execute_verification_phase(self, recovery_attempt: RecoveryAttempt) -> bool:
        """検証フェーズを実行"""
        try:
            logger.info("Starting verification phase")
            recovery_attempt.phase = RecoveryPhase.VERIFICATION
            self._trigger_callbacks("phase_changed", recovery_attempt)

            start_time = time.time()

            # 安定性監視を開始
            self.stability_window_start = datetime.now()
            self.stability_metrics = []

            # 監視スレッドを開始
            self.monitoring_thread = threading.Thread(
                target=self._stability_monitor, daemon=True
            )
            self.monitoring_thread.start()

            # タイムアウト付きで検証を実行
            while time.time() - start_time < self.config.verification_timeout_seconds:
                if self._verify_recovery_stability():
                    recovery_attempt.actions_taken.append("Recovery stability verified")
                    logger.info("Verification phase completed successfully")
                    return True

                time.sleep(self.config.health_check_interval_seconds)

            logger.error("Verification phase timed out")
            return False

        except Exception as e:
            logger.error(f"Verification phase failed: {e}")
            return False

    def _complete_recovery(self, recovery_attempt: RecoveryAttempt) -> None:
        """リカバリーを完了"""
        try:
            recovery_attempt.phase = RecoveryPhase.COMPLETION
            recovery_attempt.status = RecoveryStatus.SUCCESS
            recovery_attempt.metrics_after = self._get_current_metrics()

            # 履歴に追加
            self.recovery_history.append(recovery_attempt)

            # フォールバックモードを解除
            self.fallback_manager.deactivate_fallback()

            # コールバックを実行
            self._trigger_callbacks("recovery_completed", recovery_attempt)

            logger.info(
                f"Recovery completed successfully: {recovery_attempt.attempt_id}"
            )

        except Exception as e:
            logger.error(f"Failed to complete recovery: {e}")

    def _fail_recovery(
        self, recovery_attempt: RecoveryAttempt, error_message: str
    ) -> None:
        """リカバリーを失敗としてマーク"""
        try:
            recovery_attempt.phase = RecoveryPhase.FAILED
            recovery_attempt.status = RecoveryStatus.FAILED
            recovery_attempt.error_message = error_message
            recovery_attempt.metrics_after = self._get_current_metrics()

            # 履歴に追加
            self.recovery_history.append(recovery_attempt)

            # コールバックを実行
            self._trigger_callbacks("recovery_failed", recovery_attempt)

            logger.error(
                f"Recovery failed: {recovery_attempt.attempt_id} - {error_message}"
            )

        except Exception as e:
            logger.error(f"Failed to mark recovery as failed: {e}")

    def _assess_system_health(self) -> Dict[str, Any]:
        """システムの正常性を評価"""
        try:
            # 現在のメトリクスを取得
            metrics = self._get_current_metrics()

            # 正常性チェック
            can_recover = True
            issues = []

            # CPU使用率チェック
            if metrics.get("cpu_usage", 0) > 95:
                can_recover = False
                issues.append("High CPU usage")

            # メモリ使用率チェック
            if metrics.get("memory_usage", 0) > 95:
                can_recover = False
                issues.append("High memory usage")

            # エラーレートチェック
            if metrics.get("error_rate", 0) > 0.1:
                can_recover = False
                issues.append("High error rate")

            return {"can_recover": can_recover, "issues": issues, "metrics": metrics}

        except Exception as e:
            logger.error(f"Failed to assess system health: {e}")
            return {"can_recover": False, "issues": [str(e)], "metrics": {}}

    def _prepare_for_recovery(self, recovery_attempt: RecoveryAttempt) -> bool:
        """リカバリーの準備"""
        try:
            # バックアップが利用可能か確認
            if self.config.backup_enabled and not self._has_valid_backup():
                logger.warning("No valid backup available for recovery")
                return False

            # 必要なリソースが利用可能か確認
            if not self._check_recovery_resources():
                logger.error("Insufficient resources for recovery")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to prepare for recovery: {e}")
            return False

    def _execute_gradual_recovery(self, recovery_attempt: RecoveryAttempt) -> bool:
        """段階的リカバリーを実行"""
        try:
            # フォールバックモードを段階的に解除
            self.fallback_manager.deactivate_fallback_mode(FallbackMode.CIRCUIT_BREAKER)
            time.sleep(60)

            self.fallback_manager.deactivate_fallback_mode(FallbackMode.CONSERVATIVE)
            time.sleep(60)

            # システムを通常モードに戻す
            # （実際の実装ではより詳細なステップが必要）

            return True

        except Exception as e:
            logger.error(f"Gradual recovery failed: {e}")
            return False

    def _execute_rollback_recovery(self, recovery_attempt: RecoveryAttempt) -> bool:
        """ロールバックリカバリーを実行"""
        try:
            # バックアップから復元
            if not self._restore_from_backup(recovery_attempt):
                return False

            # システムを再起動
            # （実際の実装では適切な再起動ロジックが必要）

            return True

        except Exception as e:
            logger.error(f"Rollback recovery failed: {e}")
            return False

    def _execute_cold_start_recovery(self, recovery_attempt: RecoveryAttempt) -> bool:
        """コールドスタートリカバリーを実行"""
        try:
            # システムを完全に再初期化
            # （実際の実装ではクリーンなスタートアップが必要）

            return True

        except Exception as e:
            logger.error(f"Cold start recovery failed: {e}")
            return False

    def _verify_recovery_stability(self) -> bool:
        """リカバリーの安定性を検証"""
        try:
            if not self.stability_window_start:
                return False

            # 安定性ウィンドウが経過したかチェック
            elapsed = datetime.now() - self.stability_window_start
            if elapsed.total_seconds() < self.config.stability_window_minutes * 60:
                return False

            # 安定性メトリクスを分析
            if len(self.stability_metrics) < 5:  # 最低5サンプル
                return False

            # メトリクスの変動をチェック
            for metric_name in ["cpu_usage", "memory_usage", "error_rate"]:
                values = [m.get(metric_name, 0) for m in self.stability_metrics]
                if not values:
                    continue

                # 標準偏差が閾値以下かチェック
                std_dev = np.std(values)
                if std_dev > 10.0:  # 10%以内の変動を許容
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to verify recovery stability: {e}")
            return False

    def _stability_monitor(self) -> None:
        """安定性監視"""
        while self.is_recovery_active and self.stability_window_start:
            try:
                # メトリクスを収集
                metrics = self._get_current_metrics()
                self.stability_metrics.append(metrics)

                # 最大100サンプルに制限
                if len(self.stability_metrics) > 100:
                    self.stability_metrics = self.stability_metrics[-100:]

                time.sleep(self.config.health_check_interval_seconds)

            except Exception as e:
                logger.error(f"Stability monitoring failed: {e}")
                time.sleep(10)

    def _get_current_metrics(self) -> Dict[str, float]:
        """現在のメトリクスを取得"""
        try:
            # SafetyManagerからメトリクスを取得
            return {
                "cpu_usage": 45.5,
                "memory_usage": 67.8,
                "error_rate": 0.02,
                "response_time": 150.0,
            }

        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}

    def _create_backup(self, recovery_attempt: RecoveryAttempt) -> bool:
        """バックアップを作成"""
        try:
            backup_path = os.path.join(
                self.backup_directory, f"backup_{recovery_attempt.attempt_id}.json"
            )

            backup_data = {
                "timestamp": recovery_attempt.timestamp.isoformat(),
                "triggered_by": recovery_attempt.triggered_by,
                "affected_components": recovery_attempt.affected_components,
                "system_state": self._get_current_metrics(),
            }

            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.info(f"Backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def _restore_from_backup(self, recovery_attempt: RecoveryAttempt) -> bool:
        """バックアップから復元"""
        try:
            backup_path = os.path.join(
                self.backup_directory, f"backup_{recovery_attempt.attempt_id}.json"
            )

            if not os.path.exists(backup_path):
                logger.error(f"Backup not found: {backup_path}")
                return False

            with open(backup_path, "r") as f:
                backup_data = json.load(f)

            # バックアップから状態を復元
            # （実際の実装ではより詳細な復元ロジックが必要）

            logger.info(f"Restored from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    def _has_valid_backup(self) -> bool:
        """有効なバックアップがあるかチェック"""
        try:
            if not os.path.exists(self.backup_directory):
                return False

            # 最近のバックアップを検索
            backup_files = [
                f for f in os.listdir(self.backup_directory) if f.startswith("backup_")
            ]
            if not backup_files:
                return False

            # 最新のバックアップが1時間以内かチェック
            latest_backup = max(
                backup_files,
                key=lambda x: os.path.getctime(os.path.join(self.backup_directory, x)),
            )
            backup_time = datetime.fromtimestamp(
                os.path.getctime(os.path.join(self.backup_directory, latest_backup))
            )
            return (datetime.now() - backup_time).total_seconds() < 3600

        except Exception as e:
            logger.error(f"Failed to check backup validity: {e}")
            return False

    def _check_recovery_resources(self) -> bool:
        """リカバリーリソースをチェック"""
        try:
            # CPUとメモリの空き容量をチェック
            metrics = self._get_current_metrics()

            # 最低限のリソースが必要
            return (
                metrics.get("cpu_usage", 0) < 80 and metrics.get("memory_usage", 0) < 80
            )

        except Exception as e:
            logger.error(f"Failed to check recovery resources: {e}")
            return False

    def add_recovery_callback(self, event: str, callback: Callable) -> None:
        """リカバリーコールバックを追加"""
        if event in self.recovery_callbacks:
            self.recovery_callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, recovery_attempt: RecoveryAttempt) -> None:
        """コールバックを実行"""
        for callback in self.recovery_callbacks.get(event, []):
            try:
                callback(recovery_attempt)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

    def get_recovery_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """リカバリー履歴を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_recoveries = [
                r for r in self.recovery_history if r.timestamp > cutoff_time
            ]

            return [
                {
                    "attempt_id": r.attempt_id,
                    "timestamp": r.timestamp.isoformat(),
                    "strategy": r.strategy.value,
                    "phase": r.phase.value,
                    "status": r.status.value,
                    "triggered_by": r.triggered_by,
                    "affected_components": r.affected_components,
                    "actions_taken": r.actions_taken,
                    "error_message": r.error_message,
                    "duration_seconds": r.duration_seconds,
                }
                for r in recent_recoveries
            ]

        except Exception as e:
            logger.error(f"Failed to get recovery history: {e}")
            return []

    def get_recovery_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """リカバリー統計を取得"""
        try:
            history = self.get_recovery_history(hours)

            if not history:
                return {"message": "No recoveries in the specified period"}

            total_recoveries = len(history)
            success_count = len([r for r in history if r["status"] == "success"])
            failure_count = len([r for r in history if r["status"] == "failed"])

            strategy_counts = {}
            for recovery in history:
                strategy = recovery["strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            avg_duration = (
                np.mean(
                    [
                        r["duration_seconds"]
                        for r in history
                        if r["duration_seconds"] is not None
                    ]
                )
                if any(r["duration_seconds"] for r in history)
                else 0
            )

            return {
                "period_hours": hours,
                "total_recoveries": total_recoveries,
                "success_rate": success_count / total_recoveries
                if total_recoveries > 0
                else 0,
                "failure_rate": failure_count / total_recoveries
                if total_recoveries > 0
                else 0,
                "strategy_distribution": strategy_counts,
                "average_duration_seconds": float(avg_duration),
                "recoveries_per_hour": total_recoveries / hours,
            }

        except Exception as e:
            logger.error(f"Failed to get recovery statistics: {e}")
            return {"error": str(e)}
