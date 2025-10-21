"""
V433 Phase 5: Emergency Control Layer - Recovery System

システム障害からの自動復旧と状態修復を行う。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, Awaitable
from enum import Enum
import json
import os
import threading
import time
import shutil
import subprocess


class RecoveryPhase(Enum):
    """復旧フェーズ"""
    ASSESSMENT = "assessment"      # 評価
    BACKUP = "backup"             # バックアップ
    CLEANUP = "cleanup"           # クリーンアップ
    RESTORE = "restore"           # 復元
    VALIDATION = "validation"     # 検証
    RECOVERY = "recovery"         # 復旧
    MONITORING = "monitoring"     # 監視


class RecoveryStrategy(Enum):
    """復旧戦略"""
    ROLLING_BACK = "rolling_back"        # ロールバック
    FAILOVER = "failover"               # フェイルオーバー
    RESTART = "restart"                 # 再起動
    SCALE_UP = "scale_up"               # スケールアップ
    DATA_RESTORE = "data_restore"       # データ復元
    SERVICE_RESTART = "service_restart" # サービス再起動


class RecoveryStatus(Enum):
    """復旧ステータス"""
    PENDING = "pending"         # 待機中
    IN_PROGRESS = "in_progress" # 実行中
    COMPLETED = "completed"     # 完了
    FAILED = "failed"          # 失敗
    ROLLED_BACK = "rolled_back" # ロールバック済み


@dataclass
class RecoveryCheckpoint:
    """復旧チェックポイント"""
    checkpoint_id: str
    timestamp: datetime
    phase: RecoveryPhase
    status: RecoveryStatus
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class RecoveryPlan:
    """復旧プラン"""
    plan_id: str
    failure_type: str
    strategy: RecoveryStrategy
    estimated_duration_minutes: int
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    phases: List[RecoveryPhase] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class RecoveryExecution:
    """復旧実行"""
    execution_id: str
    plan_id: str
    failure_description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PENDING
    current_phase: Optional[RecoveryPhase] = None
    checkpoints: List[RecoveryCheckpoint] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_triggered: bool = False


class RecoverySystem:
    """
    復旧システム

    システム障害からの自動復旧と状態修復を行う。
    段階的な復旧プロセスとロールバック機能を備える。
    """

    def __init__(self, system_name: str = "V433 Trading System", max_concurrent_recoveries: int = 3):
        """
        初期化

        Args:
            system_name: システム名
            max_concurrent_recoveries: 最大同時復旧数
        """
        self.system_name = system_name
        self.max_concurrent_recoveries = max_concurrent_recoveries

        # 復旧プラン
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self._initialize_default_plans()

        # 実行管理
        self.active_recoveries: Dict[str, RecoveryExecution] = {}
        self.recovery_history: List[RecoveryExecution] = []

        # バックアップ管理
        self.backup_configs: Dict[str, Dict[str, Any]] = {}
        self.backup_schedule: Dict[str, str] = {}  # cron形式

        # ヘルスチェック
        self.health_checks: Dict[str, Callable[[], bool]] = {}

        # コールバック
        self.recovery_callbacks: List[Callable[[RecoveryExecution], Awaitable[None]]] = []
        self.phase_callbacks: List[Callable[[RecoveryCheckpoint], Awaitable[None]]] = []

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Recovery System initialized for {system_name}")

    def _initialize_default_plans(self) -> None:
        """デフォルト復旧プラン初期化"""
        # サービス再起動プラン
        service_restart_plan = RecoveryPlan(
            plan_id="service_restart",
            failure_type="service_crash",
            strategy=RecoveryStrategy.SERVICE_RESTART,
            estimated_duration_minutes=5,
            risk_level="low",
            phases=[
                RecoveryPhase.ASSESSMENT,
                RecoveryPhase.CLEANUP,
                RecoveryPhase.RECOVERY,
                RecoveryPhase.VALIDATION,
                RecoveryPhase.MONITORING
            ],
            preconditions=[
                "Service process is not running",
                "No active transactions",
                "Backup data is available"
            ],
            success_criteria=[
                "Service starts successfully",
                "Health checks pass",
                "No error logs in first 5 minutes"
            ]
        )

        # データ復元プラン
        data_restore_plan = RecoveryPlan(
            plan_id="data_restore",
            failure_type="data_corruption",
            strategy=RecoveryStrategy.DATA_RESTORE,
            estimated_duration_minutes=30,
            risk_level="high",
            phases=[
                RecoveryPhase.ASSESSMENT,
                RecoveryPhase.BACKUP,
                RecoveryPhase.RESTORE,
                RecoveryPhase.VALIDATION,
                RecoveryPhase.RECOVERY,
                RecoveryPhase.MONITORING
            ],
            preconditions=[
                "Valid backup exists",
                "Corruption is confirmed",
                "System is isolated"
            ],
            success_criteria=[
                "Data integrity verified",
                "All services restart successfully",
                "Business logic validation passes"
            ]
        )

        # フェイルオーバープラン
        failover_plan = RecoveryPlan(
            plan_id="failover",
            failure_type="node_failure",
            strategy=RecoveryStrategy.FAILOVER,
            estimated_duration_minutes=10,
            risk_level="medium",
            phases=[
                RecoveryPhase.ASSESSMENT,
                RecoveryPhase.RECOVERY,
                RecoveryPhase.VALIDATION,
                RecoveryPhase.MONITORING
            ],
            preconditions=[
                "Secondary node is healthy",
                "Data synchronization is current",
                "Load balancer is responsive"
            ],
            success_criteria=[
                "Traffic successfully routed to secondary",
                "All services operational",
                "Data consistency maintained"
            ]
        )

        # ロールバックプラン
        rollback_plan = RecoveryPlan(
            plan_id="rollback",
            failure_type="deployment_failure",
            strategy=RecoveryStrategy.ROLLING_BACK,
            estimated_duration_minutes=15,
            risk_level="medium",
            phases=[
                RecoveryPhase.ASSESSMENT,
                RecoveryPhase.BACKUP,
                RecoveryPhase.RESTORE,
                RecoveryPhase.RECOVERY,
                RecoveryPhase.VALIDATION,
                RecoveryPhase.MONITORING
            ],
            preconditions=[
                "Previous version backup exists",
                "Deployment can be rolled back",
                "No data migration issues"
            ],
            success_criteria=[
                "Previous version restored",
                "All services functional",
                "No data loss"
            ]
        )

        self.recovery_plans = {
            "service_restart": service_restart_plan,
            "data_restore": data_restore_plan,
            "failover": failover_plan,
            "rollback": rollback_plan
        }

    async def initiate_recovery(self, failure_description: str, plan_id: Optional[str] = None,
                              triggered_by: str = "system") -> Optional[RecoveryExecution]:
        """
        復旧開始

        Args:
            failure_description: 障害説明
            plan_id: 復旧プランID（自動選択の場合はNone）
            triggered_by: トリガー実行者

        Returns:
            Optional[RecoveryExecution]: 復旧実行オブジェクト
        """
        # 同時実行数チェック
        if len(self.active_recoveries) >= self.max_concurrent_recoveries:
            self.logger.warning("Maximum concurrent recoveries reached, queuing request")
            return None

        # プラン選択
        if not plan_id:
            plan_id = self._select_recovery_plan(failure_description)

        if plan_id not in self.recovery_plans:
            self.logger.error(f"Recovery plan not found: {plan_id}")
            return None

        plan = self.recovery_plans[plan_id]

        # 実行オブジェクト作成
        execution = RecoveryExecution(
            execution_id=f"RECOVERY_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            plan_id=plan_id,
            failure_description=failure_description,
            start_time=datetime.now(),
            status=RecoveryStatus.IN_PROGRESS
        )

        self.active_recoveries[execution.execution_id] = execution

        self.logger.warning(f"Recovery initiated: {execution.execution_id} using plan {plan_id}")

        # 非同期実行開始
        asyncio.create_task(self._execute_recovery_async(execution))

        return execution

    def _select_recovery_plan(self, failure_description: str) -> str:
        """
        復旧プラン自動選択

        Args:
            failure_description: 障害説明

        Returns:
            str: プランID
        """
        # 簡易的なキーワードマッチング
        description_lower = failure_description.lower()

        if "crash" in description_lower or "stopped" in description_lower:
            return "service_restart"
        elif "data" in description_lower or "corruption" in description_lower:
            return "data_restore"
        elif "node" in description_lower or "server" in description_lower:
            return "failover"
        elif "deployment" in description_lower or "update" in description_lower:
            return "rollback"
        else:
            return "service_restart"  # デフォルト

    async def _execute_recovery_async(self, execution: RecoveryExecution) -> None:
        """
        復旧非同期実行

        Args:
            execution: 復旧実行
        """
        plan = self.recovery_plans[execution.plan_id]

        try:
            # 前提条件チェック
            if not await self._check_preconditions(plan):
                execution.status = RecoveryStatus.FAILED
                await self._add_checkpoint(execution, RecoveryPhase.ASSESSMENT, RecoveryStatus.FAILED,
                                         "Preconditions not met")
                await self._notify_recovery_update(execution)
                return

            # 各フェーズ実行
            for phase in plan.phases:
                execution.current_phase = phase

                await self._add_checkpoint(execution, phase, RecoveryStatus.IN_PROGRESS,
                                         f"Starting {phase.value} phase")

                success = await self._execute_recovery_phase(execution, phase)

                if success:
                    await self._add_checkpoint(execution, phase, RecoveryStatus.COMPLETED,
                                             f"Completed {phase.value} phase")
                else:
                    await self._add_checkpoint(execution, phase, RecoveryStatus.FAILED,
                                             f"Failed {phase.value} phase")
                    await self._handle_recovery_failure(execution)
                    return

            # 成功基準チェック
            if await self._check_success_criteria(plan):
                execution.status = RecoveryStatus.COMPLETED
                execution.end_time = datetime.now()
                await self._add_checkpoint(execution, RecoveryPhase.MONITORING, RecoveryStatus.COMPLETED,
                                         "Recovery completed successfully")
            else:
                execution.status = RecoveryStatus.FAILED
                await self._handle_recovery_failure(execution)

        except Exception as e:
            execution.status = RecoveryStatus.FAILED
            await self._add_checkpoint(execution, execution.current_phase or RecoveryPhase.ASSESSMENT,
                                     RecoveryStatus.FAILED, f"Recovery error: {str(e)}")
            await self._handle_recovery_failure(execution)

        finally:
            # 履歴保存
            self.recovery_history.append(execution)

            # 履歴制限（最新100件）
            if len(self.recovery_history) > 100:
                self.recovery_history = self.recovery_history[-100:]

            # アクティブ実行から削除
            if execution.execution_id in self.active_recoveries:
                del self.active_recoveries[execution.execution_id]

            await self._notify_recovery_update(execution)

    async def _check_preconditions(self, plan: RecoveryPlan) -> bool:
        """
        前提条件チェック

        Args:
            plan: 復旧プラン

        Returns:
            bool: チェック成功フラグ
        """
        # 実際の実装では具体的な前提条件をチェック
        # ここでは簡易チェック
        await asyncio.sleep(0.1)
        return True

    async def _execute_recovery_phase(self, execution: RecoveryExecution, phase: RecoveryPhase) -> bool:
        """
        復旧フェーズ実行

        Args:
            execution: 復旧実行
            phase: 復旧フェーズ

        Returns:
            bool: 実行成功フラグ
        """
        try:
            if phase == RecoveryPhase.ASSESSMENT:
                return await self._execute_assessment_phase(execution)
            elif phase == RecoveryPhase.BACKUP:
                return await self._execute_backup_phase(execution)
            elif phase == RecoveryPhase.CLEANUP:
                return await self._execute_cleanup_phase(execution)
            elif phase == RecoveryPhase.RESTORE:
                return await self._execute_restore_phase(execution)
            elif phase == RecoveryPhase.VALIDATION:
                return await self._execute_validation_phase(execution)
            elif phase == RecoveryPhase.RECOVERY:
                return await self._execute_recovery_phase_core(execution)
            elif phase == RecoveryPhase.MONITORING:
                return await self._execute_monitoring_phase(execution)
            else:
                return False

        except Exception as e:
            self.logger.error(f"Recovery phase error: {phase.value} - {e}")
            return False

    async def _execute_assessment_phase(self, execution: RecoveryExecution) -> bool:
        """評価フェーズ実行"""
        # 障害の評価と影響範囲の特定
        await asyncio.sleep(1.0)
        execution.metrics['assessment_duration'] = 1.0
        return True

    async def _execute_backup_phase(self, execution: RecoveryExecution) -> bool:
        """バックアップフェーズ実行"""
        # 現在の状態のバックアップ
        await asyncio.sleep(2.0)
        execution.metrics['backup_size_mb'] = 150.5  # シミュレーション
        return True

    async def _execute_cleanup_phase(self, execution: RecoveryExecution) -> bool:
        """クリーンアップフェーズ実行"""
        # 障害状態のクリーンアップ
        await asyncio.sleep(1.5)
        execution.metrics['cleanup_duration'] = 1.5
        return True

    async def _execute_restore_phase(self, execution: RecoveryExecution) -> bool:
        """復元フェーズ実行"""
        # データや設定の復元
        await asyncio.sleep(3.0)
        execution.metrics['restore_duration'] = 3.0
        return True

    async def _execute_validation_phase(self, execution: RecoveryExecution) -> bool:
        """検証フェーズ実行"""
        # 復元結果の検証
        await asyncio.sleep(2.0)

        # ヘルスチェック実行
        validation_passed = True
        for check_name, check_func in self.health_checks.items():
            try:
                if not check_func():
                    validation_passed = False
                    break
            except Exception as e:
                self.logger.error(f"Health check failed: {check_name} - {e}")
                validation_passed = False
                break

        execution.metrics['validation_passed'] = validation_passed
        return validation_passed

    async def _execute_recovery_phase_core(self, execution: RecoveryExecution) -> bool:
        """復旧フェーズ実行（コア）"""
        # サービスの再起動や設定の適用
        await asyncio.sleep(2.5)
        execution.metrics['recovery_duration'] = 2.5
        return True

    async def _execute_monitoring_phase(self, execution: RecoveryExecution) -> bool:
        """監視フェーズ実行"""
        # 復旧後の安定性監視
        await asyncio.sleep(1.0)

        # 簡易的な安定性チェック
        stable = True
        execution.metrics['monitoring_stable'] = stable
        return stable

    async def _check_success_criteria(self, plan: RecoveryPlan) -> bool:
        """
        成功基準チェック

        Args:
            plan: 復旧プラン

        Returns:
            bool: チェック成功フラグ
        """
        # 実際の実装では具体的な成功基準をチェック
        await asyncio.sleep(0.5)
        return True

    async def _handle_recovery_failure(self, execution: RecoveryExecution) -> None:
        """
        復旧失敗処理

        Args:
            execution: 復旧実行
        """
        self.logger.error(f"Recovery failed: {execution.execution_id}")

        # 自動ロールバック判定
        plan = self.recovery_plans[execution.plan_id]
        if plan.risk_level in ['high', 'critical']:
            await self._trigger_rollback(execution)

    async def _trigger_rollback(self, execution: RecoveryExecution) -> None:
        """
        ロールバックトリガー

        Args:
            execution: 復旧実行
        """
        execution.rollback_triggered = True
        await self._add_checkpoint(execution, RecoveryPhase.RECOVERY, RecoveryStatus.ROLLED_BACK,
                                 "Automatic rollback triggered due to recovery failure")

        self.logger.warning(f"Automatic rollback triggered for recovery: {execution.execution_id}")

    async def _add_checkpoint(self, execution: RecoveryExecution, phase: RecoveryPhase,
                            status: RecoveryStatus, description: str,
                            metadata: Optional[Dict[str, Any]] = None,
                            error_message: Optional[str] = None) -> None:
        """
        チェックポイント追加

        Args:
            execution: 復旧実行
            phase: フェーズ
            status: ステータス
            description: 説明
            metadata: メタデータ
            error_message: エラーメッセージ
        """
        checkpoint = RecoveryCheckpoint(
            checkpoint_id=f"CP_{execution.execution_id}_{phase.value}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now(),
            phase=phase,
            status=status,
            description=description,
            metadata=metadata or {},
            error_message=error_message
        )

        execution.checkpoints.append(checkpoint)

        # コールバック実行
        for callback in self.phase_callbacks:
            try:
                asyncio.create_task(callback(checkpoint))
            except Exception as e:
                self.logger.error(f"Phase callback error: {e}")

    async def _notify_recovery_update(self, execution: RecoveryExecution) -> None:
        """
        復旧更新通知

        Args:
            execution: 復旧実行
        """
        for callback in self.recovery_callbacks:
            try:
                asyncio.create_task(callback(execution))
            except Exception as e:
                self.logger.error(f"Recovery callback error: {e}")

    def cancel_recovery(self, execution_id: str, cancelled_by: str = "system") -> bool:
        """
        復旧キャンセル

        Args:
            execution_id: 実行ID
            cancelled_by: キャンセル実行者

        Returns:
            bool: キャンセル成功フラグ
        """
        if execution_id not in self.active_recoveries:
            return False

        execution = self.active_recoveries[execution_id]
        if execution.status in [RecoveryStatus.COMPLETED, RecoveryStatus.FAILED]:
            return False

        execution.status = RecoveryStatus.FAILED
        execution.end_time = datetime.now()
        # await self._add_checkpoint(execution, execution.current_phase or RecoveryPhase.ASSESSMENT,
        #                          RecoveryStatus.FAILED, f"Cancelled by {cancelled_by}")

        del self.active_recoveries[execution_id]

        self.logger.info(f"Recovery cancelled: {execution_id}")
        return True

    def get_active_recoveries(self) -> List[RecoveryExecution]:
        """
        アクティブ復旧取得

        Returns:
            List[RecoveryExecution]: アクティブ復旧リスト
        """
        return list(self.active_recoveries.values())

    def get_recovery_history(self, limit: Optional[int] = None) -> List[RecoveryExecution]:
        """
        復旧履歴取得

        Args:
            limit: 取得件数制限

        Returns:
            List[RecoveryExecution]: 復旧履歴
        """
        history = self.recovery_history
        if limit:
            history = history[-limit:]
        return history

    def add_recovery_plan(self, plan: RecoveryPlan) -> None:
        """
        復旧プラン追加

        Args:
            plan: 復旧プラン
        """
        self.recovery_plans[plan.plan_id] = plan
        self.logger.info(f"Recovery plan added: {plan.plan_id}")

    def add_health_check(self, check_name: str, check_func: Callable[[], bool]) -> None:
        """
        ヘルスチェック追加

        Args:
            check_name: チェック名
            check_func: チェック関数
        """
        self.health_checks[check_name] = check_func
        self.logger.info(f"Health check added: {check_name}")

    def configure_backup(self, backup_id: str, config: Dict[str, Any]) -> None:
        """
        バックアップ設定

        Args:
            backup_id: バックアップID
            config: 設定
        """
        self.backup_configs[backup_id] = config
        self.logger.info(f"Backup configured: {backup_id}")

    def schedule_backup(self, backup_id: str, cron_expression: str) -> None:
        """
        バックアップスケジュール設定

        Args:
            backup_id: バックアップID
            cron_expression: cron式
        """
        self.backup_schedule[backup_id] = cron_expression
        self.logger.info(f"Backup scheduled: {backup_id} - {cron_expression}")

    def create_backup(self, backup_id: str) -> bool:
        """
        バックアップ作成

        Args:
            backup_id: バックアップID

        Returns:
            bool: 作成成功フラグ
        """
        if backup_id not in self.backup_configs:
            self.logger.error(f"Backup config not found: {backup_id}")
            return False

        config = self.backup_configs[backup_id]

        try:
            # 実際の実装では具体的なバックアップロジック
            source_path = config.get('source_path')
            backup_path = config.get('backup_path')

            if source_path and backup_path:
                # ディレクトリコピー
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, backup_path)

                self.logger.info(f"Backup created: {backup_id}")
                return True
            else:
                self.logger.error(f"Invalid backup paths for {backup_id}")
                return False

        except Exception as e:
            self.logger.error(f"Backup creation failed: {backup_id} - {e}")
            return False

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """
        復旧メトリクス取得

        Returns:
            Dict[str, Any]: メトリクス
        """
        total_recoveries = len(self.recovery_history)
        successful_recoveries = len([r for r in self.recovery_history if r.status == RecoveryStatus.COMPLETED])
        failed_recoveries = len([r for r in self.recovery_history if r.status == RecoveryStatus.FAILED])

        success_rate = (successful_recoveries / total_recoveries * 100) if total_recoveries > 0 else 0

        avg_duration = None
        durations = [r.end_time - r.start_time for r in self.recovery_history
                    if r.end_time and r.status == RecoveryStatus.COMPLETED]
        if durations:
            avg_duration = sum((d.total_seconds() for d in durations), 0) / len(durations)

        return {
            'total_recoveries': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'failed_recoveries': failed_recoveries,
            'success_rate_percent': success_rate,
            'active_recoveries': len(self.active_recoveries),
            'average_recovery_duration_seconds': avg_duration
        }

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Recovery monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Recovery monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # バックアップスケジュールチェック
                self._check_backup_schedule()

                time.sleep(60)  # 1分間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def _check_backup_schedule(self) -> None:
        """バックアップスケジュールチェック"""
        # cron式の評価は簡易的に毎時実行
        current_minute = datetime.now().minute
        if current_minute == 0:  # 毎時0分
            for backup_id in self.backup_schedule.keys():
                asyncio.run(self._create_backup_async(backup_id))

    async def _create_backup_async(self, backup_id: str) -> None:
        """
        非同期バックアップ作成

        Args:
            backup_id: バックアップID
        """
        try:
            success = self.create_backup(backup_id)
            if success:
                self.logger.info(f"Scheduled backup completed: {backup_id}")
            else:
                self.logger.error(f"Scheduled backup failed: {backup_id}")
        except Exception as e:
            self.logger.error(f"Scheduled backup error: {backup_id} - {e}")

    def add_recovery_callback(self, callback: Callable[[RecoveryExecution], Awaitable[None]]) -> None:
        """
        復旧コールバック追加

        Args:
            callback: コールバック関数
        """
        self.recovery_callbacks.append(callback)

    def add_phase_callback(self, callback: Callable[[RecoveryCheckpoint], Awaitable[None]]) -> None:
        """
        フェーズコールバック追加

        Args:
            callback: コールバック関数
        """
        self.phase_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            'system_name': self.system_name,
            'max_concurrent_recoveries': self.max_concurrent_recoveries,
            'recovery_history': [
                {
                    'execution_id': r.execution_id,
                    'plan_id': r.plan_id,
                    'failure_description': r.failure_description,
                    'start_time': r.start_time.isoformat(),
                    'end_time': r.end_time.isoformat() if r.end_time else None,
                    'status': r.status.value,
                    'current_phase': r.current_phase.value if r.current_phase else None,
                    'metrics': r.metrics,
                    'rollback_triggered': r.rollback_triggered,
                    'checkpoints': [
                        {
                            'checkpoint_id': cp.checkpoint_id,
                            'timestamp': cp.timestamp.isoformat(),
                            'phase': cp.phase.value,
                            'status': cp.status.value,
                            'description': cp.description,
                            'metadata': cp.metadata,
                            'error_message': cp.error_message
                        }
                        for cp in r.checkpoints
                    ]
                }
                for r in self.recovery_history[-50:]  # 最新50件
            ],
            'backup_configs': self.backup_configs,
            'backup_schedule': self.backup_schedule
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Recovery system state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.system_name = state['system_name']
            self.max_concurrent_recoveries = state['max_concurrent_recoveries']

            # 復旧履歴復元
            self.recovery_history = []
            for r_data in state.get('recovery_history', []):
                execution = RecoveryExecution(
                    execution_id=r_data['execution_id'],
                    plan_id=r_data['plan_id'],
                    failure_description=r_data['failure_description'],
                    start_time=datetime.fromisoformat(r_data['start_time']),
                    end_time=datetime.fromisoformat(r_data['end_time']) if r_data['end_time'] else None,
                    status=RecoveryStatus(r_data['status']),
                    current_phase=RecoveryPhase(r_data['current_phase']) if r_data['current_phase'] else None,
                    metrics=r_data['metrics'],
                    rollback_triggered=r_data['rollback_triggered']
                )

                # チェックポイント復元
                for cp_data in r_data.get('checkpoints', []):
                    checkpoint = RecoveryCheckpoint(
                        checkpoint_id=cp_data['checkpoint_id'],
                        timestamp=datetime.fromisoformat(cp_data['timestamp']),
                        phase=RecoveryPhase(cp_data['phase']),
                        status=RecoveryStatus(cp_data['status']),
                        description=cp_data['description'],
                        metadata=cp_data['metadata'],
                        error_message=cp_data['error_message']
                    )
                    execution.checkpoints.append(checkpoint)

                self.recovery_history.append(execution)

            # バックアップ設定復元
            self.backup_configs = state.get('backup_configs', {})
            self.backup_schedule = state.get('backup_schedule', {})

            self.logger.info(f"Recovery system state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load recovery system state: {e}")
            return False