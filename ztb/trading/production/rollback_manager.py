"""
V433 Phase 5: Gradual Rollout Layer - Rollback Manager

システムの問題発生時に安全にロールバックを行う。
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)

class RollbackTrigger(Enum):
    """ロールバックトリガー"""

    MANUAL = "manual"  # 手動
    AUTOMATIC = "automatic"  # 自動
    PERFORMANCE = "performance"  # パフォーマンス
    SYSTEM_FAILURE = "system_failure"  # システム障害
    EXTERNAL_SIGNAL = "external_signal"  # 外部シグナル

class RollbackState(Enum):
    """ロールバック状態"""

    IDLE = "idle"  # 待機中
    PREPARING = "preparing"  # 準備中
    EXECUTING = "executing"  # 実行中
    COMPLETED = "completed"  # 完了
    FAILED = "failed"  # 失敗
    CANCELLED = "cancelled"  # キャンセル

@dataclass
class RollbackCheckpoint:
    """ロールバックチェックポイント"""

    checkpoint_id: str
    timestamp: datetime
    system_id: str
    allocation_percentage: float
    performance_metrics: dict[str, float]
    system_state: dict[str, Any]
    is_stable: bool = True
    description: str = ""

@dataclass
class RollbackPlan:
    """ロールバックプラン"""

    plan_id: str
    timestamp: datetime
    trigger: RollbackTrigger
    reason: str
    target_allocation: float
    estimated_duration_minutes: int
    risk_assessment: str
    checkpoints: list[RollbackCheckpoint] = field(default_factory=list)
    executed_steps: list[str] = field(default_factory=list)

@dataclass
class RollbackExecution:
    """ロールバック実行"""

    execution_id: str
    plan_id: str
    start_time: datetime
    end_time: datetime | None = None
    state: RollbackState = RollbackState.IDLE
    progress_percentage: float = 0.0
    current_step: str = ""
    error_message: str | None = None
    rollback_metrics: dict[str, Any] = field(default_factory=dict)

class RollbackManager:
    """
    ロールバックマネージャー

    システムの問題発生時に安全にロールバックを行う。
    自動および手動のロールバックをサポート。
    """

    def __init__(
        self,
        max_checkpoints_per_system: int = 10,
        auto_rollback_threshold: float = 0.8,
        rollback_timeout_minutes: int = 30,
    ):
        """
        初期化

        Args:
            max_checkpoints_per_system: システムごとの最大チェックポイント数
            auto_rollback_threshold: 自動ロールバック閾値（アロケーション割合）
            rollback_timeout_minutes: ロールバックタイムアウト（分）
        """
        self.max_checkpoints_per_system = max_checkpoints_per_system
        self.auto_rollback_threshold = auto_rollback_threshold
        self.rollback_timeout_minutes = rollback_timeout_minutes

        # チェックポイント管理
        self.checkpoints: dict[str, list[RollbackCheckpoint]] = {}

        # ロールバック実行管理
        self.active_executions: dict[str, RollbackExecution] = {}
        self.execution_history: list[RollbackExecution] = []

        # ロールバックプラン
        self.rollback_plans: dict[str, RollbackPlan] = {}

        # 自動ロールバック設定
        self.auto_rollback_enabled = True
        self.auto_rollback_conditions: dict[str, Callable[[], bool]] = {}

        # コールバック
        self.rollback_callbacks: list[
            Callable[[RollbackExecution], Awaitable[None]]
        ] = []
        self.checkpoint_callbacks: list[
            Callable[[RollbackCheckpoint], Awaitable[None]]
        ] = []

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Rollback Manager initialized")

    def create_checkpoint(
        self,
        system_id: str,
        allocation_percentage: float,
        performance_metrics: dict[str, float],
        system_state: dict[str, Any],
        description: str = "",
    ) -> RollbackCheckpoint:
        """
        チェックポイント作成

        Args:
            system_id: システムID
            allocation_percentage: アロケーション割合
            performance_metrics: パフォーマンス指標
            system_state: システム状態
            description: 説明

        Returns:
            RollbackCheckpoint: 作成されたチェックポイント
        """
        checkpoint = RollbackCheckpoint(
            checkpoint_id=f"CP_{system_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            system_id=system_id,
            allocation_percentage=allocation_percentage,
            performance_metrics=performance_metrics.copy(),
            system_state=system_state.copy(),
            description=description,
        )

        # チェックポイント保存
        if system_id not in self.checkpoints:
            self.checkpoints[system_id] = []

        self.checkpoints[system_id].append(checkpoint)

        # 古いチェックポイント削除
        if len(self.checkpoints[system_id]) > self.max_checkpoints_per_system:
            self.checkpoints[system_id] = self.checkpoints[system_id][
                -self.max_checkpoints_per_system :
            ]

        # コールバック実行
        for callback in self.checkpoint_callbacks:
            try:
                asyncio.create_task(callback(checkpoint))
            except Exception as e:
                self.logger.error(f"Checkpoint callback error: {e}")

        self.logger.info(
            f"Checkpoint created: {checkpoint.checkpoint_id} for system {system_id}"
        )
        return checkpoint

    def get_latest_stable_checkpoint(
        self, system_id: str
    ) -> RollbackCheckpoint | None:
        """
        最新の安定チェックポイント取得

        Args:
            system_id: システムID

        Returns:
            RollbackCheckpoint | None: 最新の安定チェックポイント
        """
        if system_id not in self.checkpoints:
            return None

        stable_checkpoints = [cp for cp in self.checkpoints[system_id] if cp.is_stable]
        return (
            max(stable_checkpoints, key=lambda cp: cp.timestamp)
            if stable_checkpoints
            else None
        )

    def mark_checkpoint_unstable(self, checkpoint_id: str) -> bool:
        """
        チェックポイントを不安定としてマーク

        Args:
            checkpoint_id: チェックポイントID

        Returns:
            bool: マーク成功フラグ
        """
        for system_checkpoints in self.checkpoints.values():
            for checkpoint in system_checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id:
                    checkpoint.is_stable = False
                    self.logger.warning(
                        f"Checkpoint marked as unstable: {checkpoint_id}"
                    )
                    return True

        return False

    def initiate_rollback(
        self,
        system_id: str,
        trigger: RollbackTrigger,
        reason: str,
        target_allocation: float | None = None,
    ) -> RollbackExecution | None:
        """
        ロールバック開始

        Args:
            system_id: システムID
            trigger: ロールバックトリガー
            reason: 理由
            target_allocation: 目標アロケーション（指定なしの場合は最新安定チェックポイント）

        Returns:
            RollbackExecution | None: ロールバック実行オブジェクト
        """
        # ターゲットチェックポイント決定
        target_checkpoint = None
        if target_allocation is None:
            target_checkpoint = self.get_latest_stable_checkpoint(system_id)
            if not target_checkpoint:
                self.logger.error(f"No stable checkpoint found for system {system_id}")
                return None
            target_allocation = target_checkpoint.allocation_percentage

        # ロールバックプラン作成
        plan = RollbackPlan(
            plan_id=f"PLAN_{system_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            trigger=trigger,
            reason=reason,
            target_allocation=target_allocation,
            estimated_duration_minutes=self._estimate_rollback_duration(
                target_allocation
            ),
            risk_assessment=self._assess_rollback_risk(system_id, target_allocation),
        )

        if target_checkpoint:
            plan.checkpoints = [target_checkpoint]

        self.rollback_plans[plan.plan_id] = plan

        # ロールバック実行開始
        execution = RollbackExecution(
            execution_id=f"EXEC_{plan.plan_id}",
            plan_id=plan.plan_id,
            start_time=datetime.now(),
            state=RollbackState.PREPARING,
        )

        self.active_executions[system_id] = execution

        # 非同期実行開始
        asyncio.create_task(self._execute_rollback_async(execution, plan))

        self.logger.warning(f"Rollback initiated for system {system_id}: {reason}")
        return execution

    def _estimate_rollback_duration(self, target_allocation: float) -> int:
        """
        ロールバック期間推定

        Args:
            target_allocation: 目標アロケーション

        Returns:
            int: 推定期間（分）
        """
        # アロケーション減少量に基づく推定
        if target_allocation >= 0.8:
            return 5  # 高速ロールバック
        elif target_allocation >= 0.5:
            return 15  # 中速ロールバック
        elif target_allocation >= 0.2:
            return 30  # 低速ロールバック
        else:
            return 60  # 完全ロールバック

    def _assess_rollback_risk(self, system_id: str, target_allocation: float) -> str:
        """
        ロールバックリスク評価

        Args:
            system_id: システムID
            target_allocation: 目標アロケーション

        Returns:
            str: リスク評価
        """
        if target_allocation >= 0.8:
            return "LOW"
        elif target_allocation >= 0.5:
            return "MEDIUM"
        elif target_allocation >= 0.2:
            return "HIGH"
        else:
            return "CRITICAL"

    async def _execute_rollback_async(
        self, execution: RollbackExecution, plan: RollbackPlan
    ) -> None:
        """
        ロールバック非同期実行

        Args:
            execution: ロールバック実行
            plan: ロールバックプラン
        """
        try:
            # 準備フェーズ
            execution.state = RollbackState.PREPARING
            execution.current_step = "Validating rollback plan"
            await self._notify_rollback_update(execution)

            # プラン検証
            if not await self._validate_rollback_plan(plan):
                execution.state = RollbackState.FAILED
                execution.error_message = "Rollback plan validation failed"
                await self._notify_rollback_update(execution)
                return

            # 実行フェーズ
            execution.state = RollbackState.EXECUTING
            execution.current_step = "Executing rollback steps"
            await self._notify_rollback_update(execution)

            # ロールバック実行
            success = await self._perform_rollback_steps(plan, execution)

            if success:
                execution.state = RollbackState.COMPLETED
                execution.end_time = datetime.now()
                execution.progress_percentage = 100.0
            else:
                execution.state = RollbackState.FAILED
                execution.error_message = "Rollback execution failed"

        except Exception as e:
            execution.state = RollbackState.FAILED
            execution.error_message = str(e)
            self.logger.error(f"Rollback execution error: {e}")

        finally:
            # 完了通知
            await self._notify_rollback_update(execution)

            # 履歴保存
            self.execution_history.append(execution)

            # アクティブ実行から削除
            system_id = plan.checkpoints[0].system_id if plan.checkpoints else "unknown"
            if system_id in self.active_executions:
                del self.active_executions[system_id]

    async def _validate_rollback_plan(self, plan: RollbackPlan) -> bool:
        """
        ロールバックプラン検証

        Args:
            plan: ロールバックプラン

        Returns:
            bool: 検証成功フラグ
        """
        # 基本検証
        if plan.target_allocation < 0 or plan.target_allocation > 1:
            self.logger.error("Invalid target allocation")
            return False

        # タイムアウトチェック
        if datetime.now() - plan.timestamp > timedelta(
            minutes=self.rollback_timeout_minutes
        ):
            self.logger.error("Rollback plan expired")
            return False

        # システム状態チェック（実際の実装ではシステムのヘルスチェック）
        await asyncio.sleep(0.1)  # シミュレーション

        return True

    async def _perform_rollback_steps(
        self, plan: RollbackPlan, execution: RollbackExecution
    ) -> bool:
        """
        ロールバックステップ実行

        Args:
            plan: ロールバックプラン
            execution: ロールバック実行

        Returns:
            bool: 実行成功フラグ
        """
        steps = [
            "Stopping new traffic allocation",
            "Gradually reducing allocation",
            "Validating system stability",
            "Confirming rollback completion",
        ]

        total_steps = len(steps)

        for i, step in enumerate(steps):
            execution.current_step = step
            execution.progress_percentage = (i / total_steps) * 100

            await self._notify_rollback_update(execution)

            # ステップ実行（実際の実装では具体的なロールバック操作）
            success = await self._execute_rollback_step(plan, step)
            if not success:
                return False

            plan.executed_steps.append(step)

            # ステップ間待機
            await asyncio.sleep(1.0)

        return True

    async def _execute_rollback_step(self, plan: RollbackPlan, step: str) -> bool:
        """
        個別ロールバックステップ実行

        Args:
            plan: ロールバックプラン
            step: ステップ名

        Returns:
            bool: 実行成功フラグ
        """
        try:
            # 実際の実装ではここで具体的なロールバック操作を行う
            # 例: アロケーションマネージャーへの指示、トラフィック制御など

            if step == "Stopping new traffic allocation":
                # 新規トラフィック割り当て停止
                await asyncio.sleep(0.5)

            elif step == "Gradually reducing allocation":
                # 段階的にアロケーション削減
                await asyncio.sleep(2.0)

            elif step == "Validating system stability":
                # システム安定性検証
                await asyncio.sleep(1.0)

            elif step == "Confirming rollback completion":
                # ロールバック完了確認
                await asyncio.sleep(0.5)

            # 稀に失敗をシミュレート
            import random

            if random.random() < 0.05:  # 5%の確率
                raise Exception(f"Simulated failure in step: {step}")

            return True

        except Exception as e:
            self.logger.error(f"Rollback step failed: {step} - {e}")
            return False

    async def _notify_rollback_update(self, execution: RollbackExecution) -> None:
        """
        ロールバック更新通知

        Args:
            execution: ロールバック実行
        """
        for callback in self.rollback_callbacks:
            try:
                asyncio.create_task(callback(execution))
            except Exception as e:
                self.logger.error(f"Rollback callback error: {e}")

    def cancel_rollback(self, system_id: str) -> bool:
        """
        ロールバックキャンセル

        Args:
            system_id: システムID

        Returns:
            bool: キャンセル成功フラグ
        """
        if system_id not in self.active_executions:
            return False

        execution = self.active_executions[system_id]
        if execution.state in [RollbackState.COMPLETED, RollbackState.FAILED]:
            return False

        execution.state = RollbackState.CANCELLED
        execution.end_time = datetime.now()

        # 履歴保存
        self.execution_history.append(execution)

        del self.active_executions[system_id]

        self.logger.info(f"Rollback cancelled for system {system_id}")
        return True

    def get_active_rollback(self, system_id: str) -> RollbackExecution | None:
        """
        アクティブロールバック取得

        Args:
            system_id: システムID

        Returns:
            RollbackExecution | None: アクティブロールバック実行
        """
        return self.active_executions.get(system_id)

    def get_rollback_history(
        self, system_id: str | None = None, limit: int | None = None
    ) -> list[RollbackExecution]:
        """
        ロールバック履歴取得

        Args:
            system_id: システムID（指定なしの場合は全システム）
            limit: 取得件数制限

        Returns:
            list[RollbackExecution]: ロールバック履歴
        """
        history = self.execution_history

        if system_id:
            history = [
                exec for exec in history if exec.plan_id.startswith(f"PLAN_{system_id}")
            ]

        if limit:
            history = history[-limit:]

        return history

    def add_auto_rollback_condition(
        self, condition_id: str, condition_func: Callable[[], bool]
    ) -> None:
        """
        自動ロールバック条件追加

        Args:
            condition_id: 条件ID
            condition_func: 条件評価関数
        """
        self.auto_rollback_conditions[condition_id] = condition_func
        self.logger.info(f"Auto rollback condition added: {condition_id}")

    def remove_auto_rollback_condition(self, condition_id: str) -> None:
        """
        自動ロールバック条件削除

        Args:
            condition_id: 条件ID
        """
        if condition_id in self.auto_rollback_conditions:
            del self.auto_rollback_conditions[condition_id]
            self.logger.info(f"Auto rollback condition removed: {condition_id}")

    def check_auto_rollback_conditions(self, system_id: str) -> list[str]:
        """
        自動ロールバック条件チェック

        Args:
            system_id: システムID

        Returns:
            list[str]: トリガーされた条件IDリスト
        """
        triggered_conditions = []

        for condition_id, condition_func in self.auto_rollback_conditions.items():
            try:
                if condition_func():
                    triggered_conditions.append(condition_id)
            except Exception as e:
                self.logger.error(
                    f"Auto rollback condition check error for {condition_id}: {e}"
                )

        return triggered_conditions

    def trigger_auto_rollback(
        self, system_id: str, triggered_conditions: list[str]
    ) -> RollbackExecution | None:
        """
        自動ロールバックトリガー

        Args:
            system_id: システムID
            triggered_conditions: トリガーされた条件

        Returns:
            RollbackExecution | None: ロールバック実行
        """
        if not self.auto_rollback_enabled:
            return None

        reason = (
            f"Auto rollback triggered by conditions: {', '.join(triggered_conditions)}"
        )

        # 自動ロールバックではアロケーションを安全なレベルに設定
        target_allocation = min(self.auto_rollback_threshold, 0.5)  # 最大50%まで

        return self.initiate_rollback(
            system_id, RollbackTrigger.AUTOMATIC, reason, target_allocation
        )

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Rollback monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Rollback monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # 自動ロールバック条件チェック
                for system_id in self.checkpoints.keys():
                    triggered_conditions = self.check_auto_rollback_conditions(
                        system_id
                    )
                    if triggered_conditions:
                        self.trigger_auto_rollback(system_id, triggered_conditions)

                # タイムアウトチェック
                self._check_rollback_timeouts()

                time.sleep(60)  # 1分間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def _check_rollback_timeouts(self) -> None:
        """ロールバックタイムアウトチェック"""
        current_time = datetime.now()
        timeout_threshold = timedelta(minutes=self.rollback_timeout_minutes)

        for system_id, execution in list(self.active_executions.items()):
            if current_time - execution.start_time > timeout_threshold:
                self.logger.error(f"Rollback timeout for system {system_id}")
                execution.state = RollbackState.FAILED
                execution.error_message = "Rollback timeout"
                execution.end_time = current_time

                # 履歴保存
                self.execution_history.append(execution)
                del self.active_executions[system_id]

    def add_rollback_callback(
        self, callback: Callable[[RollbackExecution], Awaitable[None]]
    ) -> None:
        """
        ロールバックコールバック追加

        Args:
            callback: コールバック関数
        """
        self.rollback_callbacks.append(callback)

    def add_checkpoint_callback(
        self, callback: Callable[[RollbackCheckpoint], Awaitable[None]]
    ) -> None:
        """
        チェックポイントコールバック追加

        Args:
            callback: コールバック関数
        """
        self.checkpoint_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "max_checkpoints_per_system": self.max_checkpoints_per_system,
            "auto_rollback_threshold": self.auto_rollback_threshold,
            "rollback_timeout_minutes": self.rollback_timeout_minutes,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "checkpoints": {
                system_id: [
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "timestamp": cp.timestamp.isoformat(),
                        "system_id": cp.system_id,
                        "allocation_percentage": cp.allocation_percentage,
                        "performance_metrics": cp.performance_metrics,
                        "system_state": cp.system_state,
                        "is_stable": cp.is_stable,
                        "description": cp.description,
                    }
                    for cp in checkpoints[
                        -self.max_checkpoints_per_system :
                    ]  # 最新のみ
                ]
                for system_id, checkpoints in self.checkpoints.items()
            },
            "execution_history": [
                {
                    "execution_id": exec.execution_id,
                    "plan_id": exec.plan_id,
                    "start_time": exec.start_time.isoformat(),
                    "end_time": exec.end_time.isoformat() if exec.end_time else None,
                    "state": exec.state.value,
                    "progress_percentage": exec.progress_percentage,
                    "current_step": exec.current_step,
                    "error_message": exec.error_message,
                    "rollback_metrics": exec.rollback_metrics,
                }
                for exec in self.execution_history[-50:]  # 最新50件
            ],
            "rollback_plans": {
                plan_id: {
                    "plan_id": plan.plan_id,
                    "timestamp": plan.timestamp.isoformat(),
                    "trigger": plan.trigger.value,
                    "reason": plan.reason,
                    "target_allocation": plan.target_allocation,
                    "estimated_duration_minutes": plan.estimated_duration_minutes,
                    "risk_assessment": plan.risk_assessment,
                    "executed_steps": plan.executed_steps,
                }
                for plan_id, plan in self.rollback_plans.items()
            },
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Rollback manager state saved to {filepath}")

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

            self.max_checkpoints_per_system = state["max_checkpoints_per_system"]
            self.auto_rollback_threshold = state["auto_rollback_threshold"]
            self.rollback_timeout_minutes = state["rollback_timeout_minutes"]
            self.auto_rollback_enabled = state["auto_rollback_enabled"]

            # チェックポイント復元
            self.checkpoints = {}
            for system_id, cp_list in state.get("checkpoints", {}).items():
                self.checkpoints[system_id] = []
                for cp_data in cp_list:
                    checkpoint = RollbackCheckpoint(
                        checkpoint_id=cp_data["checkpoint_id"],
                        timestamp=datetime.fromisoformat(cp_data["timestamp"]),
                        system_id=cp_data["system_id"],
                        allocation_percentage=cp_data["allocation_percentage"],
                        performance_metrics=cp_data["performance_metrics"],
                        system_state=cp_data["system_state"],
                        is_stable=cp_data["is_stable"],
                        description=cp_data["description"],
                    )
                    self.checkpoints[system_id].append(checkpoint)

            # 実行履歴復元
            self.execution_history = []
            for exec_data in state.get("execution_history", []):
                execution = RollbackExecution(
                    execution_id=exec_data["execution_id"],
                    plan_id=exec_data["plan_id"],
                    start_time=datetime.fromisoformat(exec_data["start_time"]),
                    end_time=datetime.fromisoformat(exec_data["end_time"])
                    if exec_data["end_time"]
                    else None,
                    state=RollbackState(exec_data["state"]),
                    progress_percentage=exec_data["progress_percentage"],
                    current_step=exec_data["current_step"],
                    error_message=exec_data["error_message"],
                    rollback_metrics=exec_data["rollback_metrics"],
                )
                self.execution_history.append(execution)

            # ロールバックプラン復元
            self.rollback_plans = {}
            for plan_id, plan_data in state.get("rollback_plans", {}).items():
                plan = RollbackPlan(
                    plan_id=plan_data["plan_id"],
                    timestamp=datetime.fromisoformat(plan_data["timestamp"]),
                    trigger=RollbackTrigger(plan_data["trigger"]),
                    reason=plan_data["reason"],
                    target_allocation=plan_data["target_allocation"],
                    estimated_duration_minutes=plan_data["estimated_duration_minutes"],
                    risk_assessment=plan_data["risk_assessment"],
                    executed_steps=plan_data["executed_steps"],
                )
                self.rollback_plans[plan_id] = plan

            self.logger.info(f"Rollback manager state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load rollback manager state: {e}")
            return False
