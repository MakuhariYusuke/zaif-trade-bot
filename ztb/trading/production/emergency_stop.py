"""
V433 Phase 5: Emergency Control Layer - Emergency Stop

緊急時にシステムを即座に停止し、安全な状態に移行する。
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional


class EmergencyStopLevel(Enum):
    """緊急停止レベル"""

    WARNING = "warning"  # 警告（一部機能停止）
    CRITICAL = "critical"  # 重要（主要機能停止）
    SEVERE = "severe"  # 重大（全機能停止）
    CATASTROPHIC = "catastrophic"  # 壊滅的（完全停止）


class EmergencyStopTrigger(Enum):
    """緊急停止トリガー"""

    MANUAL = "manual"  # 手動
    SYSTEM_FAILURE = "system_failure"  # システム障害
    SECURITY_BREACH = "security_breach"  # セキュリティ侵害
    DATA_CORRUPTION = "data_corruption"  # データ破損
    EXTERNAL_THREAT = "external_threat"  # 外部脅威
    PERFORMANCE_CRITICAL = "performance_critical"  # パフォーマンス危機
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # リソース枯渇


class SystemComponent(Enum):
    """システムコンポーネント"""

    TRADING_ENGINE = "trading_engine"  # 取引エンジン
    MARKET_DATA = "market_data"  # 市場データ
    RISK_MANAGER = "risk_manager"  # リスクマネージャー
    PORTFOLIO_MANAGER = "portfolio_manager"  # ポートフォリオマネージャー
    ORDER_MANAGER = "order_manager"  # オーダーマネージャー
    DATABASE = "database"  # データベース
    NETWORK = "network"  # ネットワーク
    MONITORING = "monitoring"  # 監視システム


@dataclass
class EmergencyStopAction:
    """緊急停止アクション"""

    action_id: str
    component: SystemComponent
    action_type: str  # 'stop', 'pause', 'isolate', 'shutdown'
    priority: int  # 実行優先度（高いほど先に実行）
    timeout_seconds: int = 30
    rollback_action: Optional[str] = None  # 復旧時のアクション


@dataclass
class EmergencyStopEvent:
    """緊急停止イベント"""

    event_id: str
    timestamp: datetime
    trigger: EmergencyStopTrigger
    level: EmergencyStopLevel
    reason: str
    triggered_by: str
    actions_taken: List[str] = field(default_factory=list)
    status: str = "active"  # 'active', 'completed', 'failed', 'cancelled'


@dataclass
class EmergencyStopPlan:
    """緊急停止プラン"""

    plan_id: str
    level: EmergencyStopLevel
    name: str
    description: str
    actions: List[EmergencyStopAction] = field(default_factory=list)
    auto_trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_seconds: int = 60


class EmergencyStop:
    """
    緊急停止システム

    緊急時にシステムを即座に停止し、安全な状態に移行する。
    段階的な停止と自動トリガーをサポート。
    """

    def __init__(self, system_name: str = "V433 Trading System"):
        """
        初期化

        Args:
            system_name: システム名
        """
        self.system_name = system_name
        self.is_emergency_stop_active = False
        self.current_stop_level: Optional[EmergencyStopLevel] = None
        self.stop_timestamp: Optional[datetime] = None

        # 緊急停止プラン
        self.stop_plans: Dict[EmergencyStopLevel, EmergencyStopPlan] = {}
        self._initialize_default_plans()

        # 自動トリガー条件
        self.auto_trigger_conditions: Dict[str, Callable[[], bool]] = {}

        # イベント履歴
        self.stop_events: List[EmergencyStopEvent] = []

        # コンポーネント状態
        self.component_states: Dict[SystemComponent, str] = {}
        self._initialize_component_states()

        # コールバック
        self.stop_callbacks: List[Callable[[EmergencyStopEvent], Awaitable[None]]] = []
        self.recovery_callbacks: List[
            Callable[[EmergencyStopEvent], Awaitable[None]]
        ] = []

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Emergency Stop initialized for {system_name}")

    def _initialize_default_plans(self) -> None:
        """デフォルト停止プラン初期化"""
        # 警告レベル
        warning_plan = EmergencyStopPlan(
            plan_id="plan_warning",
            level=EmergencyStopLevel.WARNING,
            name="Warning Level Stop",
            description="一部機能の停止",
            actions=[
                EmergencyStopAction(
                    "action_pause_trading", SystemComponent.TRADING_ENGINE, "pause", 1
                ),
                EmergencyStopAction(
                    "action_reduce_market_data", SystemComponent.MARKET_DATA, "pause", 2
                ),
            ],
            estimated_duration_seconds=30,
        )

        # 重要レベル
        critical_plan = EmergencyStopPlan(
            plan_id="plan_critical",
            level=EmergencyStopLevel.CRITICAL,
            name="Critical Level Stop",
            description="主要機能の停止",
            actions=[
                EmergencyStopAction(
                    "action_stop_trading", SystemComponent.TRADING_ENGINE, "stop", 1
                ),
                EmergencyStopAction(
                    "action_stop_orders", SystemComponent.ORDER_MANAGER, "stop", 2
                ),
                EmergencyStopAction(
                    "action_isolate_portfolio",
                    SystemComponent.PORTFOLIO_MANAGER,
                    "isolate",
                    3,
                ),
            ],
            estimated_duration_seconds=60,
        )

        # 重大レベル
        severe_plan = EmergencyStopPlan(
            plan_id="plan_severe",
            level=EmergencyStopLevel.SEVERE,
            name="Severe Level Stop",
            description="全機能の停止",
            actions=[
                EmergencyStopAction(
                    "action_stop_all_trading", SystemComponent.TRADING_ENGINE, "stop", 1
                ),
                EmergencyStopAction(
                    "action_stop_market_data", SystemComponent.MARKET_DATA, "stop", 2
                ),
                EmergencyStopAction(
                    "action_stop_risk_manager", SystemComponent.RISK_MANAGER, "stop", 3
                ),
                EmergencyStopAction(
                    "action_stop_portfolio",
                    SystemComponent.PORTFOLIO_MANAGER,
                    "stop",
                    4,
                ),
                EmergencyStopAction(
                    "action_stop_orders", SystemComponent.ORDER_MANAGER, "stop", 5
                ),
                EmergencyStopAction(
                    "action_isolate_database", SystemComponent.DATABASE, "isolate", 6
                ),
            ],
            estimated_duration_seconds=120,
        )

        # 壊滅的レベル
        catastrophic_plan = EmergencyStopPlan(
            plan_id="plan_catastrophic",
            level=EmergencyStopLevel.CATASTROPHIC,
            name="Catastrophic Level Stop",
            description="完全システム停止",
            actions=[
                EmergencyStopAction(
                    "action_shutdown_all", SystemComponent.TRADING_ENGINE, "shutdown", 1
                ),
                EmergencyStopAction(
                    "action_shutdown_market_data",
                    SystemComponent.MARKET_DATA,
                    "shutdown",
                    2,
                ),
                EmergencyStopAction(
                    "action_shutdown_risk", SystemComponent.RISK_MANAGER, "shutdown", 3
                ),
                EmergencyStopAction(
                    "action_shutdown_portfolio",
                    SystemComponent.PORTFOLIO_MANAGER,
                    "shutdown",
                    4,
                ),
                EmergencyStopAction(
                    "action_shutdown_orders",
                    SystemComponent.ORDER_MANAGER,
                    "shutdown",
                    5,
                ),
                EmergencyStopAction(
                    "action_shutdown_database", SystemComponent.DATABASE, "shutdown", 6
                ),
                EmergencyStopAction(
                    "action_shutdown_network", SystemComponent.NETWORK, "shutdown", 7
                ),
                EmergencyStopAction(
                    "action_shutdown_monitoring",
                    SystemComponent.MONITORING,
                    "shutdown",
                    8,
                ),
            ],
            estimated_duration_seconds=300,
        )

        self.stop_plans = {
            EmergencyStopLevel.WARNING: warning_plan,
            EmergencyStopLevel.CRITICAL: critical_plan,
            EmergencyStopLevel.SEVERE: severe_plan,
            EmergencyStopLevel.CATASTROPHIC: catastrophic_plan,
        }

    def _initialize_component_states(self) -> None:
        """コンポーネント状態初期化"""
        for component in SystemComponent:
            self.component_states[component] = "running"

    async def trigger_emergency_stop(
        self,
        level: EmergencyStopLevel,
        trigger: EmergencyStopTrigger,
        reason: str,
        triggered_by: str = "system",
    ) -> EmergencyStopEvent:
        """
        緊急停止トリガー

        Args:
            level: 停止レベル
            trigger: トリガー
            reason: 理由
            triggered_by: トリガー実行者

        Returns:
            EmergencyStopEvent: 緊急停止イベント
        """
        if self.is_emergency_stop_active:
            self.logger.warning("Emergency stop already active, ignoring new trigger")
            # 既存のイベントを返す
            return self.stop_events[-1] if self.stop_events else None

        self.is_emergency_stop_active = True
        self.current_stop_level = level
        self.stop_timestamp = datetime.now()

        # イベント作成
        event = EmergencyStopEvent(
            event_id=f"EMERGENCY_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            trigger=trigger,
            level=level,
            reason=reason,
            triggered_by=triggered_by,
        )

        self.stop_events.append(event)

        # 履歴制限（最新50件）
        if len(self.stop_events) > 50:
            self.stop_events = self.stop_events[-50:]

        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {level.value} - {reason}")

        # 停止プラン実行
        success = await self._execute_stop_plan(event)

        if success:
            event.status = "completed"
        else:
            event.status = "failed"

        # コールバック実行
        for callback in self.stop_callbacks:
            try:
                asyncio.create_task(callback(event))
            except Exception as e:
                self.logger.error(f"Emergency stop callback error: {e}")

        return event

    async def _execute_stop_plan(self, event: EmergencyStopEvent) -> bool:
        """
        停止プラン実行

        Args:
            event: 緊急停止イベント

        Returns:
            bool: 実行成功フラグ
        """
        if event.level not in self.stop_plans:
            self.logger.error(f"No stop plan found for level {event.level.value}")
            return False

        plan = self.stop_plans[event.level]

        self.logger.critical(f"Executing emergency stop plan: {plan.name}")

        # アクションを優先度順にソート
        sorted_actions = sorted(plan.actions, key=lambda a: a.priority)

        success_count = 0

        for action in sorted_actions:
            try:
                self.logger.warning(f"Executing emergency action: {action.action_id}")

                # アクション実行
                success = await self._execute_stop_action(action)

                if success:
                    event.actions_taken.append(f"SUCCESS: {action.action_id}")
                    success_count += 1

                    # コンポーネント状態更新
                    self.component_states[action.component] = action.action_type

                else:
                    event.actions_taken.append(f"FAILED: {action.action_id}")
                    self.logger.error(f"Emergency action failed: {action.action_id}")

            except Exception as e:
                event.actions_taken.append(f"ERROR: {action.action_id} - {str(e)}")
                self.logger.error(f"Emergency action error: {action.action_id} - {e}")

        # 全体の成功判定
        success_rate = success_count / len(sorted_actions) if sorted_actions else 0
        overall_success = success_rate >= 0.8  # 80%以上成功で全体成功

        if overall_success:
            self.logger.critical("Emergency stop plan completed successfully")
        else:
            self.logger.critical(
                f"Emergency stop plan completed with failures: {success_count}/{len(sorted_actions)} actions succeeded"
            )

        return overall_success

    async def _execute_stop_action(self, action: EmergencyStopAction) -> bool:
        """
        停止アクション実行

        Args:
            action: 停止アクション

        Returns:
            bool: 実行成功フラグ
        """
        try:
            # 実際の実装では各コンポーネントの停止ロジックを呼び出す
            # ここではシミュレーション

            if action.action_type == "pause":
                # 一時停止
                await asyncio.sleep(0.5)
                self.logger.warning(f"Component {action.component.value} paused")

            elif action.action_type == "stop":
                # 停止
                await asyncio.sleep(1.0)
                self.logger.warning(f"Component {action.component.value} stopped")

            elif action.action_type == "isolate":
                # 隔離
                await asyncio.sleep(0.8)
                self.logger.warning(f"Component {action.component.value} isolated")

            elif action.action_type == "shutdown":
                # 完全停止
                await asyncio.sleep(2.0)
                self.logger.warning(f"Component {action.component.value} shut down")

            # 稀に失敗をシミュレート
            import random

            if random.random() < 0.05:  # 5%の確率
                raise Exception(f"Simulated failure in {action.action_id}")

            return True

        except Exception as e:
            self.logger.error(f"Stop action failed: {action.action_id} - {e}")
            return False

    async def initiate_recovery(
        self, event_id: str, recovered_by: str = "system"
    ) -> bool:
        """
        復旧開始

        Args:
            event_id: 緊急停止イベントID
            recovered_by: 復旧実行者

        Returns:
            bool: 復旧開始成功フラグ
        """
        # イベント検索
        event = None
        for e in self.stop_events:
            if e.event_id == event_id:
                event = e
                break

        if not event:
            self.logger.error(f"Emergency stop event not found: {event_id}")
            return False

        if not self.is_emergency_stop_active:
            self.logger.warning("No active emergency stop to recover from")
            return False

        self.logger.info(f"Initiating recovery from emergency stop: {event_id}")

        # 復旧実行
        success = await self._execute_recovery_plan(event)

        if success:
            self.is_emergency_stop_active = False
            self.current_stop_level = None
            self.stop_timestamp = None
            event.status = "recovered"

            self.logger.info("Recovery completed successfully")

            # コールバック実行
            for callback in self.recovery_callbacks:
                try:
                    asyncio.create_task(callback(event))
                except Exception as e:
                    self.logger.error(f"Recovery callback error: {e}")

        else:
            self.logger.error("Recovery failed")
            event.status = "recovery_failed"

        return success

    async def _execute_recovery_plan(self, event: EmergencyStopEvent) -> bool:
        """
        復旧プラン実行

        Args:
            event: 緊急停止イベント

        Returns:
            bool: 実行成功フラグ
        """
        plan = self.stop_plans.get(event.level)
        if not plan:
            return False

        self.logger.info(f"Executing recovery plan for level: {event.level.value}")

        # アクションを逆順で復旧
        recovery_actions = []
        for action in reversed(plan.actions):
            if action.rollback_action:
                recovery_actions.append(action)

        success_count = 0

        for action in recovery_actions:
            try:
                self.logger.info(f"Executing recovery action: {action.rollback_action}")

                # ロールバックアクション実行
                success = await self._execute_recovery_action(action)

                if success:
                    # コンポーネント状態更新
                    self.component_states[action.component] = "running"
                    success_count += 1
                else:
                    self.logger.error(
                        f"Recovery action failed: {action.rollback_action}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Recovery action error: {action.rollback_action} - {e}"
                )

        # 全体の成功判定
        success_rate = success_count / len(recovery_actions) if recovery_actions else 1
        return success_rate >= 0.8

    async def _execute_recovery_action(self, action: EmergencyStopAction) -> bool:
        """
        復旧アクション実行

        Args:
            action: 停止アクション

        Returns:
            bool: 実行成功フラグ
        """
        try:
            # 実際の実装では各コンポーネントの復旧ロジックを呼び出す
            # ここではシミュレーション

            await asyncio.sleep(1.0)  # 復旧時間シミュレーション

            self.logger.info(f"Component {action.component.value} recovered")
            return True

        except Exception as e:
            self.logger.error(f"Recovery action failed: {action.action_id} - {e}")
            return False

    def add_auto_trigger_condition(
        self,
        condition_id: str,
        condition_func: Callable[[], bool],
        trigger_level: EmergencyStopLevel,
        reason: str,
    ) -> None:
        """
        自動トリガー条件追加

        Args:
            condition_id: 条件ID
            condition_func: 条件評価関数
            trigger_level: トリガーレベル
            reason: 理由
        """
        self.auto_trigger_conditions[condition_id] = {
            "func": condition_func,
            "level": trigger_level,
            "reason": reason,
        }

        self.logger.info(f"Auto trigger condition added: {condition_id}")

    def remove_auto_trigger_condition(self, condition_id: str) -> None:
        """
        自動トリガー条件削除

        Args:
            condition_id: 条件ID
        """
        if condition_id in self.auto_trigger_conditions:
            del self.auto_trigger_conditions[condition_id]
            self.logger.info(f"Auto trigger condition removed: {condition_id}")

    def check_auto_trigger_conditions(self) -> List[Dict[str, Any]]:
        """
        自動トリガー条件チェック

        Returns:
            List[Dict[str, Any]]: トリガーされた条件リスト
        """
        triggered_conditions = []

        for condition_id, condition_data in self.auto_trigger_conditions.items():
            try:
                if condition_data["func"]():
                    triggered_conditions.append(
                        {
                            "condition_id": condition_id,
                            "level": condition_data["level"],
                            "reason": condition_data["reason"],
                        }
                    )
            except Exception as e:
                self.logger.error(
                    f"Auto trigger condition check error for {condition_id}: {e}"
                )

        return triggered_conditions

    def get_system_status(self) -> Dict[str, Any]:
        """
        システム状態取得

        Returns:
            Dict[str, Any]: システム状態
        """
        return {
            "system_name": self.system_name,
            "emergency_stop_active": self.is_emergency_stop_active,
            "current_stop_level": self.current_stop_level.value
            if self.current_stop_level
            else None,
            "stop_timestamp": self.stop_timestamp.isoformat()
            if self.stop_timestamp
            else None,
            "component_states": {
                comp.value: state for comp, state in self.component_states.items()
            },
            "active_auto_conditions": len(self.auto_trigger_conditions),
            "total_stop_events": len(self.stop_events),
        }

    def get_stop_events(self, limit: Optional[int] = None) -> List[EmergencyStopEvent]:
        """
        停止イベント取得

        Args:
            limit: 取得件数制限

        Returns:
            List[EmergencyStopEvent]: 停止イベントリスト
        """
        events = self.stop_events
        if limit:
            events = events[-limit:]
        return events

    def cancel_emergency_stop(
        self, event_id: str, cancelled_by: str = "system"
    ) -> bool:
        """
        緊急停止キャンセル

        Args:
            event_id: イベントID
            cancelled_by: キャンセル実行者

        Returns:
            bool: キャンセル成功フラグ
        """
        for event in self.stop_events:
            if event.event_id == event_id and event.status == "active":
                event.status = "cancelled"
                event.actions_taken.append(
                    f"CANCELLED by {cancelled_by} at {datetime.now().isoformat()}"
                )

                self.is_emergency_stop_active = False
                self.current_stop_level = None
                self.stop_timestamp = None

                self.logger.warning(f"Emergency stop cancelled: {event_id}")
                return True

        return False

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Emergency stop monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Emergency stop monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # 自動トリガー条件チェック
                triggered_conditions = self.check_auto_trigger_conditions()

                for condition in triggered_conditions:
                    asyncio.run(
                        self.trigger_emergency_stop(
                            level=condition["level"],
                            trigger=EmergencyStopTrigger.SYSTEM_FAILURE,
                            reason=f"Auto trigger: {condition['condition_id']} - {condition['reason']}",
                            triggered_by="auto_monitor",
                        )
                    )

                time.sleep(10)  # 10秒間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def add_stop_callback(
        self, callback: Callable[[EmergencyStopEvent], Awaitable[None]]
    ) -> None:
        """
        停止コールバック追加

        Args:
            callback: コールバック関数
        """
        self.stop_callbacks.append(callback)

    def add_recovery_callback(
        self, callback: Callable[[EmergencyStopEvent], Awaitable[None]]
    ) -> None:
        """
        復旧コールバック追加

        Args:
            callback: コールバック関数
        """
        self.recovery_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "system_name": self.system_name,
            "is_emergency_stop_active": self.is_emergency_stop_active,
            "current_stop_level": self.current_stop_level.value
            if self.current_stop_level
            else None,
            "stop_timestamp": self.stop_timestamp.isoformat()
            if self.stop_timestamp
            else None,
            "component_states": {
                comp.value: state for comp, state in self.component_states.items()
            },
            "stop_events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "trigger": e.trigger.value,
                    "level": e.level.value,
                    "reason": e.reason,
                    "triggered_by": e.triggered_by,
                    "actions_taken": e.actions_taken,
                    "status": e.status,
                }
                for e in self.stop_events[-20:]  # 最新20件
            ],
            "auto_trigger_conditions": {
                cond_id: {
                    "level": cond_data["level"].value,
                    "reason": cond_data["reason"],
                }
                for cond_id, cond_data in self.auto_trigger_conditions.items()
            },
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Emergency stop state saved to {filepath}")

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

            self.system_name = state["system_name"]
            self.is_emergency_stop_active = state["is_emergency_stop_active"]
            self.current_stop_level = (
                EmergencyStopLevel(state["current_stop_level"])
                if state["current_stop_level"]
                else None
            )
            self.stop_timestamp = (
                datetime.fromisoformat(state["stop_timestamp"])
                if state["stop_timestamp"]
                else None
            )

            # コンポーネント状態復元
            self.component_states = {}
            for comp_str, comp_state in state.get("component_states", {}).items():
                self.component_states[SystemComponent(comp_str)] = comp_state

            # 停止イベント復元
            self.stop_events = []
            for e_data in state.get("stop_events", []):
                event = EmergencyStopEvent(
                    event_id=e_data["event_id"],
                    timestamp=datetime.fromisoformat(e_data["timestamp"]),
                    trigger=EmergencyStopTrigger(e_data["trigger"]),
                    level=EmergencyStopLevel(e_data["level"]),
                    reason=e_data["reason"],
                    triggered_by=e_data["triggered_by"],
                    actions_taken=e_data["actions_taken"],
                    status=e_data["status"],
                )
                self.stop_events.append(event)

            # 自動トリガー条件復元（関数は復元できないのでレベルと理由のみ）
            self.auto_trigger_conditions = {}
            for cond_id, cond_data in state.get("auto_trigger_conditions", {}).items():
                # 実際の運用では条件関数を再登録する必要がある
                self.auto_trigger_conditions[cond_id] = {
                    "level": EmergencyStopLevel(cond_data["level"]),
                    "reason": cond_data["reason"],
                    "func": lambda: False,  # ダミー関数
                }

            self.logger.info(f"Emergency stop state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load emergency stop state: {e}")
            return False
