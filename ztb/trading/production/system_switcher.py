"""
V433 Phase 5: Parallel Running Layer - System Switcher

既存システムとV433システム間の動的切り替えを管理する。
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)


class SystemType(Enum):
    """システムタイプ"""

    LEGACY = "legacy"  # 既存システム
    V433 = "v433"  # V433システム


class SwitchMode(Enum):
    """切り替えモード"""

    MANUAL = "manual"  # 手動切り替え
    AUTOMATIC = "automatic"  # 自動切り替え
    GRADUAL = "gradual"  # 段階的切り替え


@dataclass
class SwitchCondition:
    """切り替え条件"""

    metric_name: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    consecutive_periods: int = 1  # 連続期間
    cooldown_minutes: int = 5  # クールダウン時間（分）


@dataclass
class SwitchRule:
    """切り替えルール"""

    rule_id: str
    name: str
    description: str
    from_system: SystemType
    to_system: SystemType
    conditions: List[SwitchCondition]
    priority: int = 1
    enabled: bool = True


@dataclass
class SwitchEvent:
    """切り替えイベント"""

    event_id: str
    timestamp: datetime
    from_system: SystemType
    to_system: SystemType
    trigger_rule: Optional[str]
    reason: str
    success: bool
    execution_time_ms: int
    rollback_time_ms: Optional[int] = None


@dataclass
class SystemHealth:
    """システム健全性"""

    system_type: SystemType
    is_healthy: bool
    last_check: datetime
    response_time_ms: int
    error_count: int
    metrics: Dict[str, float] = field(default_factory=dict)


class SystemSwitcher:
    """
    システムスイッチャー

    既存システムとV433システム間の動的切り替えを管理し、
    自動切り替えルールに基づいて最適なシステムを選択する。
    """

    def __init__(
        self,
        initial_system: SystemType = SystemType.LEGACY,
        switch_mode: SwitchMode = SwitchMode.MANUAL,
    ):
        """
        初期化

        Args:
            initial_system: 初期システム
            switch_mode: 切り替えモード
        """
        self.current_system = initial_system
        self.switch_mode = switch_mode

        # 切り替えルール
        self.switch_rules: Dict[str, SwitchRule] = {}
        self.active_rule: Optional[str] = None

        # システム健全性
        self.system_health: Dict[SystemType, SystemHealth] = {
            SystemType.LEGACY: SystemHealth(
                system_type=SystemType.LEGACY,
                is_healthy=True,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=0,
            ),
            SystemType.V433: SystemHealth(
                system_type=SystemType.V433,
                is_healthy=True,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=0,
            ),
        }

        # 切り替え履歴
        self.switch_history: List[SwitchEvent] = []

        # メトリクス監視
        self.metrics_buffer: Dict[str, List[float]] = {}
        self.condition_states: Dict[
            str, Dict[str, int]
        ] = {}  # rule_id -> condition_index -> consecutive_count

        # クールダウン管理
        self.last_switch_time: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None

        # コールバック
        self.switch_callbacks: List[Callable[[SwitchEvent], Awaitable[None]]] = []
        self.health_callbacks: List[Callable[[SystemType, bool], Awaitable[None]]] = []

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"System Switcher initialized. Current system: {initial_system.value}, Mode: {switch_mode.value}"
        )

    def add_switch_rule(self, rule: SwitchRule) -> None:
        """
        切り替えルール追加

        Args:
            rule: 切り替えルール
        """
        self.switch_rules[rule.rule_id] = rule
        self.condition_states[rule.rule_id] = dict.fromkeys(range(len(rule.conditions)), 0)

        # メトリクスバッファ初期化
        for condition in rule.conditions:
            if condition.metric_name not in self.metrics_buffer:
                self.metrics_buffer[condition.metric_name] = []

        self.logger.info(f"Switch rule added: {rule.name} ({rule.rule_id})")

    def remove_switch_rule(self, rule_id: str) -> bool:
        """
        切り替えルール削除

        Args:
            rule_id: ルールID

        Returns:
            bool: 削除成功フラグ
        """
        if rule_id in self.switch_rules:
            del self.switch_rules[rule_id]
            if rule_id in self.condition_states:
                del self.condition_states[rule_id]
            self.logger.info(f"Switch rule removed: {rule_id}")
            return True

        return False

    def enable_rule(self, rule_id: str) -> bool:
        """
        ルール有効化

        Args:
            rule_id: ルールID

        Returns:
            bool: 有効化成功フラグ
        """
        if rule_id in self.switch_rules:
            self.switch_rules[rule_id].enabled = True
            self.logger.info(f"Switch rule enabled: {rule_id}")
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        ルール無効化

        Args:
            rule_id: ルールID

        Returns:
            bool: 無効化成功フラグ
        """
        if rule_id in self.switch_rules:
            self.switch_rules[rule_id].enabled = False
            self.logger.info(f"Switch rule disabled: {rule_id}")
            return True
        return False

    async def switch_system(
        self,
        target_system: SystemType,
        reason: str = "Manual switch",
        force: bool = False,
    ) -> bool:
        """
        システム切り替え

        Args:
            target_system: 対象システム
            reason: 切り替え理由
            force: 強制切り替えフラグ

        Returns:
            bool: 切り替え成功フラグ
        """
        if self.current_system == target_system:
            self.logger.info(f"Already on target system: {target_system.value}")
            return True

        # クールダウンチェック
        if not force and self._is_in_cooldown():
            self.logger.warning("Switch request denied: in cooldown period")
            return False

        # 健全性チェック
        if not force and not self.system_health[target_system].is_healthy:
            self.logger.warning(
                f"Switch request denied: target system {target_system.value} is unhealthy"
            )
            return False

        start_time = time.time()

        try:
            # 切り替え実行
            success = await self._execute_switch(target_system)

            execution_time = int((time.time() - start_time) * 1000)

            # イベント記録
            event = SwitchEvent(
                event_id=f"SW_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                from_system=self.current_system,
                to_system=target_system,
                trigger_rule=self.active_rule,
                reason=reason,
                success=success,
                execution_time_ms=execution_time,
            )

            self.switch_history.append(event)

            if success:
                self.current_system = target_system
                self.last_switch_time = datetime.now()
                self.active_rule = None

                # クールダウン設定（該当ルールがある場合）
                if self.active_rule and self.active_rule in self.switch_rules:
                    rule = self.switch_rules[self.active_rule]
                    max_cooldown = max(
                        (c.cooldown_minutes for c in rule.conditions), default=5
                    )
                    self.cooldown_until = datetime.now() + timedelta(
                        minutes=max_cooldown
                    )

            # コールバック実行
            for callback in self.switch_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    self.logger.error(f"Switch callback error: {e}")

            self.logger.info(
                f"System switch {'successful' if success else 'failed'}: {self.current_system.value} -> {target_system.value}"
            )

            return success

        except Exception as e:
            self.logger.error(f"Switch execution error: {e}")
            return False

    async def _execute_switch(self, target_system: SystemType) -> bool:
        """
        切り替え実行

        Args:
            target_system: 対象システム

        Returns:
            bool: 実行成功フラグ
        """
        try:
            # 実際の切り替えロジックはシステム固有
            # ここでは簡易的な実装

            if target_system == SystemType.V433:
                # V433システムへの切り替え
                success = await self._switch_to_v433()
            else:
                # 既存システムへの切り替え
                success = await self._switch_to_legacy()

            return success

        except Exception as e:
            self.logger.error(f"Switch execution failed: {e}")
            return False

    async def _switch_to_v433(self) -> bool:
        """
        V433システムへの切り替え

        Returns:
            bool: 切り替え成功フラグ
        """
        # V433システムの準備確認
        # 実際の実装ではV433システムの初期化を行う
        await asyncio.sleep(0.1)  # シミュレーション
        return True

    async def _switch_to_legacy(self) -> bool:
        """
        既存システムへの切り替え

        Returns:
            bool: 切り替え成功フラグ
        """
        # 既存システムの準備確認
        # 実際の実装では既存システムへの切り替えを行う
        await asyncio.sleep(0.1)  # シミュレーション
        return True

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """
        メトリクス更新

        Args:
            metrics: メトリクスデータ
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_buffer:
                self.metrics_buffer[metric_name] = []

            # バッファ更新（最新100件保持）
            self.metrics_buffer[metric_name].append(value)
            if len(self.metrics_buffer[metric_name]) > 100:
                self.metrics_buffer[metric_name] = self.metrics_buffer[metric_name][
                    -100:
                ]

        # 自動切り替えモードの場合、ルール評価
        if self.switch_mode == SwitchMode.AUTOMATIC:
            asyncio.create_task(self._evaluate_switch_rules())

    async def _evaluate_switch_rules(self) -> None:
        """切り替えルール評価"""
        if self._is_in_cooldown():
            return

        # 有効なルールを取得（優先度順）
        active_rules = sorted(
            [rule for rule in self.switch_rules.values() if rule.enabled],
            key=lambda r: r.priority,
            reverse=True,
        )

        for rule in active_rules:
            if await self._evaluate_rule(rule):
                # ルール条件を満たした場合、切り替え実行
                await self.switch_system(
                    target_system=rule.to_system,
                    reason=f"Auto-switch triggered by rule: {rule.name}",
                    force=False,
                )
                self.active_rule = rule.rule_id
                break

    async def _evaluate_rule(self, rule: SwitchRule) -> bool:
        """
        ルール評価

        Args:
            rule: 切り替えルール

        Returns:
            bool: ルール条件满足フラグ
        """
        if rule.from_system != self.current_system:
            return False

        all_conditions_met = True

        for i, condition in enumerate(rule.conditions):
            if not self._evaluate_condition(condition):
                self.condition_states[rule.rule_id][i] = 0
                all_conditions_met = False
            else:
                self.condition_states[rule.rule_id][i] += 1
                # 連続期間チェック
                if (
                    self.condition_states[rule.rule_id][i]
                    < condition.consecutive_periods
                ):
                    all_conditions_met = False

        return all_conditions_met

    def _evaluate_condition(self, condition: SwitchCondition) -> bool:
        """
        条件評価

        Args:
            condition: 切り替え条件

        Returns:
            bool: 条件满足フラグ
        """
        if condition.metric_name not in self.metrics_buffer:
            return False

        buffer = self.metrics_buffer[condition.metric_name]
        if not buffer:
            return False

        # 最新値取得
        latest_value = buffer[-1]

        # 演算子評価
        if condition.operator == ">":
            return latest_value > condition.threshold
        elif condition.operator == "<":
            return latest_value < condition.threshold
        elif condition.operator == ">=":
            return latest_value >= condition.threshold
        elif condition.operator == "<=":
            return latest_value <= condition.threshold
        elif condition.operator == "==":
            return abs(latest_value - condition.threshold) < 1e-6
        elif condition.operator == "!=":
            return abs(latest_value - condition.threshold) >= 1e-6
        else:
            self.logger.warning(f"Unknown operator: {condition.operator}")
            return False

    def update_system_health(
        self,
        system_type: SystemType,
        is_healthy: bool,
        response_time_ms: int,
        error_count: int = 0,
    ) -> None:
        """
        システム健全性更新

        Args:
            system_type: システムタイプ
            is_healthy: 健全性フラグ
            response_time_ms: 応答時間
            error_count: エラー数
        """
        if system_type not in self.system_health:
            return

        old_health = self.system_health[system_type].is_healthy
        self.system_health[system_type].is_healthy = is_healthy
        self.system_health[system_type].last_check = datetime.now()
        self.system_health[system_type].response_time_ms = response_time_ms
        self.system_health[system_type].error_count = error_count

        # 健全性変化時のコールバック
        if old_health != is_healthy:
            for callback in self.health_callbacks:
                try:
                    asyncio.create_task(callback(system_type, is_healthy))
                except Exception as e:
                    self.logger.error(f"Health callback error: {e}")

        # 不健全なシステムが現在使用中の場合、自動切り替え
        if (
            not is_healthy
            and system_type == self.current_system
            and self.switch_mode == SwitchMode.AUTOMATIC
        ):
            alternative_system = (
                SystemType.LEGACY if system_type == SystemType.V433 else SystemType.V433
            )
            if self.system_health[alternative_system].is_healthy:
                asyncio.create_task(
                    self.switch_system(
                        target_system=alternative_system,
                        reason=f"Emergency switch due to {system_type.value} system failure",
                        force=True,
                    )
                )

    def _is_in_cooldown(self) -> bool:
        """クールダウン中かどうか"""
        return self.cooldown_until is not None and datetime.now() < self.cooldown_until

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("System monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # システム健全性チェック
                for system_type in SystemType:
                    self._check_system_health(system_type)

                time.sleep(30)  # 30秒間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def _check_system_health(self, system_type: SystemType) -> None:
        """
        システム健全性チェック

        Args:
            system_type: システムタイプ
        """
        try:
            # 簡易的な健全性チェック
            # 実際の実装では各システムのヘルスチェックエンドポイントを呼び出す
            start_time = time.time()

            # シミュレーション：ランダムで不健全になる場合がある
            import random

            is_healthy = random.random() > 0.05  # 95%の確率で健全

            response_time = int((time.time() - start_time) * 1000)
            error_count = 0 if is_healthy else 1

            self.update_system_health(
                system_type, is_healthy, response_time, error_count
            )

        except Exception as e:
            self.logger.error(f"Health check failed for {system_type.value}: {e}")
            self.update_system_health(system_type, False, 9999, 1)

    def get_current_system(self) -> SystemType:
        """
        現在のシステム取得

        Returns:
            SystemType: 現在のシステム
        """
        return self.current_system

    def get_system_health(self, system_type: SystemType) -> Optional[SystemHealth]:
        """
        システム健全性取得

        Args:
            system_type: システムタイプ

        Returns:
            Optional[SystemHealth]: システム健全性
        """
        return self.system_health.get(system_type)

    def get_switch_history(self, limit: Optional[int] = None) -> List[SwitchEvent]:
        """
        切り替え履歴取得

        Args:
            limit: 取得件数制限

        Returns:
            List[SwitchEvent]: 切り替え履歴
        """
        history = self.switch_history
        if limit:
            history = history[-limit:]
        return history.copy()

    def get_active_rules(self) -> List[SwitchRule]:
        """
        有効なルール取得

        Returns:
            List[SwitchRule]: 有効な切り替えルール
        """
        return [rule for rule in self.switch_rules.values() if rule.enabled]

    def add_switch_callback(
        self, callback: Callable[[SwitchEvent], Awaitable[None]]
    ) -> None:
        """
        切り替えコールバック追加

        Args:
            callback: コールバック関数
        """
        self.switch_callbacks.append(callback)

    def add_health_callback(
        self, callback: Callable[[SystemType, bool], Awaitable[None]]
    ) -> None:
        """
        健全性コールバック追加

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
            "current_system": self.current_system.value,
            "switch_mode": self.switch_mode.value,
            "last_switch_time": self.last_switch_time.isoformat()
            if self.last_switch_time
            else None,
            "cooldown_until": self.cooldown_until.isoformat()
            if self.cooldown_until
            else None,
            "switch_rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "from_system": rule.from_system.value,
                    "to_system": rule.to_system.value,
                    "conditions": [
                        {
                            "metric_name": cond.metric_name,
                            "operator": cond.operator,
                            "threshold": cond.threshold,
                            "consecutive_periods": cond.consecutive_periods,
                            "cooldown_minutes": cond.cooldown_minutes,
                        }
                        for cond in rule.conditions
                    ],
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                }
                for rule in self.switch_rules.values()
            ],
            "switch_history": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "from_system": event.from_system.value,
                    "to_system": event.to_system.value,
                    "trigger_rule": event.trigger_rule,
                    "reason": event.reason,
                    "success": event.success,
                    "execution_time_ms": event.execution_time_ms,
                    "rollback_time_ms": event.rollback_time_ms,
                }
                for event in self.switch_history[-100:]  # 最新100件
            ],
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Switcher state saved to {filepath}")

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

            self.current_system = SystemType(state["current_system"])
            self.switch_mode = SwitchMode(state["switch_mode"])
            self.last_switch_time = (
                datetime.fromisoformat(state["last_switch_time"])
                if state["last_switch_time"]
                else None
            )
            self.cooldown_until = (
                datetime.fromisoformat(state["cooldown_until"])
                if state["cooldown_until"]
                else None
            )

            # ルール復元
            self.switch_rules = {}
            for rule_data in state.get("switch_rules", []):
                conditions = [
                    SwitchCondition(
                        metric_name=cond["metric_name"],
                        operator=cond["operator"],
                        threshold=cond["threshold"],
                        consecutive_periods=cond.get("consecutive_periods", 1),
                        cooldown_minutes=cond.get("cooldown_minutes", 5),
                    )
                    for cond in rule_data["conditions"]
                ]

                rule = SwitchRule(
                    rule_id=rule_data["rule_id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    from_system=SystemType(rule_data["from_system"]),
                    to_system=SystemType(rule_data["to_system"]),
                    conditions=conditions,
                    priority=rule_data.get("priority", 1),
                    enabled=rule_data.get("enabled", True),
                )

                self.switch_rules[rule.rule_id] = rule
                self.condition_states[rule.rule_id] = dict.fromkeys(range(len(conditions)), 0)

            # 履歴復元
            self.switch_history = []
            for event_data in state.get("switch_history", []):
                event = SwitchEvent(
                    event_id=event_data["event_id"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    from_system=SystemType(event_data["from_system"]),
                    to_system=SystemType(event_data["to_system"]),
                    trigger_rule=event_data["trigger_rule"],
                    reason=event_data["reason"],
                    success=event_data["success"],
                    execution_time_ms=event_data["execution_time_ms"],
                    rollback_time_ms=event_data.get("rollback_time_ms"),
                )
                self.switch_history.append(event)

            self.logger.info(f"Switcher state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load switcher state: {e}")
            return False
