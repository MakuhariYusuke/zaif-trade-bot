"""
Fallback Manager
フォールバックマネージャー
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..monitoring.safety import SafetyManager
from .types import FallbackAction, SafetyLevel

logger = logging.getLogger(__name__)


class FallbackMode(Enum):
    """フォールバックモード"""

    CONSERVATIVE = "conservative"  # 保守的な取引（低リスク）
    CIRCUIT_BREAKER = "circuit_breaker"  # 取引停止
    GRADUAL_DEGRADATION = "gradual_degradation"  # 段階的機能低下
    EMERGENCY_SHUTDOWN = "emergency_shutdown"  # 緊急停止


@dataclass
class FallbackConfig:
    """フォールバック設定"""

    # フォールバックモード設定
    enable_conservative_mode: bool = True
    enable_circuit_breaker: bool = True
    enable_gradual_degradation: bool = True

    # 保守的モード設定
    conservative_max_position_size: float = 0.1  # 最大ポジションサイズ（通常の10%）
    conservative_max_leverage: float = 1.0  # レバレッジ制限
    conservative_trade_frequency_limit: int = 5  # 1時間あたりの最大取引回数

    # サーキットブレーカー設定
    circuit_breaker_threshold: float = 0.5  # アクティベーション閾値
    circuit_breaker_cooldown_minutes: int = 60  # クールダウン期間
    circuit_breaker_auto_reset: bool = True

    # 段階的劣化設定
    degradation_steps: int = 5  # 劣化ステップ数
    degradation_reduction_factor: float = 0.8  # 各ステップでの削減率

    # タイムアウト設定
    fallback_timeout_seconds: int = 3600  # フォールバックの最大継続時間
    recovery_check_interval_seconds: int = 60  # 回復チェック間隔


@dataclass
class FallbackState:
    """フォールバック状態"""

    mode: FallbackMode
    activated_at: datetime
    reason: str
    severity: SafetyLevel
    active_actions: List[FallbackAction] = field(default_factory=list)
    recovery_attempts: int = 0
    last_recovery_check: Optional[datetime] = None
    expected_recovery_time: Optional[datetime] = None


class FallbackManager:
    """フォールバックマネージャー"""

    def __init__(
        self, safety_manager: SafetyManager, config: Optional[FallbackConfig] = None
    ):
        self.safety_manager = safety_manager
        self.config = config or FallbackConfig()

        self.current_fallback: Optional[FallbackState] = None
        self.fallback_history: List[FallbackState] = []
        self.is_active = False

        # コールバック
        self.fallback_activated_callbacks: List[Callable[[FallbackState], None]] = []
        self.fallback_deactivated_callbacks: List[Callable[[FallbackState], None]] = []
        self.recovery_attempt_callbacks: List[Callable[[FallbackState], None]] = []

        # スレッド管理
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        logger.info("FallbackManager initialized")

    def start_monitoring(self) -> bool:
        """監視を開始"""
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return True

            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_worker, daemon=True
            )
            self.monitor_thread.start()

            logger.info("Fallback monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start fallback monitoring: {e}")
            return False

    def stop_monitoring(self) -> None:
        """監視を停止"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Fallback monitoring stopped")

    def activate_fallback(
        self,
        mode: FallbackMode,
        reason: str,
        severity: SafetyLevel = SafetyLevel.WARNING,
    ) -> bool:
        """フォールバックをアクティベート"""
        try:
            if self.current_fallback:
                logger.warning(
                    f"Fallback already active: {self.current_fallback.mode.value}"
                )
                return False

            # フォールバック状態を作成
            fallback_state = FallbackState(
                mode=mode, activated_at=datetime.now(), reason=reason, severity=severity
            )

            # フォールバックアクションを実行
            actions = self._execute_fallback_actions(mode, severity)
            fallback_state.active_actions = actions

            # 期待回復時間を設定
            fallback_state.expected_recovery_time = (
                self._calculate_expected_recovery_time(mode)
            )

            self.current_fallback = fallback_state
            self.fallback_history.append(fallback_state)
            self.is_active = True

            # コールバックを実行
            self._trigger_fallback_activated_callbacks(fallback_state)

            logger.info(f"Fallback activated: {mode.value} - {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to activate fallback: {e}")
            return False

    def deactivate_fallback(self, reason: str = "Manual deactivation") -> bool:
        """フォールバックを非アクティベート"""
        try:
            if not self.current_fallback:
                logger.warning("No active fallback to deactivate")
                return False

            # フォールバックアクションを元に戻す
            self._rollback_fallback_actions(self.current_fallback.active_actions)

            fallback_state = self.current_fallback
            self.current_fallback = None
            self.is_active = False

            # コールバックを実行
            self._trigger_fallback_deactivated_callbacks(fallback_state)

            logger.info(f"Fallback deactivated: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to deactivate fallback: {e}")
            return False

    def attempt_recovery(self) -> bool:
        """回復を試行"""
        try:
            if not self.current_fallback:
                logger.warning("No active fallback to recover from")
                return False

            self.current_fallback.recovery_attempts += 1
            self.current_fallback.last_recovery_check = datetime.now()

            # 安全状態をチェック
            safety_status = self.safety_manager.get_safety_status()

            # 回復条件をチェック
            if self._check_recovery_conditions(safety_status):
                logger.info("Recovery conditions met, deactivating fallback")
                return self.deactivate_fallback("Recovery conditions met")
            else:
                # コールバックを実行
                self._trigger_recovery_attempt_callbacks(self.current_fallback)
                logger.info("Recovery conditions not met, keeping fallback active")
                return False

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False

    def get_fallback_status(self) -> Dict[str, Any]:
        """フォールバック状態を取得"""
        if not self.current_fallback:
            return {
                "active": False,
                "mode": None,
                "activated_at": None,
                "reason": None,
                "severity": None,
            }

        return {
            "active": True,
            "mode": self.current_fallback.mode.value,
            "activated_at": self.current_fallback.activated_at.isoformat(),
            "reason": self.current_fallback.reason,
            "severity": self.current_fallback.severity.value,
            "recovery_attempts": self.current_fallback.recovery_attempts,
            "expected_recovery_time": self.current_fallback.expected_recovery_time.isoformat()
            if self.current_fallback.expected_recovery_time
            else None,
            "active_actions_count": len(self.current_fallback.active_actions),
        }

    def _execute_fallback_actions(
        self, mode: FallbackMode, severity: SafetyLevel
    ) -> List[FallbackAction]:
        """フォールバックアクションを実行"""
        actions = []

        try:
            if mode == FallbackMode.CONSERVATIVE:
                actions.extend(self._activate_conservative_mode())
            elif mode == FallbackMode.CIRCUIT_BREAKER:
                actions.extend(self._activate_circuit_breaker())
            elif mode == FallbackMode.GRADUAL_DEGRADATION:
                actions.extend(self._activate_gradual_degradation(severity))
            elif mode == FallbackMode.EMERGENCY_SHUTDOWN:
                actions.extend(self._activate_emergency_shutdown())

            logger.info(
                f"Executed {len(actions)} fallback actions for mode: {mode.value}"
            )
            return actions

        except Exception as e:
            logger.error(f"Failed to execute fallback actions: {e}")
            return []

    def _activate_conservative_mode(self) -> List[FallbackAction]:
        """保守的モードをアクティベート"""
        actions = []

        try:
            # ポジションサイズ制限
            action = FallbackAction(
                action_id=f"conservative_position_{datetime.now().timestamp()}",
                action_type="limit_position_size",
                parameters={"max_size": self.config.conservative_max_position_size},
                description="Limit position size to conservative levels",
                rollback_action="restore_position_size",
            )
            actions.append(action)

            # レバレッジ制限
            action = FallbackAction(
                action_id=f"conservative_leverage_{datetime.now().timestamp()}",
                action_type="limit_leverage",
                parameters={"max_leverage": self.config.conservative_max_leverage},
                description="Limit leverage to conservative levels",
                rollback_action="restore_leverage",
            )
            actions.append(action)

            # 取引頻度制限
            action = FallbackAction(
                action_id=f"conservative_frequency_{datetime.now().timestamp()}",
                action_type="limit_trade_frequency",
                parameters={
                    "max_trades_per_hour": self.config.conservative_trade_frequency_limit
                },
                description="Limit trade frequency to conservative levels",
                rollback_action="restore_trade_frequency",
            )
            actions.append(action)

            # アクションを実行（実際の取引システムとの統合）
            for action in actions:
                self._execute_action(action)

            return actions

        except Exception as e:
            logger.error(f"Failed to activate conservative mode: {e}")
            return []

    def _activate_circuit_breaker(self) -> List[FallbackAction]:
        """サーキットブレーカーをアクティベート"""
        actions = []

        try:
            # 取引停止
            action = FallbackAction(
                action_id=f"circuit_breaker_stop_{datetime.now().timestamp()}",
                action_type="stop_trading",
                parameters={
                    "duration_minutes": self.config.circuit_breaker_cooldown_minutes
                },
                description="Stop all trading activities",
                rollback_action="resume_trading",
            )
            actions.append(action)

            # アクションを実行
            for action in actions:
                self._execute_action(action)

            return actions

        except Exception as e:
            logger.error(f"Failed to activate circuit breaker: {e}")
            return []

    def _activate_gradual_degradation(
        self, severity: SafetyLevel
    ) -> List[FallbackAction]:
        """段階的劣化をアクティベート"""
        actions = []

        try:
            # 劣化ステップ数を決定
            steps = min(
                self.config.degradation_steps,
                max(1, int(severity.value * self.config.degradation_steps)),
            )

            for step in range(steps):
                reduction_factor = self.config.degradation_reduction_factor ** (
                    step + 1
                )

                action = FallbackAction(
                    action_id=f"degradation_step_{step}_{datetime.now().timestamp()}",
                    action_type="reduce_capacity",
                    parameters={
                        "reduction_factor": reduction_factor,
                        "step": step + 1,
                        "total_steps": steps,
                    },
                    description=f"Reduce system capacity by factor {reduction_factor} (step {step + 1}/{steps})",
                    rollback_action="restore_capacity",
                )
                actions.append(action)

            # アクションを実行
            for action in actions:
                self._execute_action(action)

            return actions

        except Exception as e:
            logger.error(f"Failed to activate gradual degradation: {e}")
            return []

    def _activate_emergency_shutdown(self) -> List[FallbackAction]:
        """緊急停止をアクティベート"""
        actions = []

        try:
            # 完全停止
            action = FallbackAction(
                action_id=f"emergency_shutdown_{datetime.now().timestamp()}",
                action_type="emergency_shutdown",
                parameters={},
                description="Complete system shutdown",
                rollback_action="emergency_startup",
            )
            actions.append(action)

            # アクションを実行
            for action in actions:
                self._execute_action(action)

            return actions

        except Exception as e:
            logger.error(f"Failed to activate emergency shutdown: {e}")
            return []

    def _execute_action(self, action: FallbackAction) -> bool:
        """アクションを実行"""
        try:
            # 実際の取引システムとの統合ポイント
            # ここではログ出力のみ（実際の実装では取引システムのAPIを呼び出す）
            logger.info(
                f"Executing fallback action: {action.action_type} - {action.description}"
            )

            # アクションタイプに応じた処理
            if action.action_type == "limit_position_size":
                # ポジションサイズ制限の実装
                pass
            elif action.action_type == "limit_leverage":
                # レバレッジ制限の実装
                pass
            elif action.action_type == "limit_trade_frequency":
                # 取引頻度制限の実装
                pass
            elif action.action_type == "stop_trading":
                # 取引停止の実装
                pass
            elif action.action_type == "reduce_capacity":
                # 容量削減の実装
                pass
            elif action.action_type == "emergency_shutdown":
                # 緊急停止の実装
                pass

            return True

        except Exception as e:
            logger.error(f"Failed to execute action {action.action_id}: {e}")
            return False

    def _rollback_fallback_actions(self, actions: List[FallbackAction]) -> None:
        """フォールバックアクションをロールバック"""
        try:
            for action in reversed(actions):
                if action.rollback_action:
                    logger.info(f"Rolling back action: {action.action_id}")
                    # ロールバックアクションの実行
                    self._execute_rollback_action(action)

        except Exception as e:
            logger.error(f"Failed to rollback actions: {e}")

    def _execute_rollback_action(self, action: FallbackAction) -> bool:
        """ロールバックアクションを実行"""
        try:
            logger.info(
                f"Executing rollback for action: {action.action_id} - {action.rollback_action}"
            )

            # ロールバックアクションタイプに応じた処理
            if action.rollback_action == "restore_position_size":
                # ポジションサイズ制限の解除
                pass
            elif action.rollback_action == "restore_leverage":
                # レバレッジ制限の解除
                pass
            elif action.rollback_action == "restore_trade_frequency":
                # 取引頻度制限の解除
                pass
            elif action.rollback_action == "resume_trading":
                # 取引再開
                pass
            elif action.rollback_action == "restore_capacity":
                # 容量復元
                pass
            elif action.rollback_action == "emergency_startup":
                # 緊急起動
                pass

            return True

        except Exception as e:
            logger.error(
                f"Failed to execute rollback for action {action.action_id}: {e}"
            )
            return False

    def _calculate_expected_recovery_time(
        self, mode: FallbackMode
    ) -> Optional[datetime]:
        """期待回復時間を計算"""
        try:
            now = datetime.now()

            if mode == FallbackMode.CONSERVATIVE:
                return now + timedelta(
                    hours=self.config.conservative_mode_duration_hours
                )
            elif mode == FallbackMode.CIRCUIT_BREAKER:
                return now + timedelta(
                    minutes=self.config.circuit_breaker_cooldown_minutes
                )
            elif mode == FallbackMode.GRADUAL_DEGRADATION:
                return now + timedelta(minutes=30)  # 30分後
            elif mode == FallbackMode.EMERGENCY_SHUTDOWN:
                return None  # 手動回復が必要

            return None

        except Exception as e:
            logger.error(f"Failed to calculate expected recovery time: {e}")
            return None

    def _check_recovery_conditions(self, safety_status) -> bool:
        """回復条件をチェック"""
        try:
            # 安全スコアが閾値を超えているかチェック
            if safety_status.system_health_score < 0.8:
                return False

            # アクティブな異常が少ないかチェック
            if len(safety_status.active_anomalies) > 2:
                return False

            # 最近の安全チェックが成功しているかチェック
            recent_checks = safety_status.recent_checks[-5:]  # 直近5件
            if len(recent_checks) < 3:
                return False

            success_rate = sum(1 for check in recent_checks if check.passed) / len(
                recent_checks
            )
            if success_rate < 0.8:
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check recovery conditions: {e}")
            return False

    def _monitoring_worker(self) -> None:
        """監視ワーカー"""
        while self.monitoring_active:
            try:
                if self.current_fallback:
                    # タイムアウトチェック
                    if self._check_fallback_timeout():
                        logger.warning("Fallback timeout reached, attempting recovery")
                        self.attempt_recovery()

                    # 定期的な回復チェック
                    elif self._should_check_recovery():
                        self.attempt_recovery()

                time.sleep(self.config.recovery_check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
                time.sleep(30)

    def _check_fallback_timeout(self) -> bool:
        """フォールバックタイムアウトをチェック"""
        if (
            not self.current_fallback
            or not self.current_fallback.expected_recovery_time
        ):
            return False

        return datetime.now() >= self.current_fallback.expected_recovery_time

    def _should_check_recovery(self) -> bool:
        """回復チェックが必要か判定"""
        if not self.current_fallback or not self.current_fallback.last_recovery_check:
            return True

        time_since_last_check = (
            datetime.now() - self.current_fallback.last_recovery_check
        )
        return (
            time_since_last_check.total_seconds()
            >= self.config.recovery_check_interval_seconds
        )

    def add_fallback_activated_callback(
        self, callback: Callable[[FallbackState], None]
    ) -> None:
        """フォールバックアクティベートコールバックを追加"""
        self.fallback_activated_callbacks.append(callback)

    def add_fallback_deactivated_callback(
        self, callback: Callable[[FallbackState], None]
    ) -> None:
        """フォールバック非アクティベートコールバックを追加"""
        self.fallback_deactivated_callbacks.append(callback)

    def add_recovery_attempt_callback(
        self, callback: Callable[[FallbackState], None]
    ) -> None:
        """回復試行コールバックを追加"""
        self.recovery_attempt_callbacks.append(callback)

    def _trigger_fallback_activated_callbacks(
        self, fallback_state: FallbackState
    ) -> None:
        """フォールバックアクティベートコールバックを実行"""
        for callback in self.fallback_activated_callbacks:
            try:
                callback(fallback_state)
            except Exception as e:
                logger.error(f"Fallback activated callback failed: {e}")

    def _trigger_fallback_deactivated_callbacks(
        self, fallback_state: FallbackState
    ) -> None:
        """フォールバック非アクティベートコールバックを実行"""
        for callback in self.fallback_deactivated_callbacks:
            try:
                callback(fallback_state)
            except Exception as e:
                logger.error(f"Fallback deactivated callback failed: {e}")

    def _trigger_recovery_attempt_callbacks(
        self, fallback_state: FallbackState
    ) -> None:
        """回復試行コールバックを実行"""
        for callback in self.recovery_attempt_callbacks:
            try:
                callback(fallback_state)
            except Exception as e:
                logger.error(f"Recovery attempt callback failed: {e}")

    def get_fallback_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """フォールバック履歴を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_fallbacks = [
                f for f in self.fallback_history if f.activated_at > cutoff_time
            ]

            return [
                {
                    "mode": f.mode.value,
                    "activated_at": f.activated_at.isoformat(),
                    "reason": f.reason,
                    "severity": f.severity.value,
                    "recovery_attempts": f.recovery_attempts,
                    "active_actions_count": len(f.active_actions),
                }
                for f in recent_fallbacks
            ]

        except Exception as e:
            logger.error(f"Failed to get fallback history: {e}")
            return []
