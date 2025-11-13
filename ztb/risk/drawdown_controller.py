#!/usr/bin/env python3
"""
Drawdown Controller for SAC v435
ドローダウン制御とリスク管理システム
"""

from typing import Any, Dict, List

import numpy as np

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DrawdownController:
    """
    ドローダウン制御クラス
    ポートフォリオのドローダウンを監視し、リスクを制御
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: ドローダウン制御設定
        """
        self.config = config

        # 基本設定
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.1)  # 10%
        self.drawdown_window = config.get("drawdown_window", 50)
        self.recovery_threshold = config.get("recovery_threshold", 0.05)  # 5%回復で緩和

        # 緊急制御設定
        self.emergency_stop_enabled = config.get("emergency_stop_enabled", True)
        self.emergency_stop_threshold = config.get(
            "emergency_stop_threshold", 0.15
        )  # 15%

        # 段階的制御設定
        self.warning_threshold = config.get("warning_threshold", 0.05)  # 5%
        self.reduction_threshold = config.get("reduction_threshold", 0.08)  # 8%

        # 状態追跡
        self.portfolio_value_history: List[float] = []
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # 制御状態
        self.is_emergency_stop = False
        self.position_reduction_factor = 1.0
        self.last_warning_step = -1

    def update_portfolio_value(
        self, portfolio_value: float, step: int
    ) -> Dict[str, Any]:
        """
        ポートフォリオ価値を更新し、ドローダウンを計算

        Args:
            portfolio_value: 現在のポートフォリオ価値
            step: 現在のステップ

        Returns:
            制御情報辞書
        """
        # 履歴更新
        self.portfolio_value_history.append(portfolio_value)

        # 履歴サイズ制限
        if len(self.portfolio_value_history) > self.drawdown_window * 2:
            self.portfolio_value_history = self.portfolio_value_history[
                -self.drawdown_window :
            ]

        # ピーク価値更新
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            # 回復時は緊急停止解除
            if self.is_emergency_stop and self._check_recovery():
                self.is_emergency_stop = False
                logger.info(f"Emergency stop lifted at step {step}")

        # ドローダウン計算
        if self.peak_value > 0:
            self.current_drawdown = (
                self.peak_value - portfolio_value
            ) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # 制御ロジック適用
        control_info = self._apply_risk_controls(step)

        logger.debug(
            f"Drawdown control: value={portfolio_value:.2f}, peak={self.peak_value:.2f}, "
            f"current_dd={self.current_drawdown:.4f}, max_dd={self.max_drawdown:.4f}, "
            f"reduction_factor={self.position_reduction_factor:.2f}"
        ) if step % 20 == 0 else None

        return control_info

    def _apply_risk_controls(self, step: int) -> Dict[str, Any]:
        """
        リスク制御を適用

        Args:
            step: 現在のステップ

        Returns:
            制御情報
        """
        control_info = {
            "emergency_stop": self.is_emergency_stop,
            "position_reduction_factor": self.position_reduction_factor,
            "warning_triggered": False,
            "reduction_triggered": False,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
        }

        # 緊急停止チェック
        if (
            self.emergency_stop_enabled
            and self.current_drawdown >= self.emergency_stop_threshold
        ):
            if not self.is_emergency_stop:
                self.is_emergency_stop = True
                logger.warning(
                    f"🚨 Emergency stop triggered at step {step}: "
                    f"drawdown {self.current_drawdown:.1%} exceeds threshold {self.emergency_stop_threshold:.1%}"
                )
            control_info["emergency_stop"] = True
            control_info["position_reduction_factor"] = 0.0  # 完全停止

        # 段階的削減
        elif self.current_drawdown >= self.reduction_threshold:
            reduction_factor = max(
                0.3, 1.0 - (self.current_drawdown - self.reduction_threshold) * 5.0
            )
            self.position_reduction_factor = reduction_factor
            control_info["reduction_triggered"] = True
            control_info["position_reduction_factor"] = reduction_factor

        # 警告
        elif self.current_drawdown >= self.warning_threshold:
            if step - self.last_warning_step > 10:  # 10ステップごとに警告
                logger.warning(
                    f"⚠️ High drawdown warning at step {step}: "
                    f"drawdown {self.current_drawdown:.1%} exceeds warning threshold {self.warning_threshold:.1%}"
                )
                self.last_warning_step = step
            control_info["warning_triggered"] = True
            control_info["position_reduction_factor"] = 0.8  # 軽微削減

        else:
            # 正常状態
            self.position_reduction_factor = 1.0
            control_info["position_reduction_factor"] = 1.0

        return control_info

    def _check_recovery(self) -> bool:
        """
        回復状態をチェック

        Returns:
            回復しているかどうか
        """
        if len(self.portfolio_value_history) < 10:
            return False

        # 最近の値動きをチェック
        recent_values = self.portfolio_value_history[-10:]
        recovery_amount = (recent_values[-1] - min(recent_values)) / self.peak_value

        return recovery_amount >= self.recovery_threshold

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        リスク指標を取得

        Returns:
            リスク指標辞書
        """
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "peak_value": self.peak_value,
            "recovery_progress": self._calculate_recovery_progress(),
            "risk_level": self._calculate_risk_level(),
        }

    def _calculate_recovery_progress(self) -> float:
        """
        回復進捗を計算

        Returns:
            回復進捗（0.0-1.0）
        """
        if self.peak_value == 0 or not self.portfolio_value_history:
            return 0.0

        current_value = self.portfolio_value_history[-1]
        recovery_needed = self.peak_value - (self.peak_value * (1 - self.max_drawdown))
        recovery_achieved = current_value - (self.peak_value * (1 - self.max_drawdown))

        if recovery_needed > 0:
            return min(1.0, recovery_achieved / recovery_needed)
        return 1.0

    def _calculate_risk_level(self) -> float:
        """
        リスクレベルを計算（0.0-1.0）

        Returns:
            リスクレベル
        """
        # ドローダウンベースのリスクレベル
        drawdown_risk = min(1.0, self.current_drawdown / self.max_drawdown_limit)

        # ボラティリティベースのリスク（簡易版）
        if len(self.portfolio_value_history) >= 10:
            recent_returns = (
                np.diff(self.portfolio_value_history[-10:])
                / self.portfolio_value_history[-11:-1]
            )
            volatility_risk = min(1.0, np.std(recent_returns) * 10)
        else:
            volatility_risk = 0.0

        return max(drawdown_risk, volatility_risk)

    def should_force_close_positions(self) -> bool:
        """
        ポジション強制決済が必要かどうか

        Returns:
            強制決済が必要かどうか
        """
        return (
            self.is_emergency_stop
            or self.current_drawdown >= self.emergency_stop_threshold
        )

    def reset(self) -> None:
        """状態のリセット"""
        self.portfolio_value_history.clear()
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.is_emergency_stop = False
        self.position_reduction_factor = 1.0
        self.last_warning_step = -1
