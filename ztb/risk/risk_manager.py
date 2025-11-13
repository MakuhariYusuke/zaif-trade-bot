#!/usr/bin/env python3
"""
Risk Manager for SAC v435
統合リスク管理システム
"""

from typing import Any, Dict, Optional

import pandas as pd

from ztb.risk.drawdown_controller import DrawdownController
from ztb.risk.dynamic_position_sizer import DynamicPositionSizer
from ztb.risk.market_adaptation_manager import MarketAdaptationManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    統合リスクマネージャー
    動的ポジションサイジング、ドローダウン制御、市場適応を統合
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: リスク管理設定
        """
        self.config = config

        # サブコンポーネント初期化
        self.position_sizer = DynamicPositionSizer(config)
        self.drawdown_controller = DrawdownController(config)
        self.market_adaptor = MarketAdaptationManager(config)

        # 統合設定
        self.enabled = config.get("enabled", True)
        self.correlation_risk_control = config.get("correlation_risk_control", True)
        self.max_correlation_exposure = config.get("max_correlation_exposure", 0.7)

        logger.info(
            "Risk Manager initialized with components: position_sizer, drawdown_controller, market_adaptor"
        )

    def calculate_risk_adjusted_position(
        self,
        base_position: float,
        current_price: float,
        portfolio_value: float,
        atr: float,
        df: Optional[pd.DataFrame] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        リスク調整済みポジションを計算

        Args:
            base_position: 基本ポジションサイズ
            current_price: 現在の価格
            portfolio_value: ポートフォリオ価値
            atr: ATR値
            df: 市場データ
            step: 現在のステップ

        Returns:
            リスク調整情報辞書
        """
        if not self.enabled:
            return {
                "adjusted_position": base_position,
                "risk_level": 0.0,
                "control_active": False,
                "reasons": ["Risk management disabled"],
            }

        # ドローダウン制御更新
        drawdown_info = self.drawdown_controller.update_portfolio_value(
            portfolio_value, step
        )

        # 緊急停止チェック
        if drawdown_info["emergency_stop"]:
            return {
                "adjusted_position": 0.0,
                "risk_level": 1.0,
                "control_active": True,
                "reasons": ["Emergency stop activated"],
                "drawdown_info": drawdown_info,
            }

        # 市場適応
        market_regime = "ranging"  # デフォルト
        if df is not None:
            adaptation_info = self.market_adaptor.adapt_to_market_conditions(
                df, base_position, portfolio_value
            )
            market_regime = adaptation_info.get("market_regime", "ranging")
        else:
            adaptation_info = {"adapted_position_size": base_position}

        # 動的ポジションサイジング
        sized_position = self.position_sizer.calculate_position_size(
            base_position=adaptation_info["adapted_position_size"],
            current_price=current_price,
            portfolio_value=portfolio_value,
            atr=atr,
            market_regime=market_regime,
            df=df,
        )

        # ドローダウンベースの削減適用
        final_position = sized_position * drawdown_info["position_reduction_factor"]

        # 相関リスク制御（将来拡張用）
        correlation_adjustment = 1.0
        if self.correlation_risk_control:
            correlation_adjustment = self._apply_correlation_risk_control(
                final_position
            )

        final_position *= correlation_adjustment

        # リスクレベル計算
        risk_level = self._calculate_overall_risk_level(drawdown_info, adaptation_info)

        # 制御理由の収集
        reasons = []
        if drawdown_info["position_reduction_factor"] < 1.0:
            reasons.append(
                f"Drawdown control: {drawdown_info['position_reduction_factor']:.2f}"
            )
        if adaptation_info.get("adapted_position_size", base_position) != base_position:
            reasons.append("Market adaptation applied")
        if correlation_adjustment < 1.0:
            reasons.append("Correlation risk control applied")

        result = {
            "adjusted_position": final_position,
            "risk_level": risk_level,
            "control_active": len(reasons) > 0,
            "reasons": reasons if reasons else ["No risk controls active"],
            "drawdown_info": drawdown_info,
            "adaptation_info": adaptation_info,
            "market_regime": market_regime,
        }

        logger.debug(f"Risk adjusted position: {result}") if len(reasons) > 0 else None
        return result

    def _apply_correlation_risk_control(self, position: float) -> float:
        """
        相関リスク制御を適用（将来拡張用）

        Args:
            position: 現在のポジション

        Returns:
            調整されたポジション
        """
        # 現時点では基本的な制限のみ
        # 将来的には複数資産間の相関を考慮した制御を実装
        return min(1.0, position / self.max_correlation_exposure)

    def _calculate_overall_risk_level(
        self, drawdown_info: Dict[str, Any], adaptation_info: Dict[str, Any]
    ) -> float:
        """
        全体的なリスクレベルを計算

        Args:
            drawdown_info: ドローダウン情報
            adaptation_info: 適応情報

        Returns:
            リスクレベル（0.0-1.0）
        """
        # ドローダウンベースのリスク
        drawdown_risk = min(
            1.0,
            drawdown_info["current_drawdown"]
            / self.drawdown_controller.max_drawdown_limit,
        )

        # 安定性ベースのリスク
        stability_risk = 1.0 - adaptation_info.get("regime_stability", 0.5)

        # 適応係数ベースのリスク
        adaptation_risk = 0.0
        factors = adaptation_info.get("adaptation_factors", {})
        if factors:
            # 適応係数が1.0から離れるほどリスクが高い
            adaptation_risk = sum(abs(f - 1.0) for f in factors.values()) / len(factors)

        # 重み付き平均
        overall_risk = (
            drawdown_risk * 0.5 + stability_risk * 0.3 + adaptation_risk * 0.2
        )

        return min(1.0, overall_risk)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        包括的なリスク指標を取得

        Returns:
            リスク指標辞書
        """
        return {
            "drawdown_metrics": self.drawdown_controller.get_risk_metrics(),
            "adaptation_metrics": self.market_adaptor.get_adaptation_metrics(),
            "position_sizing_active": self.position_sizer.volatility_adjustment,
            "emergency_stop_active": self.drawdown_controller.is_emergency_stop,
            "overall_risk_level": self._calculate_overall_risk_level(
                self.drawdown_controller.get_risk_metrics(),
                self.market_adaptor.get_adaptation_metrics(),
            ),
        }

    def should_force_close_positions(self) -> bool:
        """
        ポジション強制決済が必要かどうか

        Returns:
            強制決済が必要かどうか
        """
        return self.drawdown_controller.should_force_close_positions()

    def reset(self) -> None:
        """全コンポーネントのリセット"""
        self.position_sizer.reset()
        self.drawdown_controller.reset()
        self.market_adaptor.reset()
        logger.info("Risk Manager reset completed")
