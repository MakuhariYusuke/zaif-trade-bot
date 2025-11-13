#!/usr/bin/env python3
"""
Dynamic Position Sizing for SAC v435
市場状態に応じた適応型ポジションサイジングシステム
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DynamicPositionSizer:
    """
    動的ポジションサイジングクラス
    ボラティリティ、市場状態、ドローダウンに基づいてポジションサイズを調整
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: ポジションサイジング設定
        """
        self.config = config

        # 基本設定
        self.min_position_size = config.get("position_size_min", 0.01)
        self.max_position_size = config.get("position_size_max", 0.2)
        self.volatility_adjustment = config.get("volatility_adjustment", True)
        self.drawdown_control = config.get("drawdown_control", True)
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.1)

        # 適応パラメータ
        self.volatility_window = config.get("volatility_window", 20)
        self.drawdown_window = config.get("drawdown_window", 50)

        # 状態追跡
        self.portfolio_value_history: List[float] = []
        self.position_history: List[float] = []
        self.volatility_history: List[float] = []

        # 適応係数
        self.volatility_multiplier = 1.0
        self.drawdown_multiplier = 1.0
        self.market_regime_multiplier = 1.0

    def calculate_position_size(
        self,
        base_position: float,
        current_price: float,
        portfolio_value: float,
        atr: float,
        market_regime: str = "ranging",
        df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        動的ポジションサイズを計算

        Args:
            base_position: 基本ポジションサイズ
            current_price: 現在の価格
            portfolio_value: ポートフォリオ価値
            atr: ATR値
            market_regime: 市場状態
            df: 価格データ（オプション）

        Returns:
            調整されたポジションサイズ
        """
        # 基本サイズの計算
        adjusted_size = base_position

        # ボラティリティ調整
        if self.volatility_adjustment and atr > 0:
            adjusted_size = self._apply_volatility_adjustment(adjusted_size, atr, df)

        # ドローダウン制御
        if self.drawdown_control and len(self.portfolio_value_history) > 0:
            adjusted_size = self._apply_drawdown_control(adjusted_size, portfolio_value)

        # 市場状態適応
        adjusted_size = self._apply_market_regime_adjustment(
            adjusted_size, market_regime
        )

        # サイズ制限の適用
        adjusted_size = self._apply_size_limits(
            adjusted_size, portfolio_value, current_price
        )

        # 状態更新
        self._update_state(portfolio_value, adjusted_size, atr)

        logger.debug(
            f"Position sizing: base={base_position:.4f}, adjusted={adjusted_size:.4f}, "
            f"vol_mult={self.volatility_multiplier:.2f}, dd_mult={self.drawdown_multiplier:.2f}, "
            f"regime_mult={self.market_regime_multiplier:.2f}"
        ) if hasattr(self, '_sizing_step_count') and self._sizing_step_count % 20 == 0 else None

        return adjusted_size

    def _apply_volatility_adjustment(
        self, position_size: float, atr: float, df: Optional[pd.DataFrame]
    ) -> float:
        """
        ボラティリティに基づくポジションサイズ調整

        Args:
            position_size: 現在のポジションサイズ
            atr: ATR値
            df: 価格データ

        Returns:
            調整されたポジションサイズ
        """
        if df is None or len(df) < self.volatility_window:
            # ATRベースのシンプルな調整
            if atr > 0:
                # ATRが大きいほどポジションを小さく
                volatility_factor = 1.0 / (1.0 + atr * 10.0)
                self.volatility_multiplier = max(0.1, min(2.0, volatility_factor))
            else:
                self.volatility_multiplier = 1.0
        else:
            # ヒストリカルボラティリティベースの調整
            recent_prices = df["close"].tail(self.volatility_window)
            returns = recent_prices.pct_change().dropna()

            if len(returns) > 0:
                hist_volatility = returns.std() * np.sqrt(252)  # 年率化
                # ボラティリティが大きいほどポジションを小さく
                volatility_factor = 1.0 / (1.0 + hist_volatility * 5.0)
                self.volatility_multiplier = max(0.1, min(2.0, volatility_factor))
            else:
                self.volatility_multiplier = 1.0

        return position_size * self.volatility_multiplier

    def _apply_drawdown_control(
        self, position_size: float, current_portfolio_value: float
    ) -> float:
        """
        ドローダウン制御によるポジションサイズ調整

        Args:
            position_size: 現在のポジションサイズ
            current_portfolio_value: 現在のポートフォリオ価値

        Returns:
            調整されたポジションサイズ
        """
        if len(self.portfolio_value_history) < self.drawdown_window:
            self.drawdown_multiplier = 1.0
            return position_size

        # ドローダウン計算
        max_value = max(self.portfolio_value_history[-self.drawdown_window :])
        current_drawdown = (max_value - current_portfolio_value) / max_value

        if current_drawdown > self.max_drawdown_limit:
            # ドローダウンが大きいほどポジションを小さく
            drawdown_factor = max(
                0.1, 1.0 - (current_drawdown - self.max_drawdown_limit) * 5.0
            )
            self.drawdown_multiplier = drawdown_factor
        else:
            self.drawdown_multiplier = 1.0

        return position_size * self.drawdown_multiplier

    def _apply_market_regime_adjustment(
        self, position_size: float, market_regime: str
    ) -> float:
        """
        市場状態に応じたポジションサイズ調整

        Args:
            position_size: 現在のポジションサイズ
            market_regime: 市場状態

        Returns:
            調整されたポジションサイズ
        """
        regime_multipliers = {
            "trending": 1.2,  # トレンド相場では少し大きく
            "ranging": 0.8,  # レンジ相場では少し小さく
            "high_volatility": 0.6,  # 高ボラティリティでは小さく
            "low_volatility": 1.4,  # 低ボラティリティでは大きく
        }

        self.market_regime_multiplier = regime_multipliers.get(market_regime, 1.0)
        return position_size * self.market_regime_multiplier

    def _apply_size_limits(
        self, position_size: float, portfolio_value: float, current_price: float
    ) -> float:
        """
        ポジションサイズの制限適用

        Args:
            position_size: 調整されたポジションサイズ
            portfolio_value: ポートフォリオ価値
            current_price: 現在の価格

        Returns:
            制限適用後のポジションサイズ
        """
        # 絶対サイズ制限
        position_size = max(
            self.min_position_size, min(self.max_position_size, position_size)
        )

        # ポートフォリオ価値ベースの制限
        max_portfolio_position = (
            portfolio_value * self.max_position_size / current_price
        )
        position_size = min(position_size, max_portfolio_position)

        return position_size

    def _update_state(
        self, portfolio_value: float, position_size: float, atr: float
    ) -> None:
        """
        内部状態の更新

        Args:
            portfolio_value: ポートフォリオ価値
            position_size: ポジションサイズ
            atr: ATR値
        """
        self.portfolio_value_history.append(portfolio_value)
        self.position_history.append(position_size)
        self.volatility_history.append(atr)

        # 履歴サイズ制限
        max_history = max(self.volatility_window, self.drawdown_window) * 2
        if len(self.portfolio_value_history) > max_history:
            self.portfolio_value_history = self.portfolio_value_history[-max_history:]
            self.position_history = self.position_history[-max_history:]
            self.volatility_history = self.volatility_history[-max_history:]

    def reset(self) -> None:
        """状態のリセット"""
        self.portfolio_value_history.clear()
        self.position_history.clear()
        self.volatility_history.clear()
        self.volatility_multiplier = 1.0
        self.drawdown_multiplier = 1.0
        self.market_regime_multiplier = 1.0
