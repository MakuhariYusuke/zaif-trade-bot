#!/usr/bin/env python3
"""
Market Adaptation Manager for SAC v435
市場状態変化への適応メカニズム
"""

from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MarketAdaptationManager:
    """
    市場適応マネージャー
    市場状態の変化を検知し、取引戦略を適応させる
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 市場適応設定
        """
        self.config = config

        # 適応設定
        self.regime_window = config.get("regime_window", 50)
        self.adaptation_sensitivity = config.get("adaptation_sensitivity", 0.7)
        self.regime_stability_threshold = config.get("regime_stability_threshold", 0.8)

        # 市場状態定義
        self.regime_definitions = {
            "bull": {
                "trend_threshold": 0.02,
                "volatility_max": 0.20,
                "description": "強気トレンド",
            },
            "bear": {
                "trend_threshold": -0.02,
                "volatility_max": 0.20,
                "description": "弱気トレンド",
            },
            "sideways": {
                "trend_threshold": 0.005,
                "volatility_max": 0.10,
                "description": "横ばい",
            },
            "volatile": {"volatility_min": 0.15, "description": "高ボラティリティ"},
        }

        # 状態追跡
        self.current_regime = "sideways"
        self.regime_history: List[str] = []
        self.regime_stability: float = 0.0
        self.adaptation_factors: Dict[str, float] = {}

        # 適応パラメータ
        self.trend_adaptation = 1.0
        self.volatility_adaptation = 1.0
        self.momentum_adaptation = 1.0

    def adapt_to_market_conditions(
        self, df: pd.DataFrame, current_position: float, portfolio_value: float
    ) -> Dict[str, Any]:
        """
        市場状態に適応したパラメータを計算

        Args:
            df: 市場データ
            current_position: 現在のポジション
            portfolio_value: ポートフォリオ価値

        Returns:
            適応パラメータ辞書
        """
        # 市場状態検知
        new_regime = self._detect_market_regime(df)

        # 状態変化チェック
        regime_changed = new_regime != self.current_regime
        if regime_changed:
            self._handle_regime_change(new_regime, df)
            self.current_regime = new_regime

        # 適応係数計算
        adaptation_factors = self._calculate_adaptation_factors(df, new_regime)

        # 安定性評価
        self._update_regime_stability()

        # 適応適用
        adapted_parameters = self._apply_adaptation(
            adaptation_factors, current_position, portfolio_value
        )

        logger.debug(
            f"Market adaptation: regime={new_regime}, stability={self.regime_stability:.2f}, "
            f"factors={adaptation_factors}"
        )

        return adapted_parameters

    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        市場状態を検知

        Args:
            df: 市場データ

        Returns:
            検知された市場状態
        """
        if len(df) < self.regime_window:
            return self.current_regime

        # 最近のデータを使用
        recent_data = df.tail(self.regime_window)

        # トレンド計算
        prices = recent_data["close"].to_numpy(dtype=np.float64)
        trend = (prices[-1] - prices[0]) / prices[0]

        # ボラティリティ計算
        returns = np.diff(prices) / prices[:-1]
        from ztb.metrics.technical import calculate_volatility_from_returns

        volatility = calculate_volatility_from_returns(
            returns, window=len(returns), annualize=True
        )

        # RSI計算（簡易版）
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-8)))

        # 状態判定
        if volatility >= cast(
            float, self.regime_definitions["volatile"]["volatility_min"]
        ):
            return "volatile"
        elif abs(trend) >= cast(
            float, self.regime_definitions["bull"]["trend_threshold"]
        ):
            return "bull" if trend > 0 else "bear"
        elif abs(trend) <= cast(
            float, self.regime_definitions["sideways"]["trend_threshold"]
        ) and volatility <= cast(
            float, self.regime_definitions["sideways"]["volatility_max"]
        ):
            return "sideways"
        else:
            return "sideways"  # デフォルト

    def _handle_regime_change(self, new_regime: str, df: pd.DataFrame) -> None:
        """
        市場状態変化を処理

        Args:
            new_regime: 新しい市場状態
            df: 市場データ
        """
        logger.info(f"Market regime changed: {self.current_regime} -> {new_regime}")

        # 履歴更新
        self.regime_history.append(new_regime)
        if len(self.regime_history) > 20:
            self.regime_history = self.regime_history[-20:]

        # 適応係数のリセット（段階的）
        self._smooth_adaptation_transition(new_regime)

    def _calculate_adaptation_factors(
        self, df: pd.DataFrame, regime: str
    ) -> Dict[str, float]:
        """
        適応係数を計算

        Args:
            df: 市場データ
            regime: 現在の市場状態

        Returns:
            適応係数辞書
        """
        factors = {}

        if len(df) < 20:
            return {"trend": 1.0, "volatility": 1.0, "momentum": 1.0}

        recent_data = df.tail(20)

        # トレンド適応
        if regime in ["bull", "bear"]:
            factors["trend"] = 1.2 if regime == "bull" else 0.8
        else:
            factors["trend"] = 1.0

        # ボラティリティ適応
        returns = recent_data["close"].pct_change().dropna()
        volatility = returns.std()
        if volatility > 0.02:  # 高ボラティリティ
            factors["volatility"] = 0.7
        elif volatility < 0.005:  # 低ボラティリティ
            factors["volatility"] = 1.3
        else:
            factors["volatility"] = 1.0

        # モメンタム適応
        momentum = (
            recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]
        ) / recent_data["close"].iloc[0]
        if abs(momentum) > 0.01:
            factors["momentum"] = 1.1
        else:
            factors["momentum"] = 0.9

        # 安定性による調整
        stability_factor = min(1.0, self.regime_stability + 0.5)
        for key in factors:
            factors[key] *= stability_factor

        self.adaptation_factors = factors
        return factors

    def _smooth_adaptation_transition(self, new_regime: str) -> None:
        """
        適応遷移をスムーズに

        Args:
            new_regime: 新しい市場状態
        """
        # 急激な変化を避けるためのスムージング
        transition_rate = 0.3  # 30%ずつ変化

        target_trend = (
            1.2 if new_regime == "bull" else 0.8 if new_regime == "bear" else 1.0
        )
        self.trend_adaptation = (
            self.trend_adaptation * (1 - transition_rate)
            + target_trend * transition_rate
        )

        target_volatility = 0.7 if new_regime == "volatile" else 1.0
        self.volatility_adaptation = (
            self.volatility_adaptation * (1 - transition_rate)
            + target_volatility * transition_rate
        )

    def _update_regime_stability(self) -> None:
        """
        市場状態の安定性を更新
        """
        if len(self.regime_history) < 5:
            self.regime_stability = 0.5
            return

        # 最近の状態の一致度を計算
        recent_regimes = self.regime_history[-5:]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        consistency = recent_regimes.count(most_common) / len(recent_regimes)

        self.regime_stability = consistency

    def _apply_adaptation(
        self, factors: Dict[str, float], current_position: float, portfolio_value: float
    ) -> Dict[str, Any]:
        """
        適応を適用したパラメータを生成

        Args:
            factors: 適応係数
            current_position: 現在のポジション
            portfolio_value: ポートフォリオ価値

        Returns:
            適応適用後のパラメータ
        """
        # ポジションサイズ適応
        position_multiplier = factors.get("volatility", 1.0) * factors.get("trend", 1.0)
        adapted_position_size = min(
            0.2, max(0.01, current_position * position_multiplier)
        )

        # 取引頻度適応
        trade_frequency_multiplier = factors.get("momentum", 1.0) * (
            2.0 - self.regime_stability
        )
        adapted_trade_frequency = min(1.0, max(0.1, trade_frequency_multiplier))

        # リスク許容度適応
        risk_tolerance = self.regime_stability * factors.get("volatility", 1.0)
        adapted_risk_tolerance = min(1.0, max(0.2, risk_tolerance))

        return {
            "adapted_position_size": adapted_position_size,
            "adapted_trade_frequency": adapted_trade_frequency,
            "adapted_risk_tolerance": adapted_risk_tolerance,
            "market_regime": self.current_regime,
            "regime_stability": self.regime_stability,
            "adaptation_factors": factors,
        }

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        適応指標を取得

        Returns:
            適応指標辞書
        """
        return {
            "current_regime": self.current_regime,
            "regime_stability": self.regime_stability,
            "adaptation_factors": self.adaptation_factors,
            "trend_adaptation": self.trend_adaptation,
            "volatility_adaptation": self.volatility_adaptation,
            "momentum_adaptation": self.momentum_adaptation,
        }

    def reset(self) -> None:
        """状態のリセット"""
        self.current_regime = "sideways"
        self.regime_history.clear()
        self.regime_stability = 0.0
        self.adaptation_factors.clear()
        self.trend_adaptation = 1.0
        self.volatility_adaptation = 1.0
        self.momentum_adaptation = 1.0
