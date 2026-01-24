"""
Dynamic Reward Shaper Component.

This component applies dynamic reward shaping based on market conditions.
Follows Single Responsibility Principle by focusing only on reward shaping.
"""

import math
from typing import Optional

from ztb.utils.logging_utils import get_logger

from .interfaces import IDynamicRewardShaper, IMarketRegimeDetector


class DynamicRewardShaper(IDynamicRewardShaper):
    """
    Applies dynamic reward shaping based on market conditions.

    This class encapsulates dynamic reward shaping logic including:
    - Market regime-based adjustments
    - Volatility-adjusted rewards
    - Trend strength bonuses
    """

    def __init__(
        self,
        market_regime_detector: Optional[IMarketRegimeDetector] = None,
        enabled: bool = False,
        market_regime_awareness: bool = False,
        volatility_adjusted_rewards: bool = False,
        trend_strength_bonus: bool = False,
        # Regime coefficients
        bull_market_bonus_coeff: float = 1.2,
        bear_market_penalty_coeff: float = 0.8,
        sideways_market_penalty_coeff: float = 0.9,
        volatile_market_bonus_coeff: float = 1.1,
        # Volatility coefficients
        high_volatility_threshold: float = 0.02,
        low_volatility_threshold: float = 0.005,
        high_volatility_bonus: float = 1.3,
        low_volatility_penalty: float = 0.7,
        # Trend coefficients
        trend_strength_threshold: float = 0.001,
        strong_trend_bonus: float = 1.2,
        weak_trend_penalty: float = 0.9,
    ):
        """
        Initialize DynamicRewardShaper.

        Args:
            market_regime_detector: Component for regime detection
            enabled: Whether dynamic shaping is enabled
            market_regime_awareness: Whether to adjust based on market regime
            volatility_adjusted_rewards: Whether to adjust based on volatility
            trend_strength_bonus: Whether to adjust based on trend strength
            bull_market_bonus_coeff: Bonus multiplier for bull markets
            bear_market_penalty_coeff: Penalty multiplier for bear markets
            sideways_market_penalty_coeff: Penalty multiplier for sideways markets
            volatile_market_bonus_coeff: Bonus multiplier for volatile markets
            high_volatility_threshold: Threshold for high volatility
            low_volatility_threshold: Threshold for low volatility
            high_volatility_bonus: Bonus for high volatility
            low_volatility_penalty: Penalty for low volatility
            trend_strength_threshold: Threshold for strong trend
            strong_trend_bonus: Bonus for strong trends
            weak_trend_penalty: Penalty for weak trends
        """
        self.market_regime_detector = market_regime_detector
        self.enabled = enabled
        self.market_regime_awareness = market_regime_awareness
        self.volatility_adjusted_rewards = volatility_adjusted_rewards
        self.trend_strength_bonus = trend_strength_bonus

        # Regime coefficients
        self.bull_market_bonus_coeff = bull_market_bonus_coeff
        self.bear_market_penalty_coeff = bear_market_penalty_coeff
        self.sideways_market_penalty_coeff = sideways_market_penalty_coeff
        self.volatile_market_bonus_coeff = volatile_market_bonus_coeff

        # Volatility coefficients
        self.high_volatility_threshold = high_volatility_threshold
        self.low_volatility_threshold = low_volatility_threshold
        self.high_volatility_bonus = high_volatility_bonus
        self.low_volatility_penalty = low_volatility_penalty

        # Trend coefficients
        self.trend_strength_threshold = trend_strength_threshold
        self.strong_trend_bonus = strong_trend_bonus
        self.weak_trend_penalty = weak_trend_penalty

        self.logger = get_logger("ztb.trading.environment.dynamic_reward_shaper")

    def shape_reward(
        self, base_reward: float, current_price: float, step: int, pnl: float
    ) -> float:
        """
        Apply dynamic reward shaping based on market conditions.

        Args:
            base_reward: Base reward before shaping
            current_price: Current market price
            step: Current step number
            pnl: Profit/Loss from action

        Returns:
            Shaped reward value
        """
        if not self.enabled:
            return base_reward

        shaped_reward = base_reward

        # Market regime awareness
        if self.market_regime_awareness:
            try:
                # Support detectors with different signatures: prefer (current_price, step),
                # fall back to older interface that expects market_data.
                try:
                    regime = self.market_regime_detector.detect_regime(current_price, step)
                except TypeError:
                    # Older detectors may expect a DataFrame or single 'market_data' argument.
                    # Construct minimal DataFrame-like object when possible or call with a single arg.
                    try:
                        import pandas as pd

                        market_df = pd.DataFrame({"close": [current_price]})
                        regime = self.market_regime_detector.detect_regime(market_df)
                    except Exception:
                        # Last resort: call with current_price only
                        regime = self.market_regime_detector.detect_regime(current_price)

                if regime == "bull":
                    shaped_reward *= self.bull_market_bonus_coeff
                    self.logger.debug(
                        f"Applied bull market bonus: {self.bull_market_bonus_coeff}x"
                    )
                elif regime == "bear":
                    shaped_reward *= self.bear_market_penalty_coeff
                    self.logger.debug(
                        f"Applied bear market penalty: {self.bear_market_penalty_coeff}x"
                    )
                elif regime == "sideways":
                    shaped_reward *= self.sideways_market_penalty_coeff
                    self.logger.debug(
                        f"Applied sideways market penalty: {self.sideways_market_penalty_coeff}x"
                    )
                elif regime == "volatile":
                    shaped_reward *= self.volatile_market_bonus_coeff
                    self.logger.debug(
                        f"Applied volatile market bonus: {self.volatile_market_bonus_coeff}x"
                    )
            except Exception:
                # Be defensive: dynamic shaping must not break reward calculation
                self.logger.exception("Market regime detection failed; skipping regime-based shaping")
                regime = "sideways"

        # Volatility adjusted rewards
        if (
            self.volatility_adjusted_rewards
            and len(self.market_regime_detector.price_history) >= 10
        ):
            prices = self.market_regime_detector.price_history[-20:]  # Last 20 prices
            returns = [prices[i + 1] / prices[i] - 1 for i in range(len(prices) - 1)]
            if returns:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                volatility = math.sqrt(variance)
            else:
                volatility = 0.0

            if volatility > self.high_volatility_threshold:
                shaped_reward *= self.high_volatility_bonus
                self.logger.debug(
                    f"Applied high volatility bonus: {self.high_volatility_bonus}x"
                )
            elif volatility < self.low_volatility_threshold:
                shaped_reward *= self.low_volatility_penalty
                self.logger.debug(
                    f"Applied low volatility penalty: {self.low_volatility_penalty}x"
                )

        # Trend strength bonus
        if (
            self.trend_strength_bonus
            and len(self.market_regime_detector.price_history) >= 10
        ):
            prices = self.market_regime_detector.price_history[-20:]
            returns = [prices[i + 1] / prices[i] - 1 for i in range(len(prices) - 1)]
            trend_strength = abs(sum(returns)) if returns else 0.0

            if trend_strength > self.trend_strength_threshold:
                shaped_reward *= self.strong_trend_bonus
                self.logger.debug(
                    f"Applied strong trend bonus: {self.strong_trend_bonus}x"
                )
            else:
                shaped_reward *= self.weak_trend_penalty
                self.logger.debug(
                    f"Applied weak trend penalty: {self.weak_trend_penalty}x"
                )

        return shaped_reward
