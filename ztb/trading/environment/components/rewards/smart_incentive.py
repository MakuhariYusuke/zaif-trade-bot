import logging
from typing import Any

from ztb.trading.constants import ACTION_BUY, ACTION_SELL

from .base import RewardComponent, RewardContext

class SmartIncentiveReward(RewardComponent):
    """
    Reward component that adjusts incentives based on market regime (volatility, trend).
    Implements 'Smart Incentive' logic to adapt rewards to market conditions.

    Leverages DynamicRewardShaper if available for consistent regime-based shaping.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_name(self) -> str:
        return "smart_incentive"

    def calculate(self, context: RewardContext) -> float:
        """
        Calculate reward with regime-based adjustments.
        """
        # Basic PnL calculation
        # Normalize PnL by ATR if available to make it volatility-independent unit
        if context.atr > 1e-8:
            normalized_pnl = context.pnl / context.atr
        else:
            normalized_pnl = context.pnl

        # Apply reward scaling
        base_reward = normalized_pnl * context.reward_scaling

        smart_multiplier = 1.0

        # Use DynamicRewardShaper if available for advanced regime adaptation
        if context.dynamic_reward_shaper and getattr(
            context.dynamic_reward_shaper, "enabled", False
        ):
            base_reward = context.dynamic_reward_shaper.shape_reward(
                base_reward, context.current_price, context.step, context.pnl
            )

        # --- Regime-based Action Adjustment (Range Market Logic) ---
        # This logic runs regardless of DynamicRewardShaper to specifically address action bias
        if context.market_regime_detector:
            detector = context.market_regime_detector
            # Try to get detailed regime info if available
            regime_str = ""
            momentum = 0.0

            if hasattr(detector, "regime_history") and detector.regime_history:
                last_info = detector.regime_history[-1]
                regime_str = str(last_info.get("regime", "")).upper()
                indicators = last_info.get("indicators", {})
                momentum = indicators.get("momentum", 0.0)
            else:
                # Fallback to simple detection
                regime_str = str(
                    detector.detect_regime(context.current_price, context.step)
                ).upper()

            # Check for Range/Consolidation/Sideways
            if any(r in regime_str for r in ["SIDEWAYS", "RANGING", "CONSOLIDATION"]):
                # In ranging markets, penalize trend-following entries (breakout failures)
                # and reward mean-reversion entries.

                if context.action == ACTION_BUY:
                    if (
                        momentum > 0.002
                    ):  # Buying into strength in range -> Risk of buying top
                        smart_multiplier *= 0.8
                        # self.logger.debug(f"Range BUY penalty: momentum={momentum:.4f}")
                    elif momentum < -0.002:  # Buying into weakness -> Mean reversion
                        smart_multiplier *= 1.1

                elif context.action == ACTION_SELL:
                    if (
                        momentum < -0.002
                    ):  # Selling into weakness in range -> Risk of selling bottom
                        smart_multiplier *= 0.8
                        # self.logger.debug(f"Range SELL penalty: momentum={momentum:.4f}")
                    elif momentum > 0.002:  # Selling into strength -> Mean reversion
                        smart_multiplier *= 1.1

        # Fallback / Simplified Logic if DynamicRewardShaper is NOT used
        if not (
            context.dynamic_reward_shaper
            and getattr(context.dynamic_reward_shaper, "enabled", False)
        ):
            # Calculate volatility intensity
            volatility_ratio = (
                context.atr / context.current_price
                if context.current_price > 0
                else 0.0
            )
            vol_threshold = 0.005

            if context.pnl > 0:
                # Bonus for making profit in high volatility
                if volatility_ratio > vol_threshold:
                    smart_multiplier *= 1.1

        final_reward = base_reward * smart_multiplier

        return final_reward
        final_reward = base_reward * smart_multiplier

        return final_reward
