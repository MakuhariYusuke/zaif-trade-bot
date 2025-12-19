import logging
from typing import Any, Optional

from .base import RewardComponent, RewardContext


class ConfidencePenaltyReward(RewardComponent):
    """
    Penalizes high confidence actions that result in a loss.
    Uses a Hinge Loss formulation: penalty applies only when confidence exceeds a threshold.
    
    Formula:
        Penalty = -1.0 * LossMagnitude * (AbsAction - Threshold) * Factor
        
    Where:
        LossMagnitude = abs(ATR_Normalised_PnL)
        AbsAction = abs(continuous_action_value)
        Threshold = confidence_penalty_threshold (default 0.05)
        Factor = confidence_penalty_factor (default 1.0)
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def get_name(self) -> str:
        return "confidence_penalty"

    def _get_setting(self, context: RewardContext, key: str, default: float) -> float:
        """Local behavior: only check reward_settings (dict or object) and custom_reward_params.
        Do NOT fall back to context.config to avoid MagicMock config surprises in tests.
        """
        if not context.reward_settings:
            return default

        # Try to get from dictionary or object
        val = None
        if isinstance(context.reward_settings, dict):
            val = context.reward_settings.get(key)
        else:
            val = getattr(context.reward_settings, key, None)

        # If not found, check custom_reward_params if it exists
        if val is None:
            custom_params = getattr(context.reward_settings, "custom_reward_params", None)
            if isinstance(custom_params, dict):
                val = custom_params.get(key)

        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

        return default

    def calculate(self, context: RewardContext) -> float:
        # Only penalize if there is a loss
        if context.pnl >= 0:
            return 0.0
            
        if context.continuous_action_value is None:
            return 0.0

        # Get settings
        # Default threshold lowered to 0.05 (was 0.1 in previous implementation)
        threshold = self._get_setting(context, "confidence_penalty_threshold", 0.05)
        factor = self._get_setting(context, "confidence_penalty_factor", 1.0)

        abs_action = abs(context.continuous_action_value)
        
        if abs_action <= threshold:
            return 0.0
            
        # Calculate loss magnitude
        # Use atr_normalised if available and meaningful
        # context.atr is passed in context
        if context.atr > 1e-8 and context.atr != 1.0:
             loss_magnitude = abs(context.atr_normalised)
        else:
             # Fallback if ATR is not reliable
             loss_magnitude = abs(context.portfolio_return) * 100
             
        # Hinge calculation: proportional to how much we exceeded the threshold
        excess_confidence = abs_action - threshold
        
        penalty = -1.0 * loss_magnitude * excess_confidence * factor
        
        # Log debug info if penalty is significant
        if abs(penalty) > 1e-4:
            self.logger.debug(
                f"Confidence Penalty: action={context.continuous_action_value:.4f}, "
                f"loss_mag={loss_magnitude:.4f}, excess={excess_confidence:.4f}, "
                f"penalty={penalty:.4f}"
            )
        
        return penalty


