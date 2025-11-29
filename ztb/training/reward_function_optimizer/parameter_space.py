"""
Reward Function Parameter Space Manager

Manages parameter spaces for different reward function optimization stages.
Separated from the main optimizer to follow Single Responsibility Principle.
"""

from typing import Any, Dict

from ztb.training.hyperparameter_optimizer import ParameterSpace
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RewardFunctionParameterSpace:
    """
    Manages parameter spaces for reward function optimization.

    Responsibilities:
    - Defining parameter spaces for different stages
    - Providing parameter space configurations
    - Validating parameter ranges
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def get_parameter_spaces(self) -> Dict[str, Dict[str, ParameterSpace]]:
        """
        Get all parameter spaces for different reward function stages.

        Returns:
            Dictionary of parameter spaces by stage
        """
        return {
            "balanced_transition": self._get_balanced_transition_space(),
            "trading_focused": self._get_trading_focused_space(),
            "profit_optimized": self._get_profit_optimized_space(),
            "ultra_profit": self._get_ultra_profit_space(),
            "bull_market": self._get_bull_market_space(),
            "bear_market": self._get_bear_market_space(),
            "sideways_market": self._get_sideways_market_space(),
            "high_volatility": self._get_high_volatility_space(),
        }

    def _get_balanced_transition_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for balanced transition stage."""
        return {
            # Basic trading parameters with narrower, more realistic ranges
            "balance_penalty_tolerance": ParameterSpace(
                "balance_penalty_tolerance", "float", 0.02, 0.1, log_scale=False
            ),
            "balance_penalty": ParameterSpace(
                "balance_penalty", "float", 2.0, 10.0, log_scale=False
            ),
            "hold_penalty_rate": ParameterSpace(
                "hold_penalty_rate", "float", 0.005, 0.05, log_scale=True
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 1.2, 3.0, log_scale=False
            ),
            "trading_bonus": ParameterSpace(
                "trading_bonus", "float", 0.005, 0.03, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD] - asymmetric ranges
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 0.8, 1.5, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 0.8, 1.5, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.2, 0.8, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses - narrower ranges
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 1.0, 2.5, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 1.0, 2.5, log_scale=False
            ),
            # Advanced reward components - market-aware weights
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.1, 0.7, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.1, 0.6, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.05, 0.3, log_scale=False
            ),
            # Multi-objective weights - normalized ranges
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.3, 1.2, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.2, 0.8, log_scale=False
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.3, 1.0, log_scale=False
            ),
            # Asymmetric reward scaling parameters (v435 enhancement)
            "long_position_reward_multiplier": ParameterSpace(
                "long_position_reward_multiplier", "float", 1.0, 2.0, log_scale=False
            ),
            "short_position_reward_multiplier": ParameterSpace(
                "short_position_reward_multiplier", "float", 0.5, 1.0, log_scale=False
            ),
            "long_position_penalty_multiplier": ParameterSpace(
                "long_position_penalty_multiplier", "float", 0.5, 1.0, log_scale=False
            ),
            "short_position_penalty_multiplier": ParameterSpace(
                "short_position_penalty_multiplier", "float", 1.0, 1.5, log_scale=False
            ),
        }

    def _get_trading_focused_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for trading focused stage."""
        return {
            # Basic trading parameters
            "balance_penalty_tolerance": ParameterSpace(
                "balance_penalty_tolerance", "float", 0.01, 0.2, log_scale=False
            ),
            "balance_penalty": ParameterSpace(
                "balance_penalty", "float", 5.0, 50.0, log_scale=False
            ),
            "hold_penalty_rate": ParameterSpace(
                "hold_penalty_rate", "float", 0.01, 1.0, log_scale=True
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 1.0, 10.0, log_scale=False
            ),
            "trading_bonus": ParameterSpace(
                "trading_bonus", "float", 0.01, 1.0, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD]
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 0.8, 3.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 0.8, 3.0, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.05, 1.0, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 1.0, 5.0, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 1.0, 5.0, log_scale=False
            ),
            # Advanced reward components
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.0, 1.0, log_scale=False
            ),
            # Multi-objective weights
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.1, 2.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.1, 2.0, log_scale=False
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.1, 2.0, log_scale=False
            ),
        }

    def _get_profit_optimized_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for profit optimized stage."""
        return {
            # Profit-focused parameters
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.5, 5.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.01, 1.0, log_scale=True
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.01, 1.0, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD]
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 1.0, 5.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 1.0, 5.0, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.01, 1.0, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 2.0, 8.0, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 2.0, 8.0, log_scale=False
            ),
            # Advanced reward components
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.0, 3.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 3.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.0, 1.5, log_scale=False
            ),
            # Risk management parameters
            "position_penalty_weight": ParameterSpace(
                "position_penalty_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "drawdown_penalty_weight": ParameterSpace(
                "drawdown_penalty_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "stagnation_penalty_weight": ParameterSpace(
                "stagnation_penalty_weight", "float", 0.001, 0.1, log_scale=True
            ),
            # Performance bonuses
            "growth_bonus_weight": ParameterSpace(
                "growth_bonus_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "win_streak_bonus_weight": ParameterSpace(
                "win_streak_bonus_weight", "float", 0.001, 0.1, log_scale=True
            ),
            # Asymmetric reward scaling parameters (v435 enhancement)
            "long_position_reward_multiplier": ParameterSpace(
                "long_position_reward_multiplier", "float", 1.0, 3.0, log_scale=False
            ),
            "short_position_reward_multiplier": ParameterSpace(
                "short_position_reward_multiplier", "float", 0.3, 1.0, log_scale=False
            ),
            "long_position_penalty_multiplier": ParameterSpace(
                "long_position_penalty_multiplier", "float", 0.3, 1.0, log_scale=False
            ),
            "short_position_penalty_multiplier": ParameterSpace(
                "short_position_penalty_multiplier", "float", 1.0, 2.0, log_scale=False
            ),
        }

    def _get_ultra_profit_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for ultra profit stage."""
        return {
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 1.0, 10.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.001, 0.1, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD]
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 2.0, 10.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 2.0, 10.0, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.001, 0.5, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 3.0, 15.0, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 3.0, 15.0, log_scale=False
            ),
            # Advanced reward components
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.0, 5.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 5.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "ultra_profit_multiplier": ParameterSpace(
                "ultra_profit_multiplier", "float", 1.0, 5.0, log_scale=False
            ),
            "ultra_risk_multiplier": ParameterSpace(
                "ultra_risk_multiplier", "float", 0.1, 2.0, log_scale=False
            ),
            # Asymmetric reward scaling parameters (v435 enhancement)
            "long_position_reward_multiplier": ParameterSpace(
                "long_position_reward_multiplier", "float", 1.0, 4.0, log_scale=False
            ),
            "short_position_reward_multiplier": ParameterSpace(
                "short_position_reward_multiplier", "float", 0.2, 1.0, log_scale=False
            ),
            "long_position_penalty_multiplier": ParameterSpace(
                "long_position_penalty_multiplier", "float", 0.2, 1.0, log_scale=False
            ),
            "short_position_penalty_multiplier": ParameterSpace(
                "short_position_penalty_multiplier", "float", 1.0, 3.0, log_scale=False
            ),
        }

    def _get_bull_market_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for bull market optimization."""
        return {
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 1.0, 8.0, log_scale=False
            ),
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 1.0, 5.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 1.5, 8.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 0.5, 3.0, log_scale=False
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 2.0, 8.0, log_scale=False
            ),
        }

    def _get_bear_market_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for bear market optimization."""
        return {
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.5, 4.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.5, 3.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 1.0, 4.0, log_scale=False
            ),
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 0.1, 2.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 1.0, 6.0, log_scale=False
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 1.0, 5.0, log_scale=False
            ),
        }

    def _get_sideways_market_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for sideways market optimization."""
        return {
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 1.0, 5.0, log_scale=False
            ),
            "hold_penalty_rate": ParameterSpace(
                "hold_penalty_rate", "float", 0.001, 0.05, log_scale=True
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.5, 2.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.1, 1.0, log_scale=False
            ),
            "stagnation_penalty_weight": ParameterSpace(
                "stagnation_penalty_weight", "float", 0.001, 0.05, log_scale=True
            ),
        }

    def _get_high_volatility_space(self) -> Dict[str, ParameterSpace]:
        """Get parameter space for high volatility market optimization."""
        return {
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 2.0, 8.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 1.0, 5.0, log_scale=False
            ),
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 3.0, 12.0, log_scale=False
            ),
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 1.0, 6.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 1.0, 6.0, log_scale=False
            ),
        }

    def create_parameter_space_from_config(
        self, config: Dict[str, Any], exploration_range: float = 0.1
    ) -> Dict[str, ParameterSpace]:
        """
        Create parameter space from existing configuration values.

        Args:
            config: Configuration dictionary with parameter values
            exploration_range: Fraction of current value to explore (±range)

        Returns:
            Parameter space dictionary
        """
        parameter_space = {}

        for param_name, param_value in config.items():
            if isinstance(param_value, (int, float)):
                # Calculate exploration bounds
                if param_value == 0:
                    # For zero values, use small absolute range
                    low = -0.1
                    high = 0.1
                else:
                    # Calculate percentage-based range
                    range_value = abs(param_value) * exploration_range
                    low = param_value - range_value
                    high = param_value + range_value

                    # Ensure low < high (important for negative values)
                    if low > high:
                        low, high = high, low

                    # For very small ranges, ensure minimum spread
                    if abs(high - low) < 1e-6:
                        center = (low + high) / 2
                        spread = max(abs(center) * 0.01, 1e-6)
                        low = center - spread
                        high = center + spread

                # Special handling for certain parameters
                if param_name in ["reward_clip_min", "reward_clip_max"]:
                    # These can be negative, ensure proper ordering
                    if param_name == "reward_clip_min":
                        # reward_clip_min should be <= reward_clip_max
                        # For min, allow more negative values
                        low = min(low, param_value * 1.5)  # Allow 50% more negative
                        high = min(high, config.get("reward_clip_max", param_value))
                    elif param_name == "reward_clip_max":
                        # For max, allow more positive values
                        low = max(low, config.get("reward_clip_min", param_value))
                        high = max(high, param_value * 1.5)  # Allow 50% more positive

                # Determine parameter type and constraints
                if isinstance(param_value, int) or param_name in [
                    "batch_size",
                    "buffer_size",
                    "learning_starts",
                    "target_update_interval",
                ]:
                    # Integer parameters
                    parameter_space[param_name] = ParameterSpace(
                        param_name,
                        "int",
                        max(1, int(low))
                        if param_name not in ["reward_clip_min", "reward_clip_max"]
                        else int(low),
                        int(high),
                    )
                else:
                    # Float parameters
                    parameter_space[param_name] = ParameterSpace(
                        param_name,
                        "float",
                        low,
                        high,
                        log_scale=param_name
                        in ["learning_rate", "ent_coef"],  # Use log scale for rates
                    )

        return parameter_space
