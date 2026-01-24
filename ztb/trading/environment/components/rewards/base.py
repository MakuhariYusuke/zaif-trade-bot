from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ztb.trading.environment.components.behavioral_penalty_calculator import (
        BehavioralPenaltyCalculator,
    )
    from ztb.trading.environment.components.interfaces import (
        IDynamicRewardShaper,
        IMarketRegimeDetector,
    )
    from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


@dataclass
class RewardContext:
    """Context object passed to reward components."""

    action: int
    current_price: float
    position: float
    portfolio_value: float
    atr: float
    transaction_cost: float
    reward_scaling: float
    pnl: float
    old_position: float
    step: int
    observation: Optional[np.ndarray]
    reward_history: List[float]
    portfolio_value_history: List[float]
    config: "EnvironmentConfig"
    reward_settings: Optional["RewardSettings"] = None
    initial_portfolio_value: float = 1000000.0

    # Optional fields for specific components
    continuous_action_value: Optional[float] = None
    atr_normalised: float = 0.0
    portfolio_return: float = 0.0
    effective_max_position: float = 1.0
    action_counts: List[int] = field(default_factory=list)
    recent_actions: List[int] = field(default_factory=list)
    target_ratios: Dict[str, float] = field(default_factory=dict)
    behavioral_penalty_calculator: Optional["BehavioralPenaltyCalculator"] = None
    market_regime_detector: Optional["IMarketRegimeDetector"] = None
    dynamic_reward_shaper: Optional["IDynamicRewardShaper"] = None


class RewardComponent(ABC):
    """Base class for reward calculation components."""

    @abstractmethod
    def calculate(self, context: RewardContext) -> float:
        """Calculate the reward contribution."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the component for logging."""
        pass

    def _get_setting(self, context: RewardContext, key: str, default, cast=None):
        """Utility to retrieve a reward setting with fallbacks.

        - Checks `context.reward_settings` first (supports dict-like or object).
        - If not found, checks `context.reward_settings.custom_reward_params` if present.
        - Falls back to `context.config.get(key, default)` when available.
        - Optionally casts the result with `cast` (e.g., float, int).
        """
        print(f"DEBUG base _get_setting called with key={key}")
        if context.reward_settings:
            val = None
            # If reward_settings behaves like a dict or has a `get` method, prefer that
            if isinstance(context.reward_settings, dict):
                val = context.reward_settings.get(key)
            else:
                get_attr = getattr(context.reward_settings, "get", None)
                if callable(get_attr):
                    try:
                        val = get_attr(key, None)
                    except Exception:
                        val = None
                else:
                    val = getattr(context.reward_settings, key, None)

            if val is None:
                custom_params = getattr(context.reward_settings, "custom_reward_params", None)
                if isinstance(custom_params, dict):
                    val = custom_params.get(key)
                    print(f"DEBUG base: key={key}, val from custom_params={val}")

            if val is not None:
                try:
                    return cast(val) if cast is not None else val
                except (ValueError, TypeError):
                    pass

        if hasattr(context.config, "get"):
            try:
                val = context.config.get(key, default)
                return cast(val) if cast is not None else val
            except (ValueError, TypeError):
                return default

        return default

    def _get_setting_float(self, context: RewardContext, key: str, default: float) -> float:
        """Typed float helper that delegates to `_get_setting` with casting."""
        try:
            return self._get_setting(context, key, default, cast=float)
        except (ValueError, TypeError):
            return default

    def _get_setting_int(self, context: RewardContext, key: str, default: int) -> int:
        """Typed int helper that delegates to `_get_setting` with casting."""
        print(f"DEBUG base _get_setting_int called with key={key}")
        try:
            return self._get_setting(context, key, default, cast=int)
        except (ValueError, TypeError):
            return default
