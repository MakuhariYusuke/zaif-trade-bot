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
