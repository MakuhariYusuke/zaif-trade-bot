"""
Reward calculation components for trading environment.

This package contains modular reward calculation components that follow SOLID principles.
Each component is responsible for a specific aspect of reward calculation.
"""

from .action_penalty import ActionPenaltyCalculator
from .base_reward_calculator import BaseRewardCalculator
from .diversity_bonus import DiversityBonusCalculator
from .drawdown_penalty import DrawdownPenaltyCalculator
from .growth_bonus import GrowthBonusCalculator
from .metrics import LongTermMetrics
from .pnl_focused_reward import PnLFocusedRewardCalculator
from .position_penalty import PositionPenaltyCalculator
from .stagnation_penalty import StagnationPenaltyCalculator
from .trend_detector import TrendDetector
from .win_rate_bonus import WinRateBonusCalculator
from .win_streak_bonus import WinStreakBonusCalculator

__all__ = [
    "BaseRewardCalculator",
    "PnLFocusedRewardCalculator",
    "PositionPenaltyCalculator",
    "ActionPenaltyCalculator",
    "DiversityBonusCalculator",
    "WinRateBonusCalculator",
    "DrawdownPenaltyCalculator",
    "StagnationPenaltyCalculator",
    "GrowthBonusCalculator",
    "WinStreakBonusCalculator",
    "TrendDetector",
    "LongTermMetrics",
]
