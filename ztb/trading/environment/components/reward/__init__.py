"""
Reward calculation components for trading environment.

Active components used by the production RewardCalculator pipeline.
Orphaned legacy components (action_penalty, diversity_bonus, etc.) were
archived to archived/dead_reward_components/ in 407#.
"""

from .balance_curriculum import BalanceCurriculumManager
from .metrics import LongTermMetrics
from .mtf_weight_manager import MTFWeightManager
from .opportunity_cost_penalty_calculator import OpportunityCostPenaltyCalculator
from .trend_detector import TrendDetector
from .unrealized_loss_penalty_calculator import UnrealizedLossPenaltyCalculator

__all__ = [
    "BalanceCurriculumManager",
    "LongTermMetrics",
    "MTFWeightManager",
    "OpportunityCostPenaltyCalculator",
    "TrendDetector",
    "UnrealizedLossPenaltyCalculator",
]
