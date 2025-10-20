# Type definitions for trading environment
# 取引環境の型定義

from typing import TYPE_CHECKING, List

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

# Type aliases for better type safety
Observation = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
Action = int
Reward = float


class InfoDict(TypedDict, total=False):
    """Info dictionary returned by environment step/reset.

    Extends gymnasium's standard info dict with trading-specific information.
    """

    # Gymnasium standard fields (inherited)
    # current_step: int
    # total_steps: int

    # Trading-specific fields
    position: float
    total_pnl: float
    trades_count: int
    features: List[str]
    config: "EnvironmentConfig"  # Forward reference to avoid circular import
    pnl: float
    action: int
    step: int
    portfolio_value: float
    atr: float
    position_utilisation: float
    action_masks: NDArray[np.bool_]


class StatisticsDict(TypedDict, total=False):
    """Statistics dictionary returned by get_statistics."""

    total_reward: float
    mean_reward: float
    std_reward: float
    sharpe_ratio: float
    max_reward: float
    total_trades: int
    win_rate: float


# Type alias for backward compatibility
Info = InfoDict


# Constants
EPSILON = 1e-6  # Small value for division by zero prevention


if TYPE_CHECKING:
    from ztb.trading.environment.utils.config import EnvironmentConfig
