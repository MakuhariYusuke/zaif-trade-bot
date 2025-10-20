"""
Trade execution management for live trading bot.
"""
import logging
from typing import Any, Dict, Protocol

import numpy as np

from ztb.trading.environment.constants import continuous_to_discrete_action

logger = logging.getLogger(__name__)

# Action constants
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_NAMES = ["HOLD", "BUY", "SELL"]


class TradeExecutorProtocol(Protocol):
    """Protocol for trade executors."""

    def execute_trade(self, side: str, amount: float) -> bool:
        """Execute a trade."""
        ...


class PositionManagerProtocol(Protocol):
    """Protocol for position managers."""

    def update_position(self, action: int, current_price: float) -> None:
        """Update position based on action."""
        ...


class TradeExecutor:
    """Handles trade execution and position management."""

    def __init__(
        self,
        config: Dict[str, Any],
        trade_executor: TradeExecutorProtocol,
        position_manager: PositionManagerProtocol,
        is_sac: bool = False,
    ) -> None:
        self.config = config
        self.trade_executor = trade_executor
        self.position_manager = position_manager
        self._is_sac = is_sac

    def should_trade_sell_bias(self, action: int) -> bool:
        """Determine if we should trade based on sell bias strategy."""
        # Apply sell bias
        if action == ACTION_SELL:
            # Convert 40% of SELL predictions to BUY when position is flat to promote buying
            if np.random.random() < 0.4:
                action = ACTION_BUY
                logger.info("Converted SELL prediction to BUY for balance")
                return True

        # Basic trading logic - trade on BUY/SELL, hold on HOLD
        return action in [ACTION_BUY, ACTION_SELL]

    def execute_trade(self, side: str, amount: float) -> bool:
        """Execute a trade through the executor."""
        return self.trade_executor.execute_trade(side, amount)

    def update_position(self, action: int, current_price: float) -> None:
        """Update position through the position manager."""
        self.position_manager.update_position(action, current_price)

    def convert_action(self, raw_action: Any) -> int:
        """Convert model action to discrete action.

        Args:
            raw_action: Raw action from model (continuous for SAC, discrete for PPO)

        Returns:
            Discrete action (0=HOLD, 1=BUY, 2=SELL)
        """
        if self._is_sac:
            # SAC returns continuous actions, convert to discrete
            continuous_action = float(raw_action.item())  # Use item() for safety
            # Use the centralized continuous_to_discrete_action function for consistency
            return continuous_to_discrete_action(continuous_action)
        else:
            # PPO/MaskablePPO return discrete actions
            return int(raw_action.item())  # Use item() for consistency

    @property
    def is_sac(self) -> bool:
        """Check if using SAC model."""
        return self._is_sac

    @is_sac.setter
    def is_sac(self, value: bool) -> None:
        """Set SAC model flag."""
        self._is_sac = value
