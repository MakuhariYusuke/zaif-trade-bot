"""
Execution Model Interface.

Defines the interface for simulating trade execution, including slippage,
latency, and market impact.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    """Result of an execution simulation."""

    executed_price: float
    executed_size: float
    slippage_rate: float
    fee: float
    latency_ms: float
    timestamp: float | None = None

class ExecutionModel(ABC):
    """Abstract base class for execution models."""

    @abstractmethod
    def simulate_execution(
        self,
        action_type: str,  # "buy" or "sell"
        requested_price: float,
        requested_size: float,
        current_atr: float = 0.0,
        current_volume: float = 0.0,
        market_regime: str | None = None,
    ) -> ExecutionResult:
        """
        Simulate the execution of a trade.

        Args:
            action_type: "buy" or "sell"
            requested_price: The theoretical price at the time of decision (e.g. close price)
            requested_size: The size of the order
            current_atr: Current Average True Range (volatility metric)
            current_volume: Current trading volume
            market_regime: Current market regime (e.g. "trending", "ranging")

        Returns:
            ExecutionResult containing the actual executed price and details.
        """
        pass
