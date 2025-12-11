"""
Realistic Execution Model.

Implements volatility-adjusted slippage and latency simulation.
"""

import random
from typing import Optional

import numpy as np

from ztb.trading.execution.model import ExecutionModel, ExecutionResult
from ztb.utils.fee_model import FeeModel, FixedFeeModel
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealisticExecutionModel(ExecutionModel):
    """
    Simulates realistic execution conditions including:
    - Volatility-based slippage (ATR)
    - Random network/processing latency
    - Market impact (simplified)
    """

    def __init__(
        self,
        base_slippage: float = 0.0005,  # 0.05% base slippage
        atr_slippage_factor: float = 0.5,  # Multiplier for ATR-based slippage
        base_latency_ms: float = 50.0,
        latency_jitter_ms: float = 20.0,
        fee_model: Optional[FeeModel] = None,
        seed: Optional[int] = None,
    ):
        self.base_slippage = base_slippage
        self.atr_slippage_factor = atr_slippage_factor
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.fee_model = fee_model or FixedFeeModel()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def simulate_execution(
        self,
        action_type: str,
        requested_price: float,
        requested_size: float,
        current_atr: float = 0.0,
        current_volume: float = 0.0,
        market_regime: Optional[str] = None,
    ) -> ExecutionResult:
        # 1. Calculate Latency
        # Simple uniform jitter
        latency = self.base_latency_ms + random.uniform(
            -self.latency_jitter_ms, self.latency_jitter_ms
        )
        latency = max(0.0, latency)

        # 2. Calculate Slippage
        # Base slippage
        slippage_rate = self.base_slippage

        # Volatility adjustment (if ATR provided)
        if current_atr > 0 and requested_price > 0:
            # Normalized ATR (volatility percentage)
            volatility_pct = current_atr / requested_price
            # Add volatility component to slippage
            slippage_rate += volatility_pct * self.atr_slippage_factor

        # Directional slippage (always against the trader)
        # Buy: Price goes UP (executed > requested)
        # Sell: Price goes DOWN (executed < requested)
        if action_type.lower() == "buy":
            executed_price = requested_price * (1 + slippage_rate)
        else:
            executed_price = requested_price * (1 - slippage_rate)

        # 3. Calculate Fee
        trade_value = requested_size * executed_price
        fee = self.fee_model.calculate_fee(trade_value, action_type)

        return ExecutionResult(
            executed_price=executed_price,
            executed_size=requested_size,  # Assuming full fill for now
            slippage_rate=slippage_rate,
            fee=fee,
            latency_ms=latency,
        )
