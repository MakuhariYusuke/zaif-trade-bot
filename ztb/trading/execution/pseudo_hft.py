import math
from typing import Any, Dict, Optional, cast

from ztb.trading.environment.constants import EPSILON
from ztb.trading.execution.model import ExecutionModel, ExecutionResult
from ztb.trading.types import MarketState


class PseudoHFTExecutionModel(ExecutionModel):
    """
    Pseudo-HFT Execution Model (v455).
    Implements Taker-based execution with detailed slippage components:
    - Spread Proxy
    - Volatility Risk
    - Market Impact
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.c_spread = float(config.get("c_spread", 0.3))
        self.c_vol = float(config.get("c_vol", 0.2))
        self.c_imp = float(config.get("c_imp", 0.5))
        self.gamma = float(config.get("gamma", 0.5))
        self.min_volume = float(config.get("min_volume", 0.01))
        self.latency_sec = float(config.get("latency_sec", 1.0))

    def calculate_slippage_one_way(
        self, market_data: MarketState, order_size: float
    ) -> float:
        """
        Calculate one-way slippage in JPY/BTC.
        """
        high = market_data.get("high", 0.0)
        low = market_data.get("low", 0.0)
        atr = market_data.get("atr", 0.0)
        volume = market_data.get("volume", 0.0)
        close = market_data.get("close", 0.0)

        # Robustness: Handle NaN/Inf
        if not math.isfinite(high): high = 0.0
        if not math.isfinite(low): low = 0.0
        if not math.isfinite(atr): atr = 0.0
        if not math.isfinite(volume): volume = 0.0
        if not math.isfinite(close): close = 0.0

        # Fallback for ATR if missing/zero
        if atr <= EPSILON:
             # Fallback: 0.05% of close price if available
             if close > EPSILON:
                 atr = close * 0.0005
        
        # Fallback for Volume if missing/zero
        if volume <= EPSILON:
             # Assume min_volume to maximize impact
             volume = self.min_volume

        # 1. Spread Proxy
        spread_val = max(high - low, 0.0)
        # If spread is 0 (e.g. high/low missing) but we have ATR, use ATR-based proxy
        if spread_val <= EPSILON and atr > EPSILON:
            spread_val = atr # Conservative proxy: spread ~ ATR? Or 0.5 ATR?
            # In simulate_execution we used 0.5*ATR for half-range, so high-low = ATR.
            # So spread_val = ATR is consistent.
        
        spread_proxy = self.c_spread * spread_val

        # 2. Volatility Risk
        vol_risk = self.c_vol * atr * math.sqrt(self.latency_sec / 60.0)

        # 3. Market Impact
        # Use abs(order_size) to handle negative sizes (sells) correctly
        impact = (
            self.c_imp
            * atr
            * ((abs(order_size) / max(volume, self.min_volume)) ** self.gamma)
        )

        return spread_proxy + vol_risk + impact

    def simulate_execution(
        self,
        action_type: str,
        requested_price: float,
        requested_size: float,
        current_atr: float = 0.0,
        current_volume: float = 0.0,
        market_regime: Optional[str] = None,
        # Optional: Allow passing full market state if available, but keep signature compatible
        market_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        # Validate action_type
        if action_type not in ("buy", "sell"):
            raise ValueError(
                f"Invalid action_type: {action_type}. Must be 'buy' or 'sell'."
            )

        # Construct MarketState
        m_state: MarketState
        if market_data:
            m_state = cast(MarketState, market_data)
        else:
            # Construct from individual args (approximate if high/low missing)
            # If high/low are missing, spread proxy will be 0 unless we estimate it.
            # Use ATR to estimate High/Low range if available.

            # If current_atr is 0 (missing), use a small fallback to avoid optimistic 0-cost fills
            effective_atr = (
                current_atr
                if current_atr > EPSILON
                else (requested_price * 0.0005 if requested_price > 0 else 0.0)
            )

            estimated_half_range = 0.5 * effective_atr

            m_state = {
                "atr": effective_atr,
                "volume": current_volume,
                "high": requested_price + estimated_half_range,
                "low": requested_price - estimated_half_range,
                "close": requested_price,
                "timestamp": None,
            }

        slippage_abs = self.calculate_slippage_one_way(m_state, requested_size)

        if action_type == "buy":
            executed_price = requested_price + slippage_abs
        else:
            executed_price = requested_price - slippage_abs

        slippage_rate = (
            slippage_abs / requested_price if requested_price > EPSILON else 0.0
        )

        return ExecutionResult(
            executed_price=executed_price,
            executed_size=requested_size,
            slippage_rate=slippage_rate,
            fee=0.0,  # Fee handled by FeeModel usually
            latency_ms=self.latency_sec * 1000.0,
            timestamp=m_state.get("timestamp"),
        )
