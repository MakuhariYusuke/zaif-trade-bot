import math
from typing import Dict, Any, Optional, cast
from ztb.trading.execution.model import ExecutionModel, ExecutionResult
from ztb.trading.environment.constants import EPSILON
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
        self.c_spread = float(config.get('c_spread', 0.3))
        self.c_vol = float(config.get('c_vol', 0.2))
        self.c_imp = float(config.get('c_imp', 0.5))
        self.gamma = float(config.get('gamma', 0.5))
        self.min_volume = float(config.get('min_volume', 0.01))
        self.latency_sec = float(config.get('latency_sec', 1.0))
        
    def calculate_slippage_one_way(self, market_data: MarketState, order_size: float) -> float:
        """
        Calculate one-way slippage in JPY/BTC.
        """
        high = market_data.get('high', 0.0)
        low = market_data.get('low', 0.0)
        atr = market_data.get('atr', 0.0)
        volume = market_data.get('volume', 0.0)
        
        # 1. Spread Proxy
        spread_proxy = self.c_spread * (high - low)
        
        # 2. Volatility Risk
        vol_risk = self.c_vol * atr * math.sqrt(self.latency_sec / 60.0)
        
        # 3. Market Impact
        impact = self.c_imp * atr * ((order_size / max(volume, self.min_volume)) ** self.gamma)
        
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
        if action_type not in ('buy', 'sell'):
            raise ValueError(f"Invalid action_type: {action_type}. Must be 'buy' or 'sell'.")

        # Construct MarketState
        m_state: MarketState
        if market_data:
             m_state = cast(MarketState, market_data)
        else:
            # Construct from individual args (approximate if high/low missing)
            # If high/low are missing, spread proxy will be 0 unless we estimate it.
            # Use ATR to estimate High/Low range if available.
            estimated_half_range = 0.5 * current_atr if current_atr > 0 else 0.0
            
            m_state = {
                'atr': current_atr,
                'volume': current_volume,
                'high': requested_price + estimated_half_range,
                'low': requested_price - estimated_half_range,
                'close': requested_price,
                'timestamp': None
            }

        slippage_abs = self.calculate_slippage_one_way(m_state, requested_size)
        
        if action_type == 'buy':
            executed_price = requested_price + slippage_abs
        else:
            executed_price = requested_price - slippage_abs
            
        slippage_rate = slippage_abs / requested_price if requested_price > EPSILON else 0.0
        
        return ExecutionResult(
            executed_price=executed_price,
            executed_size=requested_size,
            slippage_rate=slippage_rate,
            fee=0.0, # Fee handled by FeeModel usually
            latency_ms=self.latency_sec * 1000.0,
            timestamp=m_state.get('timestamp')
        )
