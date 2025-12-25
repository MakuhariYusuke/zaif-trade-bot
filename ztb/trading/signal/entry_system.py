from typing import Dict, Any, Optional
from ztb.trading.signal.calibration_map import CalibrationGate, CalibrationMap
from ztb.trading.signal.types import FusedSignal, GateResult
from ztb.trading.types import MarketState

class IntegratedEntrySystem:
    """
    Integrated Entry System (v455).
    Combines RL signals, Pattern signals (optional), and CalibrationGate
    to make final entry decisions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Calibration Map & Gate
        self.calibration_map = CalibrationMap(config)
        self.gate = CalibrationGate(config, self.calibration_map)
        
    def process_signal(
        self, 
        rl_action: float, 
        market_data: MarketState, 
        regime: str,
        pattern_score: Optional[float] = None,
        order_size: Optional[float] = None
    ) -> GateResult:
        """
        Process a raw signal through the Calibration Gate.
        """
        fused_signal: FusedSignal = {
            'rl_action': rl_action,
            'regime': regime,
            'pattern_score': pattern_score
        }
        
        # Evaluate via Gate
        gate_result = self.gate.evaluate(fused_signal, market_data, order_size)
        
        return gate_result

    def update_outcome(self, regime: str, action: float, gross_pnl: float, step: int):
        """
        Update calibration stats with trade outcome.
        """
        self.calibration_map.update(regime, action, gross_pnl, step)
