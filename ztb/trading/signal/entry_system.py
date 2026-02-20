from typing import Any, Dict, Optional

from ztb.io.common import PathLike
from ztb.io.state_persistence import read_state_payload, write_state_payload
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

    @staticmethod
    def _normalize_action(action: float, threshold: float) -> float:
        """Normalize action into the calibration map range."""
        denom = max(abs(threshold), 1e-6)
        normalized_action = (action / denom) * 0.2
        return max(min(normalized_action, 0.8), -0.8)

    def process_signal(
        self,
        rl_action: float,
        market_data: MarketState,
        regime: str,
        threshold: float = 0.2,
        pattern_score: Optional[float] = None,
        order_size: Optional[float] = None,
    ) -> GateResult:
        """
        Process a raw signal through the Calibration Gate.
        """
        normalized_action = self._normalize_action(rl_action, threshold)

        fused_signal: FusedSignal = {
            "rl_action": normalized_action,
            "regime": regime,
            "pattern_score": pattern_score,
        }

        # Evaluate via Gate
        gate_result = self.gate.evaluate(fused_signal, market_data, order_size)

        # Add normalized action to result for debugging/logging
        gate_result["normalized_action"] = normalized_action

        return gate_result

    def update_outcome(
        self,
        regime: str,
        action: float,
        gross_pnl: float,
        step: int,
        threshold: float = 0.2,
    ) -> None:
        """
        Update calibration stats with trade outcome.
        """
        normalized_action = self._normalize_action(action, threshold)

        self.calibration_map.update(regime, normalized_action, gross_pnl, step)

    def save_state(self, path: PathLike) -> None:
        """Save calibration state to file."""
        state = self.calibration_map.get_state()
        write_state_payload(path, state)

    def load_state(self, path: PathLike) -> bool:
        """Load calibration state from file."""
        try:
            state = read_state_payload(path)
        except FileNotFoundError:
            return False
        self.calibration_map.load_state(state)
        return True
