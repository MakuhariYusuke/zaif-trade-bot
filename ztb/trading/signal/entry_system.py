from typing import Any, Dict, Optional

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
        threshold: float = 0.2,
        pattern_score: Optional[float] = None,
        order_size: Optional[float] = None,
    ) -> GateResult:
        """
        Process a raw signal through the Calibration Gate.
        """
        # Relative Binning Logic
        # Normalize action so that threshold maps to 0.2 (Buy boundary)
        denom = max(abs(threshold), 1e-6)
        ratio = rl_action / denom

        # If action is negative (Sell), ratio will be negative (assuming threshold is positive magnitude)
        # If threshold is passed as negative for Sell, ratio is positive.
        # We want Sell action to map to negative values.
        # So we should use abs(threshold) for denom.
        # If rl_action is negative, ratio is negative.

        normalized_action = ratio * 0.2

        # Clip for robustness (e.g. +/- 3.0 ratio -> +/- 0.6)
        # 3.0 * 0.2 = 0.6 (Strong Buy/Sell boundary)
        normalized_action = max(min(normalized_action, 0.8), -0.8)

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
        self, regime: str, action: float, gross_pnl: float, step: int, threshold: float
    ):
        """
        Update calibration stats with trade outcome.
        """
        # Relative Binning Logic
        denom = max(abs(threshold), 1e-6)
        ratio = action / denom
        normalized_action = ratio * 0.2
        normalized_action = max(min(normalized_action, 0.8), -0.8)

        self.calibration_map.update(regime, normalized_action, gross_pnl, step)

    def save_state(self, path: str):
        """Save calibration state to file."""
        import json

        state = self.calibration_map.get_state()
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load calibration state from file."""
        import json
        import os

        if os.path.exists(path):
            with open(path, "r") as f:
                state = json.load(f)
            self.calibration_map.load_state(state)
