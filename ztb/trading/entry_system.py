"""
Integrated Entry System for v458

Combines calibration gate with market state processing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class CalibrationMap:
    """Calibration map for entry decisions."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.data: Dict[str, Dict[str, float]] = {}
        self.load()

    def load(self):
        """Load calibration data from file."""
        if self.path and self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    self.data = json.load(f)
                logger.info(f"Loaded calibration map from {self.path}")
            except Exception as e:
                logger.warning(f"Failed to load calibration map: {e}")
                self.data = {}
        else:
            self.data = {}

    def update(self, regime: str, action: float, pnl: float, step: int):
        """Update calibration with trade outcome."""
        if regime not in self.data:
            self.data[regime] = {}
        # Simple update: accumulate pnl
        if 'total_pnl' not in self.data[regime]:
            self.data[regime]['total_pnl'] = 0.0
        self.data[regime]['total_pnl'] += pnl

    def get_threshold(self, regime: str) -> float:
        """Get entry threshold for regime."""
        if regime in self.data:
            total_pnl = self.data[regime].get('total_pnl', 0.0)
            # Simple threshold: allow if positive pnl
            return 0.0 if total_pnl > 0 else 1.0
        return -0.01  # Default allow for initial


    def load_state(self, path: str):
        """Load state from file."""
        self.calibration_map = CalibrationMap(path)

    def process_signal(self, rl_action: float, market_data: Any, regime: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Process entry signal."""
        if not self.enabled:
            return {"should_enter": True}

        # Check calibration threshold
        cal_threshold = self.calibration_map.get_threshold(regime)
        should_enter = abs(rl_action) > max(threshold, cal_threshold)

        return {"should_enter": should_enter}</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\ztb\trading\entry_system.py