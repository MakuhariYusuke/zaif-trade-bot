"""
Replay market data source for backtesting and simulation.

Provides market data replay functionality for testing and validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ztb.data.marketdata_registry import MarketDataSource

logger = logging.getLogger(__name__)


class ReplayMarket(MarketDataSource):
    """Market data source that replays historical data."""

    def __init__(self, data_path: str, **kwargs):
        self.data_path = Path(data_path)
        self.speed_multiplier = kwargs.get("speed_multiplier", 1.0)
        self.loop = kwargs.get("loop", False)
        self._data: Optional[pd.DataFrame] = None
        self._current_index = 0

    def load_data(self) -> pd.DataFrame:
        """Load market data from file."""
        if self._data is None:
            if self.data_path.suffix == ".csv":
                self._data = pd.read_csv(self.data_path)
            elif self.data_path.suffix == ".json":
                self._data = pd.read_json(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

            # Ensure timestamp column exists
            if "timestamp" not in self._data.columns:
                if "date" in self._data.columns:
                    self._data["timestamp"] = pd.to_datetime(self._data["date"])
                else:
                    raise ValueError("Data must contain 'timestamp' or 'date' column")

            self._data = self._data.sort_values("timestamp").reset_index(drop=True)

        return self._data

    def get_data(self, **kwargs) -> pd.DataFrame:
        """Get market data for replay."""
        data = self.load_data()

        # Simple replay logic - return next batch
        batch_size = kwargs.get("batch_size", 100)
        if self._current_index + batch_size > len(data):
            if self.loop:
                self._current_index = 0
            else:
                # Return remaining data
                batch_size = len(data) - self._current_index

        if batch_size <= 0:
            return pd.DataFrame()

        batch = data.iloc[self._current_index : self._current_index + batch_size].copy()
        self._current_index += batch_size

        return batch

    def reset(self):
        """Reset replay to beginning."""
        self._current_index = 0

    def get_progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        if self._data is None:
            return 0.0
        return min(1.0, self._current_index / len(self._data))


# Factory function for registry
def create_replay_market(data_path: str, **kwargs) -> ReplayMarket:
    """Factory function to create replay market data source."""
    return ReplayMarket(data_path, **kwargs)
