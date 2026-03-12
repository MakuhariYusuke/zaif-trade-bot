from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

@dataclass
class MarketData:
    symbol: str
    timeframe: str
    data: Any
    start_date: datetime | None = None
    end_date: datetime | None = None

    def validate_data(self) -> bool:
        # Basic validation: required OHLC fields and sanity checks
        required_columns = ["open", "high", "low", "close"]
        if not isinstance(self.data, pd.DataFrame):
            return False
        if not all(col in self.data.columns for col in required_columns):
            return False

        # Auto-fix rows where high < low by swapping; otherwise normalize
        try:
            mask = self.data["high"] < self.data["low"]
            if mask.any():
                # Swap
                tmp_high = self.data.loc[mask, "high"].copy()
                self.data.loc[mask, "high"] = self.data.loc[mask, "low"]
                self.data.loc[mask, "low"] = tmp_high
        except Exception:
            return False

        # Detect constant series with no volatility
        try:
            if (self.data["high"] == self.data["low"]).all() and (
                self.data["open"] == self.data["close"]
            ).all():
                return False
        except Exception:
            return False

        # Normalize OHLC: ensure high >= max(open, close) and low <= min(open, close)
        try:
            self.data["high"] = self.data[["open", "high", "close"]].max(axis=1)
            self.data["low"] = self.data[["open", "low", "close"]].min(axis=1)
        except Exception:
            # Keep best-effort
            pass

        return True

@dataclass
class TradeRecord:
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float

    @property
    def trade_value(self) -> float:
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        return self.trade_value + (self.commission or 0.0)

    @property
    def pnl(self) -> float:
        # default placeholder, may be set on object
        return getattr(self, "_pnl", 0.0)

__all__ = ["MarketData", "TradeRecord"]
