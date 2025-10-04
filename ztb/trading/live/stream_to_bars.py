"""
Convert streaming trades to 1-minute bars.

Reusable utility for aggregating trade data into OHLCV bars.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class StreamToBarsConverter:
    """Convert streaming trade data to OHLCV bars."""

    def __init__(self, bar_interval: str = "1min") -> None:
        """
        Initialize converter.

        Args:
            bar_interval: Pandas frequency string (e.g., '1min', '5min')
        """
        self.bar_interval = bar_interval
        self.pending_trades: List[Dict[str, Any]] = []

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the pending list.

        Args:
            trade: Trade dict with keys: timestamp, price, quantity, side
        """
        required_keys = ["timestamp", "price", "quantity"]
        if not all(key in trade for key in required_keys):
            raise ValueError(f"Trade missing required keys: {required_keys}")

        self.pending_trades.append(trade)

    def add_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)

    def convert_to_bars(self, symbol: str = "BTC_JPY") -> pd.DataFrame:
        """
        Convert pending trades to OHLCV bars.

        Args:
            symbol: Trading symbol for the bars

        Returns:
            DataFrame with OHLCV columns and datetime index
        """
        if not self.pending_trades:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Convert to DataFrame more efficiently
        df = pd.DataFrame(self.pending_trades)

        # Ensure timestamp is datetime (vectorized)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms" if isinstance(df["timestamp"].iloc[0], (int, float)) else None,
        )

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Sort by timestamp for proper resampling
        df = df.sort_index()

        # Resample to bars with optimized aggregation
        bars = df.resample(self.bar_interval).agg(
            {"price": ["first", "max", "min", "last"], "quantity": "sum"}
        )

        # Flatten column names
        bars.columns = ["open", "high", "low", "close", "volume"]

        # Add symbol column
        bars["symbol"] = symbol

        # Remove any NaN bars (periods with no trades) - more efficient
        bars = bars.dropna(subset=["open"])

        # Clear pending trades after conversion
        self.pending_trades.clear()

        return bars

    def get_current_bar(
        self, current_time: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current incomplete bar.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Current bar dict or None if no trades in current period
        """
        if not self.pending_trades:
            return None

        if current_time is None:
            current_time = datetime.now()

        # Find trades in current bar period
        current_bar_start = current_time.replace(second=0, microsecond=0)
        if self.bar_interval == "1min":
            pass  # already aligned
        elif self.bar_interval == "5min":
            current_bar_start = current_bar_start.replace(
                minute=current_bar_start.minute // 5 * 5
            )

        current_trades = [
            trade
            for trade in self.pending_trades
            if pd.to_datetime(trade["timestamp"]) >= current_bar_start
        ]

        if not current_trades:
            return None

        prices = [t["price"] for t in current_trades]
        volume = sum(t["quantity"] for t in current_trades)

        return {
            "timestamp": current_bar_start,
            "open": current_trades[0]["price"],
            "high": max(prices),
            "low": min(prices),
            "close": current_trades[-1]["price"],
            "volume": volume,
            "symbol": "BTC_JPY",  # Default
        }

    def clear_pending(self) -> None:
        """Clear all pending trades."""
        self.pending_trades.clear()

    def get_pending_count(self) -> int:
        """Get count of pending trades."""
        return len(self.pending_trades)


def trades_to_bars(
    trades: List[Dict[str, Any]], bar_interval: str = "1min", symbol: str = "BTC_JPY"
) -> pd.DataFrame:
    """
    Convenience function to convert trades to bars in one call.

    Args:
        trades: List of trade dicts
        bar_interval: Bar interval (e.g., '1min')
        symbol: Trading symbol

    Returns:
        OHLCV DataFrame
    """
    converter = StreamToBarsConverter(bar_interval)
    converter.add_trades(trades)
    return converter.convert_to_bars(symbol)
