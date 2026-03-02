"""
Price data management for live trading bot.
"""
import logging
from collections import deque
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

class PriceDataProvider(Protocol):
    """Protocol for price data providers."""

    def get_current_price(self) -> float:
        """Get current price."""
        ...

    def get_historical_prices(self, limit: int) -> list[float]:
        """Get historical prices."""
        ...

class PriceDataManager:
    """Manages price data and feature calculation."""

    def __init__(
        self, config: dict[str, Any], price_provider: PriceDataProvider
    ) -> None:
        self.config = cast(dict[str, Any], config)
        self.price_provider = price_provider
        self._price_history_max_size = config.get("price_history_length", 30)
        self.price_history: deque[float] = deque(maxlen=self._price_history_max_size)

    def update_price_history(self) -> None:
        """Update cached price history for technical indicators."""
        try:
            prices = self.price_provider.get_historical_prices(
                limit=self.config["price_history_length"]
            )
            # Convert list to deque
            self.price_history.clear()
            self.price_history.extend(prices)
            logger.info(
                f"Updated price history with {len(self.price_history)} data points"
            )
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")
            # Fallback to current price
            current_price = self.price_provider.get_current_price()
            self.price_history.clear()
            self.price_history.extend(
                [current_price] * self.config["price_history_length"]
            )

    def get_current_price(self) -> float:
        """Get current price."""
        return cast(float, self.price_provider.get_current_price())

    def calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        from ztb.features.generators.technical.momentum.rsi import compute_rsi

        df = pd.DataFrame({"close": prices})
        rsi_series = compute_rsi(df, period=period)
        last_val = rsi_series.iloc[-1]
        return float(last_val) if not pd.isna(last_val) else 50.0

    def calculate_sma(self, prices: list[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        from ztb.features.generators.technical.trend.sma import compute_sma

        df = pd.DataFrame({"close": prices})
        sma_series = compute_sma(df, period=period)
        last_val = sma_series.iloc[-1]
        return float(last_val) if not pd.isna(last_val) else 0.0

    def compute_live_features(self, prices: list[float]) -> dict[str, float]:
        """Compute live technical indicators."""
        if not prices:
            return {}

        current_price = prices[-1]

        # Basic technical indicators
        rsi = self.calculate_rsi(prices, period=14)
        sma_short = self.calculate_sma(prices, period=5)
        sma_long = self.calculate_sma(prices, period=20)

        # Price normalized
        price_norm = current_price / 1000000.0

        # Volume/quantity (mock for now)
        qty = np.random.uniform(0.001, 0.01)

        # PnL and win flag
        recent_prices = prices[-10:] if len(prices) >= 10 else prices
        if len(recent_prices) >= 2:
            pnl = (recent_prices[-1] - recent_prices[0]) * 0.001
            win = 1 if pnl > 0 else 0
        else:
            pnl = 0.0
            win = 0

        return {
            "rsi": rsi,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "price_norm": price_norm,
            "qty": qty,
            "pnl": pnl,
            "win": win,
        }

    def get_market_features(self) -> NDArray[np.floating]:
        """Get current market features for model prediction."""
        # Update price history with current price
        if self.price_history:
            current_price = self.get_current_price()
            self.price_history.append(current_price)
        else:
            # Initialize deque with current price
            current_price = self.get_current_price()
            self.price_history.clear()
            self.price_history.extend(
                [current_price] * self.config["price_history_length"]
            )

        # Convert deque to list for calculation functions
        price_list = list(self.price_history)

        # Get basic features
        features_dict = self.compute_live_features(price_list)

        # Start with basic features and extend to 68 dimensions
        features = [
            features_dict.get("rsi", 50.0),  # RSI (14-period)
            features_dict.get("sma_short", 0.0),  # Short SMA (5-period)
            features_dict.get("sma_long", 0.0),  # Long SMA (20-period)
            features_dict.get("price_norm", 0.0),  # Normalized price
            features_dict.get("qty", 0.0),  # Quantity
        ]

        # Extend to 68 features with zeros (simplified for live trading)
        while len(features) < 68:
            features.append(0.0)

        return np.array(features, dtype=np.float32)
