"""
Common interfaces for trading components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Generate trading signal based on current data and position."""
        pass

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Return list of required data columns."""
        pass


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    async def get_historical_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass


class OrderManager(ABC):
    """Abstract base class for order management."""

    @abstractmethod
    async def place_order(self, order: Dict[str, Any]) -> str:
        """Place an order and return order ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order."""
        pass
