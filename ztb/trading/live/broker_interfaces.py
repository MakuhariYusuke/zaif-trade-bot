"""
Broker interface definitions for live and paper trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None  # Market order if None
    order_type: str = "market"  # 'market' or 'limit'
    status: str = "pending"  # 'pending', 'filled', 'cancelled', 'rejected'
    client_order_id: Optional[str] = None
    sizing_reason: Optional[str] = None
    target_vol: Optional[float] = None


@dataclass
class Position:
    """Position representation."""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    pnl: float


@dataclass
class Balance:
    """Account balance representation."""

    currency: str
    free: float
    locked: float
    total: float


class IBroker(ABC):
    """Abstract broker interface for trading operations."""

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        client_order_id: Optional[str] = None,
        sizing_reason: Optional[str] = None,
        target_vol: Optional[float] = None,
    ) -> Order:
        """Place a new order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get account balance, optionally for specific currency."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        pass


class ZaifAdapter(IBroker):
    """Zaif exchange adapter (stub implementation)."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize with API credentials."""
        self.api_key = api_key
        self.api_secret = api_secret

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Order:
        """Place order on Zaif (stub - raises NotImplementedError)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Zaif (stub)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from Zaif (stub)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders from Zaif (stub)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )

    async def get_positions(self) -> List[Position]:
        """Get positions from Zaif (stub)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )

    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get balance from Zaif (stub)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Zaif (stub)."""
        raise NotImplementedError(
            "ZaifAdapter is a stub for future real trading implementation"
        )
