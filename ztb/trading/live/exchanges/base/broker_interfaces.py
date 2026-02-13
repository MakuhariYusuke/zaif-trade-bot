"""
Broker interface definitions for live and paper trading.

v460: OrderBookSnapshot / TradeRecord / MarketDataNotSupported added.
      get_orderbook / get_recent_trades are NON-abstract (default raises
      MarketDataNotSupported) so existing adapters remain unbroken.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# v460: Market data types
# ---------------------------------------------------------------------------


@dataclass
class OrderBookSnapshot:
    """Orderbook snapshot at a point in time.

    bids: [(price, size), ...] in descending price order.
    asks: [(price, size), ...] in ascending price order.
    """

    timestamp: float
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    exchange: str = ""


@dataclass
class TradeRecord:
    """Single trade (execution) record."""

    timestamp: float
    price: float
    amount: float
    side: str  # 'buy' or 'sell'


class MarketDataNotSupported(Exception):
    """Raised when an adapter does not support market data collection."""


# ---------------------------------------------------------------------------
# Normalisation helper — internal symbol is always lowercase.
# Each adapter converts to exchange-native format in API calls.
# ---------------------------------------------------------------------------


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to lowercase internal representation."""
    return symbol.lower()


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

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""

    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get account balance, optionally for specific currency."""

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""

    # -------------------------------------------------------------------
    # v460 market-data methods (non-abstract — default raises)
    # -------------------------------------------------------------------

    async def get_orderbook(
        self, symbol: str, depth: int = 10
    ) -> OrderBookSnapshot:
        """Get orderbook snapshot (top *depth* levels).

        Default implementation raises ``MarketDataNotSupported`` so that
        existing adapters (SimBroker, ZaifAdapter …) keep working without
        any changes.  Override in adapters that support this call.
        """
        raise MarketDataNotSupported(
            f"{self.__class__.__name__} does not support orderbook"
        )

    async def get_recent_trades(
        self, symbol: str, limit: int = 100
    ) -> list[TradeRecord]:
        """Get recent trades.

        Default implementation raises ``MarketDataNotSupported``.
        Override in adapters that support this call.
        """
        raise MarketDataNotSupported(
            f"{self.__class__.__name__} does not support trades"
        )


class ZaifAdapter(IBroker):
    """Zaif exchange adapter (stub implementation)."""

    def __init__(
        self, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ) -> None:
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
        client_order_id: Optional[str] = None,
        sizing_reason: Optional[str] = None,
        target_vol: Optional[float] = None,
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
