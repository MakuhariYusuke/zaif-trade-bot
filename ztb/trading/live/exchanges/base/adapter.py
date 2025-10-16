"""
Base exchange adapter with common functionality for all exchanges.

Provides shared implementation for dry-run simulation, rate limiting,
and common broker interface methods.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, TypedDict, Union

from ztb.utils.errors import InsufficientFundsError, MinimumSizeError, OrderNotFoundError, TradingBotError
from ztb.utils.rate_limiter import RateLimitConfig, RateLimiter

from .broker_interfaces import Balance, IBroker, Order, Position

logger = logging.getLogger(__name__)


# Type definitions for better type safety
class BaseOrderResponse(TypedDict, total=False):
    """Base response structure for order operations."""
    order_id: str
    symbol: str
    side: Union[Literal["buy"], Literal["sell"]]
    quantity: float
    price: Optional[float]
    order_type: Union[Literal["market"], Literal["limit"]]
    status: str
    client_order_id: Optional[str]
    timestamp: Optional[int]


class BaseBalanceResponse(TypedDict, total=False):
    """Base response structure for balance operations."""
    currency: str
    free: float
    locked: float
    total: float


class BasePositionResponse(TypedDict, total=False):
    """Base response structure for position operations."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    pnl: float


class BaseExchangeAdapter(IBroker, ABC):
    """
    Base class for exchange adapters with common functionality.

    Provides dry-run simulation, rate limiting, and shared broker interface methods.
    Subclasses must implement exchange-specific API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        dry_run: bool = True,
        rate_limiter: Optional[RateLimiter] = None,
        fixed_price: Optional[float] = None,
        random_seed: Optional[int] = None,
        requests_per_second: float = 5.0,
    ) -> None:
        """Initialize base exchange adapter.

        Args:
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            dry_run: If True, simulate all operations without real API calls
            rate_limiter: Rate limiter for API calls
            fixed_price: If set, always return this price in get_current_price (for testing)
            random_seed: If set, seed the random number generator for reproducibility
            requests_per_second: Rate limit for API calls
        """
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.fixed_price = fixed_price
        if random_seed is not None:
            random.seed(random_seed)

        if rate_limiter is None:
            config = RateLimitConfig(requests_per_second=requests_per_second)
            self.rate_limiter = RateLimiter(config)
        else:
            self.rate_limiter = rate_limiter

        # Dry-run state
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._balances: Dict[str, Balance] = {
            "JPY": Balance(currency="JPY", free=100000.0, locked=0.0, total=100000.0),
            "BTC": Balance(currency="BTC", free=0.1, locked=0.0, total=0.1),
        }
        self._order_counter = 0
        self._current_prices: Dict[str, float] = {"btc_jpy": 5000000.0}  # Default price

    async def _simulate_delay(self) -> None:
        """Simulate API call delay."""
        if not self.dry_run:
            await asyncio.sleep(random.uniform(0.1, 0.5))
        else:
            await asyncio.sleep(0.01)  # Minimal delay for dry-run

    async def _check_rate_limit(self) -> None:
        """Check rate limit before API call."""
        if self.rate_limiter:
            await self.rate_limiter.wait()

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"{self.__class__.__name__.lower()}_{self._order_counter}_{int(time.time())}"

    # Common dry-run implementations
    async def _get_balance_dry_run(
        self, currency: Optional[str] = None
    ) -> List[Balance]:
        """Get balance in dry-run mode."""
        balances = list(self._balances.values())
        if currency:
            balances = [b for b in balances if b.currency == currency]
        return balances

    async def _place_order_dry_run(
        self,
        symbol: str,
        side: Union[str, Literal["buy"], Literal["sell"]],
        quantity: float,
        price: Optional[float] = None,
        order_type: Union[str, Literal["market"], Literal["limit"]] = "market",
        client_order_id: Optional[str] = None,
        sizing_reason: Optional[str] = None,
        target_vol: Optional[float] = None,
    ) -> Order:
        """Place order in dry-run mode."""
        # Validate minimum order size
        if quantity <= 0.00001:  # Minimum BTC order size
            raise MinimumSizeError(
                f"Order quantity {quantity} is below minimum size requirement (0.00001 BTC)"
            )

        order_id = self._generate_order_id()
        current_price = self._current_prices.get(symbol, 5000000.0)

        # Simulate order execution
        if order_type == "market":
            exec_price = current_price * (1 + random.uniform(-0.001, 0.001))
        else:
            exec_price = price if price is not None else current_price

        # Simulate partial fills for realism
        fill_probability = random.random()
        if fill_probability > 0.1:  # 90% fill rate
            status = "filled"
            # Update balances/positions
            if side == "buy":
                cost = exec_price * quantity
                if self._balances["JPY"].free < cost:
                    raise InsufficientFundsError(
                        f"Insufficient JPY balance for buy order. Required: {cost}, Available: {self._balances['JPY'].free}"
                    )
                self._balances["JPY"].free -= cost
                self._balances["JPY"].total -= cost
                # Add to position
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    total_qty = pos.quantity + quantity
                    total_cost = (pos.quantity * pos.avg_price) + (
                        quantity * exec_price
                    )
                    new_avg = total_cost / total_qty
                    pos.quantity = total_qty
                    pos.avg_price = new_avg
                    pos.current_price = exec_price
                    pos.pnl = (exec_price - pos.avg_price) * total_qty
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=exec_price,
                        current_price=exec_price,
                        pnl=0.0,
                    )
            elif side == "sell":
                if (
                    symbol not in self._positions
                    or self._positions[symbol].quantity < quantity
                ):
                    available_qty = self._positions.get(symbol, Position(symbol, 0, 0, 0, 0)).quantity
                    raise InsufficientFundsError(
                        f"Insufficient {symbol} position for sell order. Required: {quantity}, Available: {available_qty}"
                    )
                pos = self._positions[symbol]
                proceeds = exec_price * quantity
                self._balances["JPY"].free += proceeds
                pos.quantity -= quantity
                pos.current_price = exec_price
                pos.pnl = (exec_price - pos.avg_price) * pos.quantity
                if pos.quantity <= 0:
                    del self._positions[symbol]
        else:
            status = "pending"  # Simulate unfilled order

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=exec_price,
            order_type=order_type,
            status=status,
            client_order_id=client_order_id,
            sizing_reason=sizing_reason,
            target_vol=target_vol,
        )

        self._orders[order_id] = order
        return order

    async def _cancel_order_dry_run(self, order_id: str) -> bool:
        """Cancel order in dry-run mode."""
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order with ID {order_id} not found")

        order = self._orders[order_id]
        if order.status == "pending":
            order.status = "cancelled"
            return True
        return False

    async def _get_order_status_dry_run(self, order_id: str) -> Optional[Order]:
        """Get order status in dry-run mode."""
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order with ID {order_id} not found")
        return self._orders.get(order_id)

    async def _get_open_orders_dry_run(
        self, symbol: Optional[str] = None
    ) -> List[Order]:
        """Get open orders in dry-run mode."""
        orders = [o for o in self._orders.values() if o.status == "pending"]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def _get_positions_dry_run(self) -> List[Position]:
        """Get positions in dry-run mode."""
        return list(self._positions.values())

    async def _get_current_price_dry_run(self, symbol: str) -> Optional[float]:
        """Get current price in dry-run mode."""
        if self.fixed_price is not None:
            return self.fixed_price
        base_price = self._current_prices.get(symbol, 5000000.0)
        self._current_prices[symbol] = base_price * (1 + random.uniform(-0.005, 0.005))
        return self._current_prices[symbol]

    # Abstract methods that subclasses must implement
    @abstractmethod
    async def _get_balance_real(self, currency: Optional[str] = None) -> List[Balance]:
        """Get balance from real API."""

    @abstractmethod
    async def _place_order_real(
        self,
        symbol: str,
        side: Union[str, Literal["buy"], Literal["sell"]],
        quantity: float,
        price: Optional[float] = None,
        order_type: Union[str, Literal["market"], Literal["limit"]] = "market",
        client_order_id: Optional[str] = None,
        sizing_reason: Optional[str] = None,
        target_vol: Optional[float] = None,
    ) -> Order:
        """Place order via real API."""

    @abstractmethod
    async def _cancel_order_real(self, order_id: str) -> bool:
        """Cancel order via real API."""

    @abstractmethod
    async def _get_order_status_real(self, order_id: str) -> Optional[Order]:
        """Get order status from real API."""

    @abstractmethod
    async def _get_open_orders_real(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders from real API."""

    @abstractmethod
    async def _get_positions_real(self) -> List[Position]:
        """Get positions from real API."""

    @abstractmethod
    async def _get_current_price_real(self, symbol: str) -> Optional[float]:
        """Get current price from real API."""

    # Public interface implementations
    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get account balance, optionally for specific currency."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._get_balance_dry_run(currency)
        else:
            return await self._get_balance_real(currency)

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
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._place_order_dry_run(
                symbol,
                side,
                quantity,
                price,
                order_type,
                client_order_id,
                sizing_reason,
                target_vol,
            )
        else:
            return await self._place_order_real(
                symbol,
                side,
                quantity,
                price,
                order_type,
                client_order_id,
                sizing_reason,
                target_vol,
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._cancel_order_dry_run(order_id)
        else:
            return await self._cancel_order_real(order_id)

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._get_order_status_dry_run(order_id)
        else:
            return await self._get_order_status_real(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._get_open_orders_dry_run(symbol)
        else:
            return await self._get_open_orders_real(symbol)

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._get_positions_dry_run()
        else:
            return await self._get_positions_real()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if self.dry_run:
            return await self._get_current_price_dry_run(symbol)
        else:
            return await self._get_current_price_real(symbol)
