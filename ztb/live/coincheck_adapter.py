"""
Coincheck exchange adapter with dry-run support.

Implements IBroker interface with dry-run simulation for testing.
Real trading implementation is stubbed for future development.
"""

import asyncio
import random
import time
from typing import Dict, List, Optional

from ..utils.rate_limiter import RateLimitConfig, RateLimiter
from .broker_interfaces import Balance, IBroker, Order, Position


class CoincheckAdapter(IBroker):
    """Coincheck exchange adapter with dry-run simulation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        dry_run: bool = True,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Initialize Coincheck adapter.

        Args:
            api_key: API key (ignored in dry-run)
            api_secret: API secret (ignored in dry-run)
            dry_run: If True, simulate all operations without real API calls
            rate_limiter: Rate limiter for API calls
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        if rate_limiter is None:
            config = RateLimitConfig(
                requests_per_second=5.0
            )  # 300 calls per minute = 5 per second
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
        self._current_prices: Dict[str, float] = {
            "btc_jpy": 5000000.0  # Sample price
        }

    async def _simulate_delay(self):
        """Simulate API call delay."""
        if not self.dry_run:
            await asyncio.sleep(random.uniform(0.1, 0.5))
        else:
            await asyncio.sleep(0.01)  # Minimal delay for dry-run

    async def _check_rate_limit(self):
        """Check rate limit before API call."""
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed()

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"cc_dry_{self._order_counter}_{int(time.time())}"

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

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        # Dry-run simulation
        order_id = self._generate_order_id()
        current_price = self._current_prices.get(symbol, 5000000.0)

        # Simulate order execution
        if order_type == "market":
            exec_price = current_price * (
                1 + random.uniform(-0.001, 0.001)
            )  # Small slippage
        else:
            exec_price = price

        # Simulate partial fills for realism
        fill_probability = random.random()
        if fill_probability > 0.1:  # 90% fill rate
            status = "filled"
            # Update balances/positions
            if side == "buy":
                cost = exec_price * quantity
                if self._balances["JPY"].free >= cost:
                    self._balances["JPY"].free -= cost
                    self._balances["JPY"].locked += cost
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
                        pos.pnl = (exec_price - new_avg) * total_qty
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
                    symbol in self._positions
                    and self._positions[symbol].quantity >= quantity
                ):
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

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status == "pending":
                order.status = "cancelled"
                return True
        return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        orders = [o for o in self._orders.values() if o.status == "pending"]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        return list(self._positions.values())

    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get account balance, optionally for specific currency."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        balances = list(self._balances.values())
        if currency:
            balances = [b for b in balances if b.currency == currency]
        return balances

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            raise NotImplementedError("Real Coincheck trading not implemented")

        # Simulate price movement
        base_price = self._current_prices.get(symbol, 5000000.0)
        self._current_prices[symbol] = base_price * (1 + random.uniform(-0.005, 0.005))
        return self._current_prices[symbol]
