"""
Coincheck exchange adapter with dry-run support.

Implements IBroker interface with dry-run simulation for testing.
Real trading implementation is stubbed for future development.
"""

import hashlib
import hmac
import json
import logging
import random
import time
import urllib.parse
from typing import Any, Dict, List, Literal, Optional, Union

import requests

from ztb.utils.errors import InsufficientFundsError, MinimumSizeError, NetworkError
from ztb.utils.rate_limiter import RateLimitConfig, RateLimiter

from ..base.broker_interfaces import (
    Balance,
    IBroker,
    MarketDataNotSupported,
    Order,
    OrderBookSnapshot,
    Position,
    TradeRecord,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


# ---- Helpers ----

def _parse_timestamp(value: Any) -> float:
    """Parse timestamp from epoch float/int or ISO 8601 string.

    Coincheck created_at can be either epoch or ISO 8601 (e.g. "2025-01-01T00:00:00.000Z").
    003# #4: float() on ISO string raises ValueError — added dual parser.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try float (epoch) first
        try:
            return float(value)
        except ValueError:
            pass
        # Try ISO 8601
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            logger.warning(f"Unparseable timestamp: {value!r}, using current time")
            return time.time()
    return time.time()


# Type definitions for API responses
CoincheckOrderResponse = Dict[str, Union[str, int, float]]
CoincheckBalanceResponse = Dict[str, Union[str, float]]
CoincheckErrorResponse = Dict[str, str]


class CoincheckAdapter(IBroker):
    """
    Coincheck exchange adapter (dry-run simulation only).

    Real trading is not implemented; all operations are simulated for testing.
    Real trading support may be added in the future.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        dry_run: bool = True,
        rate_limiter: Optional[RateLimiter] = None,
        fixed_price: Optional[float] = None,
        random_seed: Optional[int] = None,
        api_base_url: str = "https://coincheck.com",
        request_timeout: float = 10.0,
    ) -> None:
        """Initialize Coincheck adapter.

        Args:
            api_key: API key (ignored in dry-run)
            api_secret: API secret (ignored in dry-run)
            dry_run: If True, simulate all operations without real API calls
            rate_limiter: Rate limiter for API calls
            fixed_price: If set, always return this price in get_current_price (for testing)
            random_seed: If set, seed the random number generator for reproducibility
            api_base_url: Base URL for Coincheck API
            request_timeout: Timeout for API requests in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.dry_run = dry_run
        self.api_base_url = api_base_url
        self.request_timeout = request_timeout
        self.fixed_price = fixed_price
        if random_seed is not None:
            random.seed(random_seed)
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
        self._current_prices: Dict[str, float] = {"btc_jpy": 5000000.0}  # Sample price

    async def _simulate_delay(self) -> None:
        """Simulate API call delay for dry-run mode."""
        import asyncio

        if not self.dry_run:
            await asyncio.sleep(random.uniform(0.1, 0.5))
        else:
            await asyncio.sleep(0.01)

    def _create_signature(self, message: str) -> str:
        """Create HMAC-SHA256 signature for Coincheck API.

        Args:
            message: Message to sign

        Returns:
            Hexadecimal signature string
        """
        if not self.api_secret:
            raise ValueError("API secret is required for authentication")

        return hmac.new(
            self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _make_api_request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        url: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Union[
        CoincheckOrderResponse,
        CoincheckBalanceResponse,
        CoincheckErrorResponse,
        Dict[str, Any],
    ]:
        """Make authenticated API request to Coincheck.

        Args:
            method: HTTP method
            url: API endpoint URL
            data: Request data for POST requests

        Returns:
            API response as dictionary

        Raises:
            NetworkError: For network/API errors
        """
        nonce = str(int(time.time() * 1000000))

        if data:
            message = nonce + url + json.dumps(data, separators=(",", ":"))
        else:
            message = nonce + url

        signature = self._create_signature(message)

        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-NONCE": nonce,
            "ACCESS-SIGNATURE": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            if method.upper() == "GET":
                response = requests.get(
                    url, headers=headers, timeout=self.request_timeout
                )
            elif method.upper() == "POST":
                # Use URL-encoded data for POST requests
                request_data = None
                if data:
                    request_data = urllib.parse.urlencode(data)
                response = requests.post(
                    url,
                    headers=headers,
                    data=request_data,
                    timeout=self.request_timeout,
                )
            elif method.upper() == "DELETE":
                response = requests.delete(
                    url, headers=headers, timeout=self.request_timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            logger.info(f"API Response status: {response.status_code}")
            logger.info(
                f"API Response content: {response.text[:500]}"
            )  # Log first 500 chars
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Coincheck API request failed: {e}")
            raise NetworkError(f"Coincheck API error: {e}")

    async def _check_rate_limit(self) -> None:
        """Check rate limit before API call."""
        if self.rate_limiter:
            await self.rate_limiter.wait()

    def _generate_order_id(self) -> str:
        import uuid

        return str(uuid.uuid4())

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
            # Real API call
            url = f"{self.api_base_url}/api/exchange/orders"

            # Prepare order data for Coincheck API
            order_data = {
                "pair": symbol.replace("_", "_"),  # btc_jpy -> btc_jpy
                "order_type": side,  # "buy" or "sell"
                "amount": str(quantity),
            }

            if order_type == "limit" and price is not None:
                order_data["rate"] = str(int(price))
            # For market orders, Coincheck uses market_buy_amount for buy orders

            try:
                result = self._make_api_request("POST", url, order_data)
                logger.info(f"Placed order: {result}")

                # Convert API response to Order object
                order_id = str(result.get("id", self._generate_order_id()))
                status = "pending"  # Assume pending initially

                order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    status=status,
                    client_order_id=client_order_id,
                    sizing_reason=sizing_reason,
                    target_vol=target_vol,
                )

                return order

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to place order: {error_msg}")

                # Check for specific API errors and raise appropriate exceptions
                if (
                    "insufficient" in error_msg.lower()
                    or "balance" in error_msg.lower()
                ):
                    logger.warning(
                        f"Order failed due to insufficient balance: {error_msg}"
                    )
                    raise InsufficientFundsError(
                        f"Insufficient balance for {side} order of {quantity} {symbol}"
                    )
                elif "minimum" in error_msg.lower() or "size" in error_msg.lower():
                    logger.warning(
                        f"Order failed due to minimum size requirements: {error_msg}"
                    )
                    raise MinimumSizeError(
                        f"Order size {quantity} below minimum requirements"
                    )
                else:
                    # Re-raise network/API errors
                    raise
        order_id = self._generate_order_id()
        current_price = self._current_prices.get(symbol, 5000000.0)

        # Simulate order execution
        if order_type == "market":
            exec_price = current_price * (
                1 + random.uniform(-0.001, 0.001)
            )  # Small slippage
        else:
            exec_price = price if price is not None else current_price

        # Simulate partial fills for realism
        fill_probability = random.random()
        if fill_probability > 0.1:  # 90% fill rate
            status = "filled"
            # Update balances/positions
            if side == "buy":
                cost = exec_price * quantity
                if self._balances["JPY"].free >= cost:
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
            # Real API call
            url = f"{self.api_base_url}/api/exchange/orders/{order_id}"
            try:
                result = self._make_api_request("DELETE", url)
                logger.info(f"Cancelled order {order_id}: {result}")
                success_value = result.get("success", False)
                return (
                    bool(success_value)
                    if isinstance(success_value, (bool, int))
                    else False
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to cancel order {order_id}: {error_msg}")

                # Check for specific API errors
                if (
                    "not found" in error_msg.lower()
                    or "already cancelled" in error_msg.lower()
                ):
                    logger.warning(
                        f"Order cancellation failed (order may have already executed or been cancelled): {error_msg}"
                    )
                    # Don't raise exception for order not found, just return False
                    return False
                else:
                    # Re-raise other errors
                    raise

        # Dry-run simulation
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status == "pending":
                order.status = "cancelled"
                return True
        return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order.

        Real mode: ``GET /api/exchange/orders/opens`` で確認後、
        見つからなければ ``GET /api/exchange/orders/transactions`` で約定済みを確認。
        009# P2-0: real mode 実装.
        """
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            import asyncio

            # 1) Check open orders first
            url_opens = f"{self.api_base_url}/api/exchange/orders/opens"
            try:
                result = await asyncio.to_thread(
                    self._make_api_request, "GET", url_opens, None,
                )
                orders_list = result.get("orders", [])
                for o in orders_list:
                    if str(o.get("id")) == str(order_id):
                        return Order(
                            order_id=str(o["id"]),
                            symbol=str(o.get("pair", "")),
                            side=str(o.get("order_type", "buy")),
                            quantity=float(o.get("pending_amount", o.get("amount", 0))),
                            price=float(o.get("rate", 0)) if o.get("rate") else None,
                            order_type="limit" if o.get("rate") else "market",
                            status="pending",
                        )
            except Exception as e:
                logger.warning(f"Failed to check open orders for {order_id}: {e}")

            # 2) Check transactions (filled orders)
            url_txns = f"{self.api_base_url}/api/exchange/orders/transactions"
            try:
                result = await asyncio.to_thread(
                    self._make_api_request, "GET", url_txns, None,
                )
                txns = result.get("transactions", [])
                for t in txns:
                    if str(t.get("order_id")) == str(order_id):
                        return Order(
                            order_id=str(t["order_id"]),
                            symbol=str(t.get("pair", "")),
                            side=str(t.get("side", "buy")),
                            quantity=float(t.get("funds", {}).get("btc", 0)),
                            price=float(t.get("rate", 0)) if t.get("rate") else None,
                            order_type="limit",
                            status="filled",
                        )
            except Exception as e:
                logger.warning(f"Failed to check transactions for {order_id}: {e}")

            return None

        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol.

        Real mode: ``GET /api/exchange/orders/opens``.
        009# P2-0: real mode 実装.
        """
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            import asyncio

            url = f"{self.api_base_url}/api/exchange/orders/opens"
            try:
                result = await asyncio.to_thread(
                    self._make_api_request, "GET", url, None,
                )
                orders_list = result.get("orders", [])
                orders: List[Order] = []
                for o in orders_list:
                    pair = str(o.get("pair", ""))
                    if symbol and pair != normalize_symbol(symbol):
                        continue
                    orders.append(
                        Order(
                            order_id=str(o["id"]),
                            symbol=pair,
                            side=str(o.get("order_type", "buy")),
                            quantity=float(o.get("pending_amount", o.get("amount", 0))),
                            price=float(o.get("rate", 0)) if o.get("rate") else None,
                            order_type="limit" if o.get("rate") else "market",
                            status="pending",
                        )
                    )
                return orders
            except Exception as e:
                logger.error(f"Failed to get open orders: {e}")
                raise

        orders = [o for o in self._orders.values() if o.status == "pending"]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_positions(self) -> List[Position]:
        """Get current positions.

        Coincheck doesn't have a direct positions API for spot.
        Return balance-based positions (BTC holdings as position).
        009# P2-0: real mode — balance から推定.
        """
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            # Use balance to infer spot position
            balances = await self.get_balance("BTC")
            positions: List[Position] = []
            for b in balances:
                if b.total > 0:
                    positions.append(
                        Position(
                            symbol="btc_jpy",
                            quantity=b.total,
                            avg_price=0.0,  # Not tracked by exchange
                            current_price=0.0,
                            pnl=0.0,
                        )
                    )
            return positions

        return list(self._positions.values())

    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get account balance, optionally for specific currency."""
        await self._check_rate_limit()
        await self._simulate_delay()

        if not self.dry_run:
            # Real API call
            url = f"{self.api_base_url}/api/accounts/balance"
            try:
                result = self._make_api_request("GET", url)
                logger.info(f"Retrieved balance: {result}")

                # Check if result is a dict
                if not isinstance(result, dict):
                    raise ValueError(
                        f"Unexpected API response type: {type(result)}, content: {result}"
                    )

                # Check for API errors
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown API error")
                    raise Exception(f"Coincheck API error: {error_msg}")

                # Convert API response to Balance objects
                balances = []
                for currency_code, balance_str in result.items():
                    if currency_code not in [
                        "success",
                        "error",
                    ]:  # Skip metadata fields
                        try:
                            balance_value = float(balance_str)
                            balances.append(
                                Balance(
                                    currency=currency_code.upper(),
                                    free=balance_value,
                                    locked=0.0,  # Coincheck doesn't separate free/locked in balance
                                    total=balance_value,
                                )
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Failed to parse balance for {currency_code}: {balance_str}, error: {e}"
                            )

                # Filter by currency if specified
                if currency:
                    balances = [b for b in balances if b.currency == currency.upper()]

                return balances

            except Exception as e:
                logger.error(f"Failed to get balance: {e}")
                raise

        # Dry-run simulation
        balances = list(self._balances.values())
        if currency:
            balances = [b for b in balances if b.currency == currency]
        return balances

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol.

        Real mode: ``GET /api/ticker`` (public, no auth).
        009# P2-0: dry-run / real 共通で public ticker を参照.
        """
        await self._check_rate_limit()
        await self._simulate_delay()

        # Public API — real / dry-run 共通でティッカーを取得
        try:
            import asyncio

            response = await asyncio.to_thread(
                requests.get,
                f"{self.api_base_url}/api/ticker",
                timeout=self.request_timeout,
                headers={"User-Agent": "ZaifTradeBot/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "last" in data:
                real_price = float(data["last"])
                self._current_prices[symbol] = real_price
                logger.debug(f"Retrieved real price from Coincheck API: {real_price}")
                return real_price
        except requests.exceptions.Timeout:
            logger.warning("Coincheck API request timed out")
            if not self.dry_run:
                raise NetworkError(
                    "Coincheck API request timed out",
                    details={"symbol": symbol, "timeout": self.request_timeout},
                )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get real price from Coincheck API: {e}")
            if not self.dry_run:
                raise NetworkError(
                    f"Coincheck API request failed: {str(e)}",
                    details={"symbol": symbol, "error": str(e)},
                )
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid response from Coincheck API: {e}")
            if not self.dry_run:
                raise NetworkError(
                    f"Invalid Coincheck API response: {str(e)}",
                    details={"symbol": symbol, "error": str(e)},
                )

        # Dry-run fallback
        if self.fixed_price is not None:
            return self.fixed_price
        base_price = self._current_prices.get(symbol, 5000000.0)
        self._current_prices[symbol] = base_price * (1 + random.uniform(-0.005, 0.005))
        return self._current_prices[symbol]

    # -------------------------------------------------------------------
    # v460: Market data methods (板情報・約定フロー)
    # -------------------------------------------------------------------

    async def get_orderbook(
        self, symbol: str, depth: int = 10
    ) -> OrderBookSnapshot:
        """Fetch orderbook from Coincheck ``GET /api/order_books``.

        Public API — no authentication required.
        003# #9: sync requests wrapped with asyncio.to_thread.
        """
        import asyncio

        await self._check_rate_limit()
        url = f"{self.api_base_url}/api/order_books"
        try:
            response = await asyncio.to_thread(
                requests.get, url, timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            raw_bids: list[list[str]] = data.get("bids", [])[:depth]
            raw_asks: list[list[str]] = data.get("asks", [])[:depth]

            bids = [(float(p), float(s)) for p, s in raw_bids]
            asks = [(float(p), float(s)) for p, s in raw_asks]

            return OrderBookSnapshot(
                timestamp=time.time(),
                bids=bids,
                asks=asks,
                exchange="coincheck",
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Coincheck orderbook request failed: {e}")
            raise NetworkError(f"Coincheck orderbook error: {e}")

    async def get_recent_trades(
        self, symbol: str, limit: int = 100
    ) -> list[TradeRecord]:
        """Fetch recent trades from Coincheck ``GET /api/trades``.

        Public API — no authentication required.

        Args:
            symbol: Trading pair in internal format (e.g. ``btc_jpy``).
            limit: Max number of trades to fetch.
        """
        await self._check_rate_limit()
        pair = normalize_symbol(symbol)
        url = f"{self.api_base_url}/api/trades?pair={pair}&limit={limit}"
        try:
            import asyncio

            response = await asyncio.to_thread(
                requests.get, url, timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            trades: list[TradeRecord] = []
            items = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = []
            for item in items[:limit]:
                trades.append(
                    TradeRecord(
                        timestamp=_parse_timestamp(item.get("created_at", time.time())),
                        price=float(item.get("rate", 0)),
                        amount=float(item.get("amount", 0)),
                        side=str(item.get("order_type", "buy")),
                    )
                )
            return trades
        except requests.exceptions.RequestException as e:
            logger.error(f"Coincheck trades request failed: {e}")
            raise NetworkError(f"Coincheck trades error: {e}")
