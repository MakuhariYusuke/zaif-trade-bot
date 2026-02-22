"""
Coincheck exchange adapter with dry-run and real trading support.

Inherits BaseExchangeAdapter for shared dry-run simulation, rate limiting,
and order state management. Implements Coincheck-specific API calls.

013# C-3/C-4/C-7/C-9/D-3/D-5: Signature fix, async unification,
order_type mapping, post_only support, rate limit correction.

145# §13: Migrated from IBroker direct to BaseExchangeAdapter inheritance.
"""

import hashlib
import hmac
import logging
import time
import urllib.parse
import uuid
from typing import Dict, List, Literal, Optional, Union

import requests

from ztb.utils.errors import InsufficientFundsError, MinimumSizeError, NetworkError
from ztb.utils.rate_limiter import RateLimiter

from ..base.adapter import BaseExchangeAdapter
from ..base.broker_interfaces import (
    Balance,
    Order,
    OrderBookSnapshot,
    Position,
    TradeRecord,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


# ---- Helpers ----

def _parse_timestamp(value: Union[int, float, str, None]) -> float:
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


class CoincheckAdapter(BaseExchangeAdapter):
    """
    Coincheck exchange adapter with dry-run and real trading support.

    Inherits BaseExchangeAdapter for shared dry-run simulation,
    rate limiting, and order state management (145# §13 migration).

    Real API paths: place_order, cancel_order, get_order_status,
    get_open_orders, get_balance, get_current_price, get_orderbook,
    get_recent_trades.

    013# C-9: Updated to reflect actual implementation state.
    """

    BASE_URL = "https://coincheck.com"

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
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            dry_run=dry_run,
            rate_limiter=rate_limiter,
            fixed_price=fixed_price,
            random_seed=random_seed,
            requests_per_second=4.0,  # 013# D-5: Coincheck rate limit
        )
        self.api_base_url = api_base_url
        self.request_timeout = request_timeout
        # Override default prices for Coincheck
        self._current_prices: Dict[str, float] = {"btc_jpy": 5000000.0}
        # 146# §13: lazy-init Session with retry (see _get_session)
        self._session: Optional[requests.Session] = None

    def _generate_order_id(self) -> str:
        """Generate unique order ID using UUID (Coincheck convention)."""
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # HTTP session (lazy-init with retry)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close HTTP session and release resources."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None
        super().close()

    def _get_session(self) -> requests.Session:
        """Return lazily-created persistent HTTP session."""
        if self._session is None:
            self._session = self._create_session()
        return self._session

    def _create_session(self) -> requests.Session:
        """Create ``requests.Session`` with retry/back-off.

        Imports are **lazy** to avoid breaking test environments where
        ``requests`` is stubbed without ``adapters`` / ``urllib3`` sub-modules.

        146# §13: 3-retry, 0.5 s back-off, 5xx forcelist.
        """
        session = requests.Session()
        try:
            from requests.adapters import HTTPAdapter  # noqa: WPS433
            from urllib3.util.retry import Retry  # noqa: WPS433

            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST", "DELETE"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except (ImportError, AttributeError):
            # Test env: stub requests has no adapters sub-module
            logger.debug("requests.adapters unavailable; using plain Session")
        return session

    # ------------------------------------------------------------------
    # Coincheck API helpers (authentication, signing, HTTP)
    # ------------------------------------------------------------------

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
        data: Optional[Dict[str, str]] = None,
    ) -> Union[
        CoincheckOrderResponse,
        CoincheckBalanceResponse,
        CoincheckErrorResponse,
        Dict[str, object],
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

        # 013# C-3 FIX: 署名対象と実送信ボディを一致させる。
        request_body: Optional[str] = None
        if data and method.upper() == "POST":
            request_body = urllib.parse.urlencode(data)

        if request_body:
            message = nonce + url + request_body
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
            session = self._get_session()
            if method.upper() == "GET":
                response = session.get(
                    url, headers=headers, timeout=self.request_timeout
                )
            elif method.upper() == "POST":
                response = session.post(
                    url,
                    headers=headers,
                    data=request_body,
                    timeout=self.request_timeout,
                )
            elif method.upper() == "DELETE":
                response = session.delete(
                    url, headers=headers, timeout=self.request_timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            # 047# A6/Issue13: API Response ログを DEBUG に降格
            logger.debug(f"API Response status: {response.status_code}")
            logger.debug(f"API Response content: {response.text[:500]}")
            return response.json()

        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                body = e.response.text[:500]
            logger.error(f"Coincheck API request failed: {e} | body={body}")
            raise NetworkError(f"Coincheck API error: {e} | body={body}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Coincheck API request failed: {e}")
            raise NetworkError(f"Coincheck API error: {e}")

    # ------------------------------------------------------------------
    # BaseExchangeAdapter abstract method implementations (_xxx_real)
    # ------------------------------------------------------------------

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
        """Place order via Coincheck real API."""
        import asyncio

        url = f"{self.api_base_url}/api/exchange/orders"

        # 013# C-7 FIX: Coincheck order_type mapping
        order_data: Dict[str, str] = {
            "pair": normalize_symbol(symbol),
        }

        if order_type == "limit" and price is not None:
            order_data["order_type"] = side  # "buy" or "sell"
            # 044# E-3: round() for sell-side bias fix
            order_data["rate"] = str(round(price))
            order_data["amount"] = str(quantity)
            # 013# D-3: maker-only — post_only prevents taker fill
            order_data["time_in_force"] = "post_only"
        elif order_type == "market":
            if side == "buy":
                order_data["order_type"] = "market_buy"
                current_price = self._current_prices.get(
                    normalize_symbol(symbol), 0.0
                )
                if current_price > 0:
                    jpy_amount = quantity * current_price
                else:
                    jpy_amount = quantity * 5000000.0
                    logger.warning(
                        f"No cached price for {symbol}, using fallback for market_buy_amount"
                    )
                order_data["market_buy_amount"] = str(int(jpy_amount))
            else:
                order_data["order_type"] = "market_sell"
                order_data["amount"] = str(quantity)
        else:
            order_data["order_type"] = side
            order_data["amount"] = str(quantity)
            if price is not None:
                order_data["rate"] = str(round(price))

        try:
            # 013# C-4 FIX: asyncio.to_thread
            result = await asyncio.to_thread(
                self._make_api_request, "POST", url, order_data,
            )
            logger.info(f"Placed order: {result}")

            if isinstance(result, dict) and not result.get("success", True):
                error_msg = result.get("error", "Unknown API error")
                raise Exception(f"Coincheck API error: {error_msg}")

            order_id = str(result.get("id", self._generate_order_id()))

            return Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
                status="pending",
                client_order_id=client_order_id,
                sizing_reason=sizing_reason,
                target_vol=target_vol,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to place order: {error_msg}")

            if (
                "insufficient" in error_msg.lower()
                or "balance" in error_msg.lower()
            ):
                raise InsufficientFundsError(
                    f"Insufficient balance for {side} order of {quantity} {symbol}"
                )
            elif "minimum" in error_msg.lower() or "size" in error_msg.lower():
                raise MinimumSizeError(
                    f"Order size {quantity} below minimum requirements"
                )
            else:
                raise

    async def _cancel_order_real(self, order_id: str) -> bool:
        """Cancel order via Coincheck real API."""
        import asyncio

        url = f"{self.api_base_url}/api/exchange/orders/{order_id}"
        try:
            result = await asyncio.to_thread(
                self._make_api_request, "DELETE", url, None,
            )
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

            if (
                "not found" in error_msg.lower()
                or "already cancelled" in error_msg.lower()
            ):
                logger.warning(
                    f"Order cancellation failed (order may have already executed "
                    f"or been cancelled): {error_msg}"
                )
                return False
            else:
                raise

    async def _get_order_status_real(self, order_id: str) -> Optional[Order]:
        """Get order status from Coincheck real API.

        Two-step: check opens first, then transactions for filled orders.
        009# P2-0: real mode implementation.
        """
        import asyncio

        # 1) Check open orders
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
        # 044# E-1: rate limit before second API call
        await self._check_rate_limit()
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

    async def _get_open_orders_real(
        self, symbol: Optional[str] = None
    ) -> List[Order]:
        """Get open orders from Coincheck real API.

        009# P2-0: real mode implementation.
        """
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

    async def _get_positions_real(self) -> List[Position]:
        """Get positions from Coincheck (inferred from balance).

        Coincheck has no direct positions API for spot.
        009# P2-0: real mode — balance-based inference.
        """
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

    async def _get_balance_real(
        self, currency: Optional[str] = None
    ) -> List[Balance]:
        """Get balance from Coincheck real API."""
        import asyncio

        url = f"{self.api_base_url}/api/accounts/balance"
        try:
            result = await asyncio.to_thread(
                self._make_api_request, "GET", url, None,
            )
            logger.debug(f"Retrieved balance (raw): {result}")
            # 047# Refactor: log non-zero balances only
            nonzero = {
                k: v for k, v in result.items()
                if k not in ("success", "error")
                and isinstance(v, str)
                and v != "0.0"
            }
            if nonzero:
                logger.info(f"Non-zero balances: {nonzero}")

            if not isinstance(result, dict):
                raise ValueError(
                    f"Unexpected API response type: {type(result)}, content: {result}"
                )

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown API error")
                raise Exception(f"Coincheck API error: {error_msg}")

            balances: list[Balance] = []
            reserved_suffix = "_reserved"
            # 046# Coincheck reserved/lending/etc. keys
            _ignore_suffixes = (
                "_reserved", "_lending", "_lend_in_use",
                "_lent", "_debt", "_tsumitate",
            )
            currency_keys = [
                k for k in result.keys()
                if k not in ("success", "error")
                and not any(k.endswith(s) for s in _ignore_suffixes)
            ]
            for currency_code in currency_keys:
                try:
                    free_val = float(result.get(currency_code, 0))
                    locked_val = float(
                        result.get(f"{currency_code}{reserved_suffix}", 0)
                    )
                    total_val = free_val + locked_val
                    balances.append(
                        Balance(
                            currency=currency_code.upper(),
                            free=free_val,
                            locked=locked_val,
                            total=total_val,
                        )
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse balance for {currency_code}: "
                        f"{result.get(currency_code)}, error: {e}"
                    )

            if currency:
                balances = [b for b in balances if b.currency == currency.upper()]

            return balances

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    async def _get_current_price_real(self, symbol: str) -> Optional[float]:
        """Get current price from Coincheck public ticker API."""
        import asyncio

        try:
            response = await asyncio.to_thread(
                self._get_session().get,
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
            raise NetworkError(
                "Coincheck API request timed out",
                details={"symbol": symbol, "timeout": self.request_timeout},
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get real price from Coincheck API: {e}")
            raise NetworkError(
                f"Coincheck API request failed: {str(e)}",
                details={"symbol": symbol, "error": str(e)},
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid response from Coincheck API: {e}")
            raise NetworkError(
                f"Invalid Coincheck API response: {str(e)}",
                details={"symbol": symbol, "error": str(e)},
            )
        return None

    # ------------------------------------------------------------------
    # Override: get_current_price — mixed behavior (public API in dry-run)
    # 009# P2-0: dry-run / real 共通で public ticker を参照
    # ------------------------------------------------------------------

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol.

        Overrides BaseExchangeAdapter to use public ticker API even in dry-run.
        Falls back to simulated price if API is unavailable in dry-run.
        """
        await self._check_rate_limit()
        await self._simulate_delay()

        # Public API — try real ticker first regardless of dry_run
        try:
            return await self._get_current_price_real(symbol)
        except (NetworkError, Exception):
            if not self.dry_run:
                raise

        # Dry-run fallback
        return await self._get_current_price_dry_run(symbol)

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
                self._get_session().get, url, timeout=self.request_timeout,
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
                self._get_session().get, url, timeout=self.request_timeout,
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
