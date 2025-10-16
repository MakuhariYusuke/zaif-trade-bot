"""
bitFlyer exchange adapter with dry-run support.

Implements IBroker interface with real API integration and dry-run simulation.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, Literal

import requests

from ztb.utils.rate_limiter import RateLimiter
from ztb.utils.errors import InsufficientFundsError, MinimumSizeError

from ..base.adapter import BaseExchangeAdapter
from ..base.broker_interfaces import Balance, Order, Position

logger = logging.getLogger(__name__)


# Type definitions for API responses
BitFlyerOrderResponse = Dict[str, Union[str, int, float]]
BitFlyerBalanceResponse = List[Dict[str, Union[str, float]]]
BitFlyerErrorResponse = Dict[str, str]


class BitFlyerAdapter(BaseExchangeAdapter):
    """
    bitFlyer exchange adapter with real API support and dry-run simulation.

    Supports both real trading and dry-run simulation for testing.
    """

    BASE_URL = "https://api.bitflyer.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        dry_run: bool = True,
        rate_limiter: Optional[RateLimiter] = None,
        fixed_price: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize bitFlyer adapter.

        Args:
            api_key: bitFlyer API key
            api_secret: bitFlyer API secret
            dry_run: If True, simulate all operations without real API calls
            rate_limiter: Rate limiter for API calls
            fixed_price: If set, always return this price in get_current_price (for testing)
            random_seed: If set, seed the random number generator for reproducibility
        """
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            dry_run=dry_run,
            rate_limiter=rate_limiter,
            fixed_price=fixed_price,
            random_seed=random_seed,
            requests_per_second=2.0,  # bitFlyer allows 200 calls per minute = ~3.33 per second
        )

        # Override default prices for bitFlyer
        self._current_prices: Dict[str, float] = {"BTC_JPY": 5000000.0}

    def _generate_signature(
        self, timestamp: str, method: str, path: str, body: str = ""
    ) -> str:
        """Generate HMAC-SHA256 signature for bitFlyer API.

        Args:
            timestamp: Request timestamp
            method: HTTP method
            path: API endpoint path
            body: Request body

        Returns:
            Hexadecimal signature string

        Raises:
            ValueError: If API secret is not set
        """
        if not self.api_secret:
            raise ValueError("API secret is required for authentication")

        message = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return signature

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Get authenticated headers for API request."""
        if not self.api_key:
            raise ValueError("API key is required for authentication")

        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path, body)

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-SIGN": signature,
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make authenticated API request to bitFlyer.

        Args:
            method: HTTP method
            path: API endpoint path
            data: Request data for POST requests

        Returns:
            API response

        Raises:
            Exception: For API errors
        """
        """Make authenticated API request to bitFlyer.

        Args:
            method: HTTP method
            path: API endpoint path
            data: Request data for POST requests

        Returns:
            API response

        Raises:
            Exception: For API errors
        """
        """Make authenticated API request to bitFlyer."""
        url = self.BASE_URL + path
        body = json.dumps(data) if data else ""

        headers = self._get_headers(method, path, body)

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=body, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"bitFlyer API request failed: {e}")
            raise

    # Abstract method implementations
    async def _get_balance_real(self, currency: Optional[str] = None) -> List[Balance]:
        """Get balance from bitFlyer API."""
        try:
            response = await self._make_request("GET", "/v1/me/getbalance")
            balances = []
            for balance_data in response if isinstance(response, list) else []:
                if not isinstance(balance_data, dict):
                    continue
                currency_code = str(balance_data.get("currency_code", ""))
                if currency and currency_code != currency:
                    continue

                balance = Balance(
                    currency=currency_code,
                    free=float(balance_data.get("available", 0)),
                    locked=float(balance_data.get("amount", 0))
                    - float(balance_data.get("available", 0)),
                    total=float(balance_data.get("amount", 0)),
                )
                balances.append(balance)
            return balances
        except Exception as e:
            logger.error(f"Failed to get balance from bitFlyer: {e}")
            raise

    async def _place_order_real(
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
        """Place order via bitFlyer API."""
        try:
            order_data = {
                "product_code": symbol,
                "child_order_type": order_type.upper(),
                "side": side.upper(),
                "size": quantity,
            }

            if order_type.lower() == "limit" and price is not None:
                order_data["price"] = price

            response = await self._make_request(
                "POST", "/v1/me/sendchildorder", order_data
            )

            order_id = str(response.get("child_order_acceptance_id", ""))
            if not order_id:
                raise ValueError("No order ID in response")

            order = Order(
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

            self._orders[order_id] = order
            return order
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to place order on bitFlyer: {error_msg}")
            
            # Check for specific API errors and raise appropriate exceptions
            if "insufficient funds" in error_msg.lower():
                logger.warning(f"Order failed due to insufficient funds: {error_msg}")
                raise InsufficientFundsError(f"Insufficient funds for {side} order of {quantity} {symbol}")
            elif "minimum" in error_msg.lower() or "size" in error_msg.lower():
                logger.warning(f"Order failed due to minimum size requirements: {error_msg}")
                raise MinimumSizeError(f"Order size {quantity} below minimum requirements")
            else:
                # Re-raise other errors
                raise

    async def _cancel_order_real(self, order_id: str) -> bool:
        """Cancel order via bitFlyer API."""
        try:
            cancel_data = {"child_order_acceptance_id": order_id}

            await self._make_request("POST", "/v1/me/cancelchildorder", cancel_data)

            if order_id in self._orders:
                self._orders[order_id].status = "cancelled"

            return True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to cancel order on bitFlyer: {error_msg}")
            
            # Check for specific API errors
            if "not found" in error_msg.lower() or "already cancelled" in error_msg.lower():
                logger.warning(f"Order cancellation failed (order may have already executed or been cancelled): {error_msg}")
                return False
            else:
                # Re-raise other errors
                raise

    async def _get_order_status_real(self, order_id: str) -> Optional[Order]:
        """Get order status from bitFlyer API."""
        try:
            params = {"child_order_acceptance_id": order_id}

            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            path = f"/v1/me/getchildorders?{query_string}"

            response = await self._make_request("GET", path)

            if not response:
                return None

            order_data = (
                response[0] if isinstance(response, list) and response else None
            )
            if not order_data:
                return None

            bf_state = order_data.get("child_order_state", "")
            if bf_state == "ACTIVE":
                status = "pending"
            elif bf_state == "COMPLETED":
                status = "filled"
            elif bf_state == "CANCELED":
                status = "cancelled"
            elif bf_state == "EXPIRED":
                status = "cancelled"
            elif bf_state == "REJECTED":
                status = "cancelled"
            else:
                status = "pending"

            order = Order(
                order_id=str(order_data.get("child_order_acceptance_id", "")),
                symbol=str(order_data.get("product_code", "")),
                side=str(order_data.get("side", "")).lower(),
                quantity=float(order_data.get("size", 0)),
                price=(
                    float(order_data.get("price", 0))
                    if order_data.get("price")
                    else None
                ),
                order_type=str(order_data.get("child_order_type", "")).lower(),
                status=status,
                client_order_id=None,
                sizing_reason=None,
                target_vol=None,
            )

            self._orders[order.order_id] = order
            return order
        except Exception as e:
            logger.error(f"Failed to get order status from bitFlyer: {e}")
            return None

    async def _get_open_orders_real(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders from bitFlyer API."""
        try:
            params = {}
            if symbol:
                params["product_code"] = symbol
            params["child_order_state"] = "ACTIVE"

            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            path = f"/v1/me/getchildorders?{query_string}"

            response = await self._make_request("GET", path)

            orders = []
            for order_data in response if isinstance(response, list) else []:
                order = Order(
                    order_id=str(order_data.get("child_order_acceptance_id", "")),
                    symbol=str(order_data.get("product_code", "")),
                    side=str(order_data.get("side", "")).lower(),
                    quantity=float(order_data.get("size", 0)),
                    price=(
                        float(order_data.get("price", 0))
                        if order_data.get("price")
                        else None
                    ),
                    order_type=str(order_data.get("child_order_type", "")).lower(),
                    status="pending",
                    client_order_id=None,
                    sizing_reason=None,
                    target_vol=None,
                )
                orders.append(order)
                self._orders[order.order_id] = order

            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders from bitFlyer: {e}")
            return []

    async def _get_positions_real(self) -> List[Position]:
        """Get positions from bitFlyer API."""
        try:
            response = await self._make_request("GET", "/v1/me/getpositions")

            positions = []
            for position_data in response if isinstance(response, list) else []:
                if not isinstance(position_data, dict):
                    continue

                position = Position(
                    symbol=str(position_data.get("product_code", "")),
                    quantity=float(position_data.get("size", 0)),
                    avg_price=float(position_data.get("price", 0)),
                    current_price=float(position_data.get("price", 0)),
                    pnl=float(position_data.get("pnl", 0)),
                )
                positions.append(position)

            return positions
        except Exception as e:
            logger.error(f"Failed to get positions from bitFlyer: {e}")
            return []

    async def _get_current_price_real(self, symbol: str) -> Optional[float]:
        """Get current price from bitFlyer API."""
        try:
            response = await self._make_request(
                "GET", f"/v1/ticker?product_code={symbol}"
            )

            if isinstance(response, dict) and "ltp" in response:
                price = float(response["ltp"])
                self._current_prices[symbol] = price
                return price
            else:
                logger.warning(f"Invalid ticker response format: {response}")
                return None
        except Exception as e:
            logger.error(f"Failed to get current price from bitFlyer: {e}")
            return None
