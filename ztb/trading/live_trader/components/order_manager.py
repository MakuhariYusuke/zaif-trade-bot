"""Order management component for live trading.

013# D-1 FIX: OrderManager を CoincheckAdapter.place_order() に接続。
実取引が可能な状態にする。
"""

import asyncio
from typing import TYPE_CHECKING, Optional

from ztb.utils.errors import validate_quantity
from ztb.utils.exceptions.custom_exceptions import ValidationError
from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class OrderManager:
    """Manages order execution and trade operations.

    013# D-1: Live mode now calls exchange_adapter.place_order() directly.
    """

    def __init__(self, live_trader: "LiveTrader") -> None:
        """Initialize order manager with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

    def execute_trade(self, side: str, amount: float) -> bool:
        """Execute trade with enhanced error handling and notifications.

        Args:
            side: Trade side ('buy' or 'sell')
            amount: Trade amount in BTC

        Returns:
            True if trade executed successfully, False otherwise
        """
        # Input validation
        validate_quantity(amount, "amount")
        if side not in ["buy", "sell"]:
            raise ValidationError(f"Invalid side: {side}, must be 'buy' or 'sell'")

        if self.live_trader.demo_mode:
            self.logger.info(f"DEMO MODE: Would execute {side} {amount} BTC")
            self.live_trader._send_notification(
                "📈 Demo Trade Executed",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Mode: DEMO (no real trade)",
                "info",
            )
            return True

        # 013# D-1 FIX: Live trading via exchange_adapter.place_order()
        exchange_adapter = getattr(self.live_trader, "exchange_adapter", None)
        if exchange_adapter is None:
            self.logger.error(
                "LIVE MODE: exchange_adapter not available, cannot execute trade"
            )
            self.live_trader._send_notification(
                "🚨 Trade Failed: No Exchange Adapter",
                f"Cannot execute {side.upper()} {amount} BTC\n"
                f"Exchange adapter is not initialized",
                "error",
            )
            return False

        try:
            # Get current price for limit order
            current_price = getattr(self.live_trader, "_last_valid_price", None)
            if current_price is None:
                current_price = self.live_trader._current_prices.get("btc_jpy", 0.0)

            if current_price <= 0:
                self.logger.error("Cannot place order: no valid price available")
                self.live_trader._send_notification(
                    "🚨 Trade Failed: No Price",
                    f"Cannot determine current price for {side.upper()} order",
                    "error",
                )
                return False

            # v460 戦略: maker-only (指値注文)
            # CoincheckAdapter が post_only を付与して taker 約定を防止
            order = self._execute_order_async(
                exchange_adapter=exchange_adapter,
                symbol="btc_jpy",
                side=side,
                quantity=amount,
                price=current_price,
                order_type="limit",
            )

            if order is not None:
                self.logger.info(
                    f"LIVE: Order placed - {side.upper()} {amount} BTC "
                    f"@ ¥{current_price:,.0f} (order_id={order.order_id})"
                )
                self.live_trader._send_notification(
                    "📈 Live Order Placed",
                    f"Side: {side.upper()}\n"
                    f"Amount: {amount} BTC\n"
                    f"Price: ¥{current_price:,.0f}\n"
                    f"Order ID: {order.order_id}\n"
                    f"Type: limit (post_only/maker)",
                    "info",
                )
                return True
            else:
                self.logger.error(f"LIVE: Order returned None for {side} {amount} BTC")
                return False

        except Exception as e:
            error_msg = f"Trade execution failed: {str(e)}"
            self.logger.error(error_msg)
            self.live_trader._send_notification(
                "🚨 CRITICAL: Trade Execution Error",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Error: {str(e)}\n"
                f"Position: {self.live_trader.position}",
                "error",
            )
            return False

    def _execute_order_async(
        self,
        exchange_adapter: object,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "limit",
    ) -> Optional[object]:
        """Bridge sync OrderManager to async exchange_adapter.place_order().

        Returns:
            Order object on success, None on failure.
        """
        async def _place():
            return await exchange_adapter.place_order(  # type: ignore[union-attr]
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
            )

        try:
            # If already inside an event loop, use nest_asyncio pattern
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're inside an existing event loop (e.g. trading_loop)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _place())
                    return future.result(timeout=30)
            else:
                return asyncio.run(_place())
        except Exception as e:
            self.logger.error(f"Async order execution failed: {e}")
            raise

    def validate_trade_parameters(self, side: str, amount: float) -> None:
        """Validate trade parameters before execution.

        Args:
            side: Trade side
            amount: Trade amount

        Raises:
            ValidationError: If parameters are invalid
        """
        validate_quantity(amount, "amount")
        if side not in ["buy", "sell"]:
            raise ValidationError(f"Invalid trade side: {side}")
        if amount <= 0:
            raise ValidationError(f"Trade amount must be positive: {amount}")

    def get_trade_info(self) -> dict:
        """Get current trade information.

        Returns:
            Dictionary with current trade state
        """
        return {
            "position": self.live_trader.position,
            "entry_price": self.live_trader.entry_price,
            "total_pnl": self.live_trader.total_pnl,
            "trades_count": self.live_trader.trades_count,
            "demo_mode": self.live_trader.demo_mode,
        }
