"""Order management component for live trading."""

from typing import TYPE_CHECKING

from ztb.utils.errors import validate_quantity
from ztb.utils.exceptions.custom_exceptions import ValidationError
from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class OrderManager:
    """Manages order execution and trade operations."""

    def __init__(self, live_trader: "LiveTrader"):
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
            # Send notification for demo trades
            self.live_trader._send_notification(
                "📈 Demo Trade Executed",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Mode: DEMO (no real trade)",
                "info",
            )
            return True

        # Enhanced error notification for live trading
        try:
            # TODO: Implement actual exchange API trading calls
            self.logger.warning(
                f"LIVE MODE: Trade execution not implemented yet - {side} {amount} BTC"
            )
            self.live_trader._send_notification(
                "⚠️ Live Trade Not Implemented",
                f"Would execute: {side.upper()} {amount} BTC\n"
                f"Please implement actual API calls\n"
                f"Position: {self.live_trader.position}, Entry: ¥{self.live_trader.entry_price:,.0f}",
                "warning",
            )
            # Still send trade info notification even though execution is not implemented
            self.live_trader._send_notification(
                "📈 Live Trade Info",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Position: {self.live_trader.position}\n"
                f"Entry Price: ¥{self.live_trader.entry_price:,.0f}",
                "info",
            )
            return False
        except Exception as e:
            # Critical error notification
            error_msg = f"CRITICAL: Trade execution failed - {str(e)}"
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
