"""Risk management component for live trading."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Any

from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import safe_to_float
from ztb.utils.statistics import calculate_volatility, calculate_sharpe_ratio, calculate_max_drawdown, rolling_statistics

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class RiskManager:
    """Manages trading risk limits and emergency stops."""

    def __init__(self, live_trader: "LiveTrader"):
        """Initialize risk manager with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

        # Risk limits from config
        self.max_daily_loss = safe_to_float(live_trader.config.get("max_daily_loss", 10000.0))
        self.max_daily_trades = int(safe_to_float(live_trader.config.get("max_daily_trades", 50)))
        self.max_trades_per_hour = int(safe_to_float(live_trader.config.get("max_trades_per_hour", 6)))
        self.emergency_stop_loss = safe_to_float(live_trader.config.get("emergency_stop_loss", 0.05))

        # Tracking variables
        self.daily_start_pnl = live_trader.daily_start_pnl
        self.daily_trades = live_trader.daily_trades
        self.hourly_trades = 0
        self.last_trade_hour = datetime.now().hour

        # Statistical tracking
        self.pnl_history = []  # Track PnL over time for statistics
        self.last_pnl = live_trader.total_pnl

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded.

        Returns:
            True if trading should continue, False if stopped due to loss limit
        """
        current_pnl = self.live_trader.total_pnl - self.daily_start_pnl
        if current_pnl <= -self.max_daily_loss:
            self.logger.critical(
                f"DAILY LOSS LIMIT EXCEEDED: {current_pnl:.2f} JPY <= -{self.max_daily_loss:.2f} JPY"
            )
            self.live_trader._send_notification(
                "🚨 EMERGENCY STOP: Daily Loss Limit",
                f"Current P&L: ¥{current_pnl:,.2f}\n"
                f"Limit: ¥{self.max_daily_loss:,.2f}\n"
                f"Trading stopped to prevent further losses.",
                "error",
            )
            return False
        return True

    def check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit has been exceeded.

        Returns:
            True if trading should continue, False if stopped due to trade limit
        """
        if self.daily_trades >= self.max_daily_trades:
            self.logger.warning(
                f"DAILY TRADE LIMIT REACHED: {self.daily_trades}/{self.max_daily_trades}"
            )
            self.live_trader._send_notification(
                "⚠️ Daily Trade Limit Reached",
                f"Trades today: {self.daily_trades}/{self.max_daily_trades}\n"
                f"Trading paused until next day.",
                "warning",
            )
            return False
        return True

    def check_hourly_trade_limit(self) -> bool:
        """Check if hourly trade limit has been exceeded.

        Returns:
            True if trading should continue, False if stopped due to trade limit
        """
        current_hour = datetime.now().hour
        if current_hour != self.last_trade_hour:
            self.hourly_trades = 0
            self.last_trade_hour = current_hour

        if self.hourly_trades >= self.max_trades_per_hour:
            self.logger.warning(
                f"HOURLY TRADE LIMIT REACHED: {self.hourly_trades}/{self.max_trades_per_hour}"
            )
            return False
        return True

    def check_emergency_stop_loss(self, current_price: float) -> bool:
        """Check if emergency stop loss should be triggered.

        Args:
            current_price: Current market price

        Returns:
            True if trading should continue, False if emergency stop triggered
        """
        if self.live_trader.position != 0 and self.live_trader.entry_price > 0:
            loss_ratio = abs(current_price - self.live_trader.entry_price) / self.live_trader.entry_price
            if loss_ratio >= self.emergency_stop_loss:
                self.logger.critical(
                    f"EMERGENCY STOP LOSS TRIGGERED: {loss_ratio:.3f} >= {self.emergency_stop_loss:.3f}"
                )
                self.live_trader._send_notification(
                    "🚨 EMERGENCY STOP LOSS",
                    f"Loss ratio: {loss_ratio:.1%}\n"
                    f"Entry: ¥{self.live_trader.entry_price:,.0f}\n"
                    f"Current: ¥{current_price:,.0f}\n"
                    f"Position: {self.live_trader.position:.4f} BTC",
                    "error",
                )
                return False
        return True

    def can_trade(self, current_price: float) -> bool:
        """Check if all risk limits allow trading to continue.

        Args:
            current_price: Current market price

        Returns:
            True if trading is allowed, False if any risk limit is violated
        """
        return (
            self.check_daily_loss_limit() and
            self.check_daily_trade_limit() and
            self.check_hourly_trade_limit() and
            self.check_emergency_stop_loss(current_price)
        )

    def record_trade(self) -> None:
        """Record that a trade has been executed for limit tracking."""
        self.daily_trades += 1
        self.hourly_trades += 1
        self.live_trader.daily_trades = self.daily_trades

        # Record PnL change for statistics
        current_pnl = self.live_trader.total_pnl
        pnl_change = current_pnl - self.last_pnl
        self.pnl_history.append(pnl_change)
        self.last_pnl = current_pnl

        # Keep only recent history (last 1000 trades)
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]

    def reset_daily_limits(self) -> None:
        """Reset daily limits (called at start of new trading day)."""
        self.daily_start_pnl = self.live_trader.total_pnl
        self.daily_trades = 0
        self.live_trader.daily_start_pnl = self.daily_start_pnl
        self.live_trader.daily_trades = self.daily_trades
        self.logger.info("Daily risk limits reset")

    def calculate_pnl_statistics(self) -> Dict[str, Any]:
        """Calculate statistical metrics for PnL history.

        Returns:
            Dictionary with statistical metrics
        """
        if not self.pnl_history:
            return {
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
            }

        # Calculate returns (cumulative PnL as portfolio value proxy)
        cumulative_pnl = [float(sum(self.pnl_history[:i+1])) for i in range(len(self.pnl_history))]

        # Calculate rolling statistics
        rolling_window = min(20, len(self.pnl_history))
        if rolling_window >= 5:  # Need minimum window for meaningful statistics
            rolling_stats = rolling_statistics(self.pnl_history, window=rolling_window)
            rolling_volatility = rolling_stats.get("std", [])
            rolling_mean = rolling_stats.get("mean", [])
        else:
            rolling_volatility = []
            rolling_mean = []

        # Volatility (rolling standard deviation of returns)
        if len(self.pnl_history) >= 20:
            volatility = calculate_volatility(self.pnl_history, window=min(20, len(self.pnl_history)))
            current_volatility = volatility[-1] if volatility else 0.0
        else:
            current_volatility = 0.0

        # Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(self.pnl_history)

        # Max drawdown
        drawdown_stats = calculate_max_drawdown(cumulative_pnl)
        max_drawdown = drawdown_stats["max_drawdown"]

        # Win rate
        winning_trades = sum(1 for pnl in self.pnl_history if pnl > 0)
        win_rate = winning_trades / len(self.pnl_history) if self.pnl_history else 0.0

        return {
            "volatility": current_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.pnl_history),
            "win_rate": win_rate,
            "rolling_volatility": rolling_volatility[-1] if rolling_volatility else 0.0,
            "rolling_mean_return": rolling_mean[-1] if rolling_mean else 0.0,
        }

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status for monitoring.

        Returns:
            Dictionary with current risk metrics
        """
        current_pnl = self.live_trader.total_pnl - self.daily_start_pnl
        statistics = self.calculate_pnl_statistics()

        return {
            "daily_pnl": current_pnl,
            "daily_loss_limit": self.max_daily_loss,
            "daily_trades": self.daily_trades,
            "daily_trade_limit": self.max_daily_trades,
            "hourly_trades": self.hourly_trades,
            "hourly_trade_limit": self.max_trades_per_hour,
            "emergency_stop_loss": self.emergency_stop_loss,
            "position": self.live_trader.position,
            "entry_price": self.live_trader.entry_price,
            "statistics": statistics,
        }