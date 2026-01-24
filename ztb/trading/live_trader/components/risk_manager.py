"""Risk management component for live trading."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ztb.metrics.metrics import max_drawdown as calculate_max_drawdown
from ztb.metrics.metrics import sharpe_ratio as calculate_sharpe_ratio
from ztb.metrics.metrics import calculate_volatility, rolling_statistics
from ztb.trading.risk.risk_manager import RiskManager as BaseRiskManager
from ztb.trading.types import PositionManagementConfig
from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import safe_to_float

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class RiskManager(BaseRiskManager):
    """Manages trading risk limits and emergency stops."""

    def __init__(self, live_trader: "LiveTrader"):
        """Initialize risk manager with reference to live trader."""
        # Create PositionManagementConfig from live trader config
        pm_config = PositionManagementConfig(
            max_portfolio_risk_pct=live_trader.config.get("max_portfolio_risk_pct", 0.10),
            max_single_position_risk_pct=live_trader.config.get("max_single_position_risk_pct", 0.05),
            stop_loss_pct=live_trader.config.get("stop_loss_pct", 0.02),
            take_profit_pct=live_trader.config.get("take_profit_pct", 0.05),
            capital_buffer_pct=live_trader.config.get("capital_buffer_pct", 0.05),
            min_signal_strength=live_trader.config.get("min_signal_strength", 0.5),
            max_volatility_threshold=live_trader.config.get("max_volatility_threshold", 0.05),
            default_position_size_pct=live_trader.config.get("default_position_size_pct", 0.02),
        )

        # Initialize base risk manager
        super().__init__(pm_config)

        self.live_trader = live_trader
        self.logger = get_logger(__name__)

        # Risk limits from config
        self.max_daily_loss = safe_to_float(
            live_trader.config.get("max_daily_loss", 10000.0)
        )
        self.max_daily_trades = int(
            safe_to_float(live_trader.config.get("max_daily_trades", 50))
        )
        self.max_trades_per_hour = int(
            safe_to_float(live_trader.config.get("max_trades_per_hour", 6))
        )
        self.emergency_stop_loss = safe_to_float(
            live_trader.config.get("emergency_stop_loss", 0.05)
        )

        # Tracking variables
        self.daily_start_pnl = live_trader.daily_start_pnl
        self.daily_trades = live_trader.daily_trades
        self.hourly_trades = 0
        self.last_trade_hour = datetime.now().hour

        # Statistical tracking
        self.pnl_history = []  # Track PnL over time for statistics
        self.last_pnl = live_trader.total_pnl
        # Protocol attributes
        self.test_mode: bool = bool(live_trader.config.get("test_mode", False))
        # Portfolio value estimate: use position_manager or fallback
        self.portfolio_value: float = float(
            getattr(live_trader, "total_pnl", 0.0)
            + getattr(live_trader, "initial_portfolio_value", 0.0)
        )

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
            loss_ratio = (
                abs(current_price - self.live_trader.entry_price)
                / self.live_trader.entry_price
            )
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
            self.check_daily_loss_limit()
            and self.check_daily_trade_limit()
            and self.check_hourly_trade_limit()
            and self.check_emergency_stop_loss(current_price)
        )

    # Protocol methods
    def should_open_position(
        self,
        signal_strength: float,
        market_volatility: float,
        current_portfolio_value: float,
    ) -> bool:
        """Return True if risk limits allow entering new position.

        Uses existing can_trade() logic and a basic signal strength threshold.
        """
        if self.test_mode:
            return True

        # If we have a last valid price, use it; otherwise use a fallback
        current_price = getattr(self.live_trader, "_last_valid_price", 0.0)
        if current_price <= 0:
            current_price = getattr(self.live_trader, "entry_price", 0.0) or 0.0

        # If `can_trade` requires a price, we call it
        if not self.can_trade(current_price):
            return False

        # Minimal signal threshold
        min_strength = float(self.live_trader.config.get("min_signal_strength", 0.6))
        if signal_strength < min_strength:
            return False

        return True

    def should_close_position(
        self,
        position_data: Dict[str, Any],
        current_price: float,
        current_portfolio_value: float,
    ) -> Tuple[bool, str]:
        # Use emergency stop and explicit stops if available in position_data
        stop_loss = position_data.get("stop_loss")
        take_profit = position_data.get("take_profit")
        pos_type = position_data.get("type", "long")
        if stop_loss is not None and pos_type == "long" and current_price <= stop_loss:
            return True, "stop_loss"
        if stop_loss is not None and pos_type == "short" and current_price >= stop_loss:
            return True, "stop_loss"
        if (
            take_profit is not None
            and pos_type == "long"
            and current_price >= take_profit
        ):
            return True, "take_profit"
        if (
            take_profit is not None
            and pos_type == "short"
            and current_price <= take_profit
        ):
            return True, "take_profit"
        if not self.check_emergency_stop_loss(current_price):
            return True, "emergency_stop"
        return False, ""

    def get_risk_adjusted_position_size(
        self, signal_strength: float, market_volatility: float
    ) -> float:
        # Prefer using position manager if available
        try:
            pm = getattr(self.live_trader, "position_manager", None)
            if pm and hasattr(pm, "min_unit_manager"):
                # Use minimal unit as rough size
                min_unit = getattr(
                    pm.min_unit_manager, "get_min_unit", lambda a, b: 0.0001
                )("coincheck", "btc_jpy")
                # Calculate a conservative position size (in units)
                base = float(self.live_trader.config.get("default_base_position", 0.01))
                return min(base, float(min_unit))
        except Exception:
            pass
        # fallback
        return float(self.live_trader.config.get("default_base_position", 0.01))

    def calculate_atr_stop_levels(
        self, data: Optional[pd.DataFrame], entry_price: float, position_type: str
    ) -> Tuple[float, float]:
        if data is not None and ("atr" in data.columns):
            base_atr = float(data["atr"].iloc[-1])
        else:
            base_atr = float(entry_price * 0.02)
        stop_loss = entry_price - base_atr * float(
            self.live_trader.config.get("stop_loss_atr_multiplier", 2.0)
        )
        take_profit = entry_price + base_atr * float(
            self.live_trader.config.get("take_profit_atr_multiplier", 4.0)
        )
        if position_type == "short":
            stop_loss, take_profit = -stop_loss, -take_profit
        return float(stop_loss), float(take_profit)

    def update_risk_metrics(
        self, trade_result: Optional[Dict[str, Any]] = None
    ) -> None:
        if trade_result:
            pnl = float(trade_result.get("pnl", 0.0))
            # Update live trader total pnl and our tracked history
            self.live_trader.total_pnl = float(
                getattr(self.live_trader, "total_pnl", 0.0) + pnl
            )
            self.last_pnl = self.live_trader.total_pnl
            # Update portfolio estimation
            self.portfolio_value = float(
                getattr(self.live_trader, "initial_portfolio_value", 0.0)
                + self.live_trader.total_pnl
            )
        else:
            # Sync with live_trader if no trade result provided
            self.portfolio_value = float(
                getattr(self.live_trader, "initial_portfolio_value", 0.0)
                + getattr(self.live_trader, "total_pnl", 0.0)
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
        cumulative_pnl = [
            float(sum(self.pnl_history[: i + 1])) for i in range(len(self.pnl_history))
        ]

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
            volatility = calculate_volatility(
                self.pnl_history, window=min(20, len(self.pnl_history))
            )
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
