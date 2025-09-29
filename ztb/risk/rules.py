"""
Risk management rules and validation.

Implements hard stops, trailing stops, and cooldown rules.
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .profiles import RiskLimits


class RiskRuleEngine:
    """Engine for evaluating and enforcing risk rules."""

    def __init__(self, limits: RiskLimits):
        """Initialize with risk limits."""
        self.limits = limits

        # State tracking
        self.daily_start_capital = 0.0
        self.daily_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.daily_loss = 0.0

        self.portfolio_value = 0.0
        self.portfolio_volatility = 0.0

        self.last_trade_time = 0
        self.trades_this_hour = 0
        self.hour_start_time = time.time()

        self.trailing_stop_level = None
        self.trailing_stop_distance = 0.0

        # Trade history for analysis
        self.trade_history: List[Dict] = []

    def reset_daily_tracking(self):
        """Reset daily loss tracking at start of new day."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if today_start > self.daily_start_time:
            self.daily_start_time = today_start
            self.daily_start_capital = self.portfolio_value
            self.daily_loss = 0.0

    def update_portfolio_state(self, current_value: float, volatility: float):
        """Update current portfolio state."""
        self.portfolio_value = current_value
        self.portfolio_volatility = volatility

        # Update daily loss
        if self.daily_start_capital > 0:
            self.daily_loss = self.daily_start_capital - current_value

    def check_daily_loss_limit(self) -> Tuple[bool, str]:
        """Check if daily loss limit is exceeded."""
        if self.daily_start_capital <= 0:
            return True, ""

        loss_pct = self.daily_loss / self.daily_start_capital
        if loss_pct >= self.limits.daily_loss_limit_pct:
            return False, f"Daily loss limit exceeded: {loss_pct:.1%} >= {self.limits.daily_loss_limit_pct:.1%}"

        return True, ""

    def check_max_drawdown(self, peak_value: float) -> Tuple[bool, str]:
        """Check if maximum drawdown limit is exceeded."""
        if peak_value <= 0:
            return True, ""

        drawdown = (peak_value - self.portfolio_value) / peak_value
        if drawdown >= self.limits.max_drawdown_pct:
            return False, f"Max drawdown exceeded: {drawdown:.1%} >= {self.limits.max_drawdown_pct:.1%}"

        return True, ""

    def check_position_size(self, position_notional: float) -> Tuple[bool, str]:
        """Check if position size exceeds limits."""
        if position_notional > self.limits.max_position_notional:
            return False, f"Position size exceeds limit: {position_notional:,.0f} > {self.limits.max_position_notional:,.0f}"

        return True, ""

    def check_single_trade_size(self, trade_notional: float) -> Tuple[bool, str]:
        """Check if single trade size exceeds limits."""
        if self.portfolio_value <= 0:
            return True, ""

        trade_pct = trade_notional / self.portfolio_value
        if trade_pct > self.limits.max_single_trade_pct:
            return False, f"Trade size exceeds limit: {trade_pct:.1%} > {self.limits.max_single_trade_pct:.1%}"

        return True, ""

    def check_trade_frequency(self) -> Tuple[bool, str]:
        """Check trade frequency limits."""
        current_time = time.time()

        # Reset hourly counter if needed
        if current_time - self.hour_start_time >= 3600:
            self.trades_this_hour = 0
            self.hour_start_time = current_time

        if self.trades_this_hour >= self.limits.max_trades_per_hour:
            return False, f"Trade frequency limit exceeded: {self.trades_this_hour} >= {self.limits.max_trades_per_hour}"

        # Check minimum interval
        if current_time - self.last_trade_time < self.limits.min_trade_interval_sec:
            return False, f"Minimum trade interval not met: {current_time - self.last_trade_time:.0f}s < {self.limits.min_trade_interval_sec}s"

        return True, ""

    def check_volatility_limit(self) -> Tuple[bool, str]:
        """Check portfolio volatility against limits."""
        if self.portfolio_volatility > self.limits.max_volatility_pct:
            return False, f"Portfolio volatility exceeds limit: {self.portfolio_volatility:.1%} > {self.limits.max_volatility_pct:.1%}"

        return True, ""

    def check_performance_thresholds(self, sharpe_ratio: float) -> Tuple[bool, str]:
        """Check if performance meets minimum thresholds."""
        if sharpe_ratio < self.limits.required_sharpe_ratio:
            return False, f"Sharpe ratio below threshold: {sharpe_ratio:.2f} < {self.limits.required_sharpe_ratio:.2f}"

        return True, ""

    def update_trailing_stop(self, current_price: float, position_side: str):
        """Update trailing stop level."""
        if position_side not in ['long', 'short']:
            return

        # Initialize trailing stop
        if self.trailing_stop_level is None:
            if position_side == 'long':
                self.trailing_stop_level = current_price * (1 - self.limits.stop_loss_pct)
            else:  # short
                self.trailing_stop_level = current_price * (1 + self.limits.stop_loss_pct)
            return

        # Update trailing stop
        if position_side == 'long':
            # For long positions, trail below the highest price
            new_stop = current_price * (1 - self.limits.stop_loss_pct)
            if new_stop > self.trailing_stop_level:
                self.trailing_stop_level = new_stop
        else:  # short
            # For short positions, trail above the lowest price
            new_stop = current_price * (1 + self.limits.stop_loss_pct)
            if new_stop < self.trailing_stop_level:
                self.trailing_stop_level = new_stop

    def check_trailing_stop(self, current_price: float, position_side: str) -> Tuple[bool, str]:
        """Check if trailing stop is hit."""
        if self.trailing_stop_level is None:
            return True, ""

        stop_hit = False
        if position_side == 'long' and current_price <= self.trailing_stop_level:
            stop_hit = True
        elif position_side == 'short' and current_price >= self.trailing_stop_level:
            stop_hit = True

        if stop_hit:
            return False, f"Trailing stop hit at {self.trailing_stop_level:.2f}"

        return True, ""

    def check_take_profit(self, entry_price: float, current_price: float, position_side: str) -> Tuple[bool, str]:
        """Check if take profit target is reached."""
        if position_side == 'long':
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price

        if profit_pct >= self.limits.take_profit_pct:
            return False, f"Take profit target reached: {profit_pct:.1%} >= {self.limits.take_profit_pct:.1%}"

        return True, ""

    def record_trade(self, trade_data: Dict):
        """Record a completed trade."""
        self.trade_history.append({
            **trade_data,
            'timestamp': time.time()
        })

        self.last_trade_time = time.time()
        self.trades_this_hour += 1

    def get_cooldown_period(self) -> int:
        """Get cooldown period after losses (in seconds)."""
        # Simple cooldown based on recent losses
        recent_trades = [t for t in self.trade_history[-5:] if t.get('pnl', 0) < 0]
        if len(recent_trades) >= 3:
            return 300  # 5 minutes cooldown after 3 consecutive losses

        return 0

    def validate_trade(self,
                      trade_notional: float,
                      position_notional: float,
                      peak_value: float,
                      sharpe_ratio: Optional[float] = None) -> Tuple[bool, str]:
        """
        Comprehensive trade validation against all risk rules.

        Returns: (is_allowed, reason_if_rejected)
        """

        # Update daily tracking
        self.reset_daily_tracking()

        # Check all rules
        checks = [
            self.check_daily_loss_limit(),
            self.check_max_drawdown(peak_value),
            self.check_position_size(position_notional),
            self.check_single_trade_size(trade_notional),
            self.check_trade_frequency(),
            self.check_volatility_limit(),
        ]

        if sharpe_ratio is not None:
            checks.append(self.check_performance_thresholds(sharpe_ratio))

        # Check cooldown
        cooldown = self.get_cooldown_period()
        if cooldown > 0:
            time_since_last_trade = time.time() - self.last_trade_time
            if time_since_last_trade < cooldown:
                checks.append((False, f"In cooldown period: {cooldown - time_since_last_trade:.0f}s remaining"))

        # Return first failure
        for allowed, reason in checks:
            if not allowed:
                return False, reason

        return True, ""