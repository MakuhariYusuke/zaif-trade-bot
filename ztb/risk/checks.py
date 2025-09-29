"""
Risk management checks for pre and post-trade validation.

Provides hooks for integrating risk management into trading systems.
"""

from typing import Dict, Any, Optional, Tuple, Callable
from .rules import RiskRuleEngine
from .profiles import RiskLimits


class RiskChecker:
    """Pre and post-trade risk validation."""

    def __init__(self, limits: RiskLimits):
        """Initialize with risk limits."""
        self.engine = RiskRuleEngine(limits)

    def pre_trade_check(self,
                       trade_notional: float,
                       position_notional: float,
                       peak_value: float,
                       sharpe_ratio: Optional[float] = None) -> Tuple[bool, str]:
        """
        Pre-trade risk validation.

        Args:
            trade_notional: Size of proposed trade
            position_notional: Current position size
            peak_value: Portfolio peak value
            sharpe_ratio: Current Sharpe ratio (optional)

        Returns:
            (allowed, rejection_reason)
        """
        return self.engine.validate_trade(
            trade_notional=trade_notional,
            position_notional=position_notional,
            peak_value=peak_value,
            sharpe_ratio=sharpe_ratio
        )

    def post_trade_update(self,
                         current_value: float,
                         volatility: float,
                         trade_data: Optional[Dict[str, Any]] = None):
        """
        Post-trade state update.

        Args:
            current_value: Current portfolio value
            volatility: Current portfolio volatility
            trade_data: Trade execution details (optional)
        """
        self.engine.update_portfolio_state(current_value, volatility)

        if trade_data:
            self.engine.record_trade(trade_data)

    def update_trailing_stop(self, current_price: float, position_side: str):
        """Update trailing stop level."""
        self.engine.update_trailing_stop(current_price, position_side)

    def check_trailing_stop(self, current_price: float, position_side: str) -> Tuple[bool, str]:
        """Check if trailing stop is hit."""
        return self.engine.check_trailing_stop(current_price, position_side)

    def check_take_profit(self, entry_price: float, current_price: float, position_side: str) -> Tuple[bool, str]:
        """Check if take profit target is reached."""
        return self.engine.check_take_profit(entry_price, current_price, position_side)

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        return {
            'daily_loss': self.engine.daily_loss,
            'daily_loss_limit': self.engine.limits.daily_loss_limit_pct,
            'portfolio_value': self.engine.portfolio_value,
            'portfolio_volatility': self.engine.portfolio_volatility,
            'trades_this_hour': self.engine.trades_this_hour,
            'max_trades_per_hour': self.engine.limits.max_trades_per_hour,
            'trailing_stop_level': self.engine.trailing_stop_level,
            'cooldown_period': self.engine.get_cooldown_period()
        }


class RiskManager:
    """High-level risk management coordinator."""

    def __init__(self, profile_name: str = "balanced"):
        """Initialize with risk profile."""
        from .profiles import get_risk_profile
        self.limits = get_risk_profile(profile_name)
        self.checker = RiskChecker(self.limits)

        # Callbacks for integration
        self.on_risk_violation: Optional[Callable[[str], None]] = None
        self.on_trade_blocked: Optional[Callable[[str], None]] = None

    def validate_and_execute_trade(self,
                                  trade_func: Callable,
                                  trade_notional: float,
                                  position_notional: float,
                                  peak_value: float,
                                  **trade_kwargs) -> Tuple[bool, Any, str]:
        """
        Validate trade and execute if allowed.

        Args:
            trade_func: Function to execute the trade
            trade_notional: Proposed trade size
            position_notional: Current position size
            peak_value: Portfolio peak value
            **trade_kwargs: Arguments for trade_func

        Returns:
            (success, result, message)
        """
        # Pre-trade check
        allowed, reason = self.checker.pre_trade_check(
            trade_notional=trade_notional,
            position_notional=position_notional,
            peak_value=peak_value
        )

        if not allowed:
            if self.on_trade_blocked:
                self.on_trade_blocked(reason)
            return False, None, reason

        try:
            # Execute trade
            result = trade_func(**trade_kwargs)
            return True, result, "Trade executed successfully"

        except Exception as e:
            error_msg = f"Trade execution failed: {str(e)}"
            if self.on_risk_violation:
                self.on_risk_violation(error_msg)
            return False, None, error_msg

    def monitor_position(self,
                        current_price: float,
                        entry_price: float,
                        position_side: str) -> Dict[str, Any]:
        """
        Monitor position for stop loss/take profit triggers.

        Returns:
            Dict with trigger status
        """
        triggers = {}

        # Check trailing stop
        ts_allowed, ts_reason = self.checker.check_trailing_stop(current_price, position_side)
        triggers['trailing_stop'] = {
            'triggered': not ts_allowed,
            'reason': ts_reason
        }

        # Check take profit
        tp_allowed, tp_reason = self.checker.check_take_profit(entry_price, current_price, position_side)
        triggers['take_profit'] = {
            'triggered': not tp_allowed,
            'reason': tp_reason
        }

        return triggers

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive risk status report."""
        return {
            'profile': self.limits,
            'current_status': self.checker.get_risk_status(),
            'limits': {
                'max_position_notional': self.limits.max_position_notional,
                'max_single_trade_pct': self.limits.max_single_trade_pct,
                'daily_loss_limit_pct': self.limits.daily_loss_limit_pct,
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'max_trades_per_hour': self.limits.max_trades_per_hour,
                'min_trade_interval_sec': self.limits.min_trade_interval_sec,
                'max_volatility_pct': self.limits.max_volatility_pct,
                'required_sharpe_ratio': self.limits.required_sharpe_ratio,
                'stop_loss_pct': self.limits.stop_loss_pct,
                'take_profit_pct': self.limits.take_profit_pct
            }
        }