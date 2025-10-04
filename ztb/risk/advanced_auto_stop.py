#!/usr/bin/env python3
"""
Advanced Auto-Stop System for Live Trading.

Implements sophisticated risk management with multiple stop mechanisms:
- Volatility-based stops
- Drawdown-based stops
- Time-based stops
- Performance-based stops
- Market condition-based stops
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StopReason(Enum):
    """Reasons for stopping trading."""

    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_LIMIT = "drawdown_limit"
    TIME_LIMIT = "time_limit"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MARKET_CONDITION = "market_condition"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class StopCondition:
    """Configuration for a stop condition."""

    name: str
    enabled: bool = True
    threshold: float = 0.0
    window_size: int = 60  # minutes
    cooldown_period: int = 300  # seconds
    severity: str = "warning"  # warning, critical, emergency


class AdvancedAutoStop:
    """Advanced automatic stop system for live trading."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the auto-stop system.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or self._get_default_config()

        # Stop conditions
        self.stop_conditions = self._initialize_stop_conditions()

        # State tracking
        self.is_active = True
        self.last_stop_time: Optional[datetime] = None
        self.stop_reason: Optional[StopReason] = None
        self.cooldown_until: Optional[datetime] = None

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.price_history: List[Tuple[datetime, float]] = []

        # Risk metrics
        self.current_drawdown = 0.0
        self.volatility = 0.0
        self.consecutive_losses = 0
        self.sharpe_ratio = 0.0

        logger.info("Advanced Auto-Stop system initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "volatility_stop": {
                "enabled": True,
                "threshold": 0.05,  # 5% volatility
                "window_size": 60,  # 1 hour
                "cooldown_period": 1800,  # 30 minutes
                "severity": "warning",
            },
            "drawdown_stop": {
                "enabled": True,
                "threshold": 0.10,  # 10% drawdown
                "window_size": 1440,  # 1 day
                "cooldown_period": 3600,  # 1 hour
                "severity": "critical",
            },
            "time_stop": {
                "enabled": True,
                "threshold": 28800,  # 8 hours
                "cooldown_period": 7200,  # 2 hours
                "severity": "warning",
            },
            "performance_stop": {
                "enabled": True,
                "threshold": -0.05,  # -5% performance degradation
                "window_size": 240,  # 4 hours
                "cooldown_period": 1800,  # 30 minutes
                "severity": "warning",
            },
            "market_condition_stop": {
                "enabled": True,
                "volatility_multiplier": 2.0,
                "trend_strength_threshold": 0.7,
                "cooldown_period": 3600,  # 1 hour
                "severity": "warning",
            },
            "consecutive_losses_stop": {
                "enabled": True,
                "threshold": 5,  # 5 consecutive losses
                "cooldown_period": 1800,  # 30 minutes
                "severity": "critical",
            },
        }

    def _initialize_stop_conditions(self) -> Dict[str, StopCondition]:
        """Initialize stop conditions from config."""
        conditions = {}

        for key, config in self.config.items():
            if isinstance(config, dict) and "enabled" in config:
                conditions[key] = StopCondition(
                    name=key,
                    enabled=config.get("enabled", True),
                    threshold=config.get("threshold", 0.0),
                    window_size=config.get("window_size", 60),
                    cooldown_period=config.get("cooldown_period", 300),
                    severity=config.get("severity", "warning"),
                )

        return conditions

    def update_market_data(self, timestamp: datetime, price: float) -> None:
        """
        Update market data for analysis.

        Args:
            timestamp: Current timestamp
            price: Current price
        """
        self.price_history.append((timestamp, price))

        # Keep only recent data (last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.price_history = [
            (ts, p) for ts, p in self.price_history if ts > cutoff_time
        ]

        # Update volatility
        if len(self.price_history) >= 10:
            prices = [p for _, p in self.price_history[-60:]]  # Last hour
            if len(prices) > 1:
                returns = np.diff(np.log(prices))
                self.volatility = np.std(returns) * np.sqrt(
                    60
                )  # Annualized hourly volatility

    def update_trade_result(self, pnl: float, trade_info: Dict[str, Any]) -> None:
        """
        Update trade result for risk analysis.

        Args:
            pnl: Profit/Loss from the trade
            trade_info: Additional trade information
        """
        trade_record = {"timestamp": datetime.now(), "pnl": pnl, "info": trade_info}
        self.trade_history.append(trade_record)

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Keep only recent trades (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.trade_history = [
            t for t in self.trade_history if t["timestamp"] > cutoff_time
        ]

        # Update drawdown
        self._update_drawdown()

    def _update_drawdown(self) -> None:
        """Update current drawdown calculation."""
        if not self.performance_history:
            return

        # Calculate running maximum
        cumulative_pnl = np.cumsum([t["pnl"] for t in self.trade_history])
        running_max = np.maximum.accumulate(cumulative_pnl)

        # Current drawdown
        current_value = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
        peak_value = running_max[-1] if len(running_max) > 0 else 0

        if peak_value > 0:
            self.current_drawdown = (peak_value - current_value) / peak_value
        else:
            self.current_drawdown = 0.0

    def check_stop_conditions(self) -> Tuple[bool, Optional[StopReason], str]:
        """
        Check all stop conditions.

        Returns:
            Tuple of (should_stop, reason, message)
        """
        # Check cooldown period
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds()
            return False, None, f"Cooldown active: {remaining:.0f}s remaining"

        # Check each condition
        for condition_name, condition in self.stop_conditions.items():
            if not condition.enabled:
                continue

            should_stop, reason, message = self._check_single_condition(
                condition_name, condition
            )
            if should_stop and reason is not None:
                self._trigger_stop(reason, condition.cooldown_period, message)
                return True, reason, message

        return False, None, "All conditions normal"

    def _check_single_condition(
        self, name: str, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check a single stop condition."""
        if name == "volatility_stop":
            return self._check_volatility_stop(condition)
        elif name == "drawdown_stop":
            return self._check_drawdown_stop(condition)
        elif name == "time_stop":
            return self._check_time_stop(condition)
        elif name == "performance_stop":
            return self._check_performance_stop(condition)
        elif name == "market_condition_stop":
            return self._check_market_condition_stop(condition)
        elif name == "consecutive_losses_stop":
            return self._check_consecutive_losses_stop(condition)

        return False, None, f"Unknown condition: {name}"

    def _check_volatility_stop(
        self, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check volatility-based stop."""
        if self.volatility > condition.threshold:
            return (
                True,
                StopReason.VOLATILITY_SPIKE,
                f"Volatility {self.volatility:.2%} exceeds threshold {condition.threshold:.2%}",
            )
        return False, None, "Volatility normal"

    def _check_drawdown_stop(
        self, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check drawdown-based stop."""
        if self.current_drawdown > condition.threshold:
            return (
                True,
                StopReason.DRAWDOWN_LIMIT,
                f"Drawdown {self.current_drawdown:.2%} exceeds threshold {condition.threshold:.2%}",
            )
        return False, None, "Drawdown within limits"

    def _check_time_stop(
        self, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check time-based stop."""
        if self.last_stop_time:
            elapsed = (datetime.now() - self.last_stop_time).total_seconds()
            if elapsed > condition.threshold:
                return (
                    True,
                    StopReason.TIME_LIMIT,
                    f"Trading time {elapsed/3600:.1f} hours exceeds limit {condition.threshold/3600:.1f} hours",
                )
        return False, None, "Time limit not reached"

    def _check_performance_stop(
        self, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check performance-based stop."""
        if len(self.trade_history) < 10:
            return False, None, "Insufficient trade history"

        recent_trades = self.trade_history[-20:]  # Last 20 trades
        recent_pnl = sum(t["pnl"] for t in recent_trades)
        recent_performance = recent_pnl / len(recent_trades)

        if recent_performance < condition.threshold:
            return (
                True,
                StopReason.PERFORMANCE_DEGRADATION,
                f"Recent performance {recent_performance:.2%} below threshold {condition.threshold:.2%}",
            )
        return False, None, "Performance acceptable"

    def _check_market_condition_stop(
        self, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check market condition-based stop."""
        if len(self.price_history) < 50:
            return False, None, "Insufficient price history"

        # Check if volatility is abnormally high
        normal_volatility = np.mean([self.volatility] * 10)  # Simplified
        if self.volatility > normal_volatility * condition.threshold:
            return (
                True,
                StopReason.MARKET_CONDITION,
                f"Volatility {self.volatility:.2%} exceeds normal range",
            )
        return False, None, "Market conditions normal"

    def _check_consecutive_losses_stop(
        self, condition: StopCondition
    ) -> Tuple[bool, Optional[StopReason], str]:
        """Check consecutive losses stop."""
        if self.consecutive_losses >= condition.threshold:
            return (
                True,
                StopReason.CONSECUTIVE_LOSSES,
                f"{self.consecutive_losses} consecutive losses",
            )
        return False, None, "Loss streak within limits"

    def _trigger_stop(
        self, reason: StopReason, cooldown_seconds: int, message: str
    ) -> None:
        """Trigger a stop condition."""
        self.is_active = False
        self.stop_reason = reason
        self.last_stop_time = datetime.now()
        self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)

        logger.warning(f"Trading stopped: {reason.value} - {message}")
        logger.info(f"Cooldown until: {self.cooldown_until}")

    def resume_trading(self) -> bool:
        """Manually resume trading."""
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds()
            logger.warning(
                f"Cannot resume - cooldown active: {remaining:.0f}s remaining"
            )
            return False

        self.is_active = True
        self.stop_reason = None
        logger.info("Trading resumed manually")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the auto-stop system."""
        return {
            "is_active": self.is_active,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "cooldown_until": (
                self.cooldown_until.isoformat() if self.cooldown_until else None
            ),
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "consecutive_losses": self.consecutive_losses,
            "total_trades": len(self.trade_history),
            "active_conditions": len(
                [c for c in self.stop_conditions.values() if c.enabled]
            ),
        }


def create_production_auto_stop() -> AdvancedAutoStop:
    """Create production-ready auto-stop system."""
    config = {
        "volatility_stop": {
            "enabled": True,
            "threshold": 0.03,  # 3% volatility (conservative)
            "window_size": 30,  # 30 minutes
            "cooldown_period": 900,  # 15 minutes
            "severity": "warning",
        },
        "drawdown_stop": {
            "enabled": True,
            "threshold": 0.05,  # 5% drawdown (conservative)
            "window_size": 720,  # 12 hours
            "cooldown_period": 1800,  # 30 minutes
            "severity": "critical",
        },
        "time_stop": {
            "enabled": True,
            "threshold": 21600,  # 6 hours (conservative)
            "cooldown_period": 3600,  # 1 hour
            "severity": "warning",
        },
        "performance_stop": {
            "enabled": True,
            "threshold": -0.02,  # -2% performance (conservative)
            "window_size": 120,  # 2 hours
            "cooldown_period": 900,  # 15 minutes
            "severity": "warning",
        },
        "market_condition_stop": {
            "enabled": True,
            "volatility_multiplier": 1.5,
            "trend_strength_threshold": 0.8,
            "cooldown_period": 1800,  # 30 minutes
            "severity": "warning",
        },
        "consecutive_losses_stop": {
            "enabled": True,
            "threshold": 3,  # 3 consecutive losses (conservative)
            "cooldown_period": 900,  # 15 minutes
            "severity": "critical",
        },
    }

    return AdvancedAutoStop(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    auto_stop = create_production_auto_stop()
    print("Advanced Auto-Stop system created")
    print(f"Status: {auto_stop.get_status()}")
