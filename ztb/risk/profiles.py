"""
Risk management profiles for trading.

Defines conservative, balanced, and aggressive risk profiles.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class RiskProfile(Enum):
    """Risk profile levels."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    # Position limits
    max_position_notional: float  # Maximum position size in JPY
    max_single_trade_pct: float  # Max % of capital per trade

    # Daily loss limits
    daily_loss_limit_pct: float  # Max daily loss as % of starting capital
    max_drawdown_pct: float  # Max drawdown before stopping

    # Trade frequency
    max_trades_per_hour: int  # Maximum trades per hour
    min_trade_interval_sec: int  # Minimum seconds between trades

    # Risk metrics
    max_volatility_pct: float  # Max portfolio volatility
    required_sharpe_ratio: float  # Minimum Sharpe ratio threshold

    # Stop loss settings
    stop_loss_pct: float  # Stop loss percentage
    take_profit_pct: float  # Take profit percentage


# Predefined risk profiles
RISK_PROFILES: Dict[RiskProfile, RiskLimits] = {
    RiskProfile.CONSERVATIVE: RiskLimits(
        max_position_notional=50000.0,
        max_single_trade_pct=0.02,  # 2% of capital
        daily_loss_limit_pct=0.02,  # 2% daily loss limit
        max_drawdown_pct=0.05,  # 5% max drawdown
        max_trades_per_hour=2,
        min_trade_interval_sec=1800,  # 30 minutes
        max_volatility_pct=0.10,  # 10% volatility
        required_sharpe_ratio=0.5,
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.05,  # 5% take profit
    ),
    RiskProfile.BALANCED: RiskLimits(
        max_position_notional=100000.0,
        max_single_trade_pct=0.05,  # 5% of capital
        daily_loss_limit_pct=0.05,  # 5% daily loss limit
        max_drawdown_pct=0.10,  # 10% max drawdown
        max_trades_per_hour=5,
        min_trade_interval_sec=600,  # 10 minutes
        max_volatility_pct=0.15,  # 15% volatility
        required_sharpe_ratio=1.0,
        stop_loss_pct=0.03,  # 3% stop loss
        take_profit_pct=0.08,  # 8% take profit
    ),
    RiskProfile.AGGRESSIVE: RiskLimits(
        max_position_notional=200000.0,
        max_single_trade_pct=0.10,  # 10% of capital
        daily_loss_limit_pct=0.10,  # 10% daily loss limit
        max_drawdown_pct=0.20,  # 20% max drawdown
        max_trades_per_hour=10,
        min_trade_interval_sec=300,  # 5 minutes
        max_volatility_pct=0.25,  # 25% volatility
        required_sharpe_ratio=1.5,
        stop_loss_pct=0.05,  # 5% stop loss
        take_profit_pct=0.15,  # 15% take profit
    ),
}


def get_risk_profile(profile_name: str) -> RiskLimits:
    """Get risk limits for a profile name."""
    try:
        profile = RiskProfile(profile_name.lower())
        return RISK_PROFILES[profile]
    except ValueError:
        raise ValueError(
            f"Unknown risk profile: {profile_name}. "
            f"Available: {[p.value for p in RiskProfile]}"
        )


def create_custom_risk_profile(**kwargs: Any) -> RiskLimits:
    """Create a custom risk profile from parameters."""
    defaults = {
        "max_position_notional": 100000.0,
        "max_single_trade_pct": 0.05,
        "daily_loss_limit_pct": 0.05,
        "max_drawdown_pct": 0.10,
        "max_trades_per_hour": 5,
        "min_trade_interval_sec": 600,
        "max_volatility_pct": 0.15,
        "required_sharpe_ratio": 1.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.08,
    }

    # Update defaults with provided values
    config = {**defaults, **kwargs}

    return RiskLimits(
        max_position_notional=float(config["max_position_notional"]),
        max_single_trade_pct=float(config["max_single_trade_pct"]),
        daily_loss_limit_pct=float(config["daily_loss_limit_pct"]),
        max_drawdown_pct=float(config["max_drawdown_pct"]),
        max_trades_per_hour=int(config["max_trades_per_hour"]),
        min_trade_interval_sec=int(config["min_trade_interval_sec"]),
        max_volatility_pct=float(config["max_volatility_pct"]),
        required_sharpe_ratio=float(config["required_sharpe_ratio"]),
        stop_loss_pct=float(config["stop_loss_pct"]),
        take_profit_pct=float(config["take_profit_pct"]),
    )
