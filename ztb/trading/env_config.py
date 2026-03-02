"""
Common environment configurations and utilities.

This module provides standardized environment configurations to reduce duplication
across training scripts and improve consistency.
"""

from typing import Any, Optional, TypedDict, cast

class TradingEnvConfig(TypedDict, total=False):
    """Type definition for trading environment configuration."""

    reward_scaling: float
    transaction_cost: float
    position_penalty_scale: float
    inventory_penalty_scale: float
    trade_frequency_penalty: float
    max_position_size: float
    fee_model: str
    fee_rate: float
    features: list[str]  # Optional field
    atr_period: int

# Default trading environment configuration
DEFAULT_TRADING_ENV_CONFIG: TradingEnvConfig = {
    "reward_scaling": 6.0,  # Optimized value from hyperparameter search
    "transaction_cost": 0.001,  # 0.1% transaction cost
    "position_penalty_scale": 0.01,
    "inventory_penalty_scale": 0.001,
    "trade_frequency_penalty": 0.0001,
    "max_position_size": 1.0,
    "fee_model": "percentage",
    "fee_rate": 0.001,
    "atr_period": 14,
}

def get_trading_env_config(
    overrides: dict[str, Any] | None = None,
) -> TradingEnvConfig:
    """Get trading environment configuration with optional overrides."""
    config: dict[str, Any] = dict(DEFAULT_TRADING_ENV_CONFIG)
    if overrides:
        config.update(overrides)
    return cast(TradingEnvConfig, config)

# Legacy constants for backward compatibility
DEFAULT_REWARD_SCALING = DEFAULT_TRADING_ENV_CONFIG["reward_scaling"]
DEFAULT_TRANSACTION_COST = DEFAULT_TRADING_ENV_CONFIG["transaction_cost"]
