"""
Configuration schemas using Pydantic for type safety and validation.

This module defines the configuration models for various components of the trading system.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    model_config = ConfigDict(extra="ignore")

    total_timesteps: int = Field(
        default=1000000, description="Total training timesteps"
    )
    n_envs: int = Field(default=4, description="Number of parallel environments")
    seed: Optional[int] = Field(default=None, description="Random seed")
    eval_interval: int = Field(
        default=10000, description="Evaluation interval in steps"
    )
    log_interval: int = Field(default=1000, description="Logging interval in steps")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    batch_size: int = Field(default=64, description="Batch size for training")
    n_epochs: int = Field(default=10, description="Number of epochs per update")


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint management."""

    model_config = ConfigDict(extra="ignore")

    async_save: bool = Field(
        default=True, description="Enable asynchronous checkpoint saving"
    )
    compress: str = Field(
        default="zstd", description="Compression algorithm (none, zstd, lz4)"
    )
    max_pending: int = Field(default=1, description="Maximum pending checkpoints")
    retention: int = Field(default=5, description="Number of checkpoints to retain")
    interval_steps: int = Field(
        default=10000, description="Checkpoint interval in steps"
    )
    light_mode: bool = Field(default=False, description="Use light checkpoint mode")


class StreamingConfig(BaseModel):
    """Configuration for data streaming."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=False, description="Enable streaming mode")
    batch_size: int = Field(default=64, description="Streaming batch size")
    buffer_policy: str = Field(
        default="drop_oldest", description="Buffer overflow policy"
    )
    prefetch_factor: int = Field(default=2, description="Prefetch factor for streaming")


class EvalConfig(BaseModel):
    """Configuration for evaluation parameters."""

    model_config = ConfigDict(extra="ignore")

    dsr_trials: int = Field(default=1000, description="Number of DSR trials")
    bootstrap_resamples: int = Field(
        default=1000, description="Number of bootstrap resamples"
    )
    bootstrap_block: Optional[int] = Field(
        default=None, description="Bootstrap block size"
    )
    bootstrap_overlap: Optional[int] = Field(
        default=None, description="Bootstrap overlap"
    )
    eval_freq: int = Field(default=50000, description="Evaluation frequency in steps")
    benchmark_strategies: List[str] = Field(
        default=["sma", "buy_hold"], description="Benchmark strategies"
    )


class VenuePrecisionConfig(BaseModel):
    """Configuration for venue-specific precision policies."""

    model_config = ConfigDict(extra="ignore")

    price_tick: float = Field(default=0.01, description="Minimum price increment")
    quantity_step: float = Field(
        default=0.0001, description="Minimum quantity increment"
    )
    min_quantity: Optional[float] = Field(
        default=None, description="Minimum order quantity"
    )
    max_quantity: Optional[float] = Field(
        default=None, description="Maximum order quantity"
    )
    min_price: Optional[float] = Field(default=None, description="Minimum order price")
    max_price: Optional[float] = Field(default=None, description="Maximum order price")


class RiskProfileConfig(BaseModel):
    """Configuration for risk management profiles."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Profile name")
    max_position_size: float = Field(
        default=0.1, description="Maximum position size as fraction of portfolio"
    )
    max_daily_loss: float = Field(
        default=0.05, description="Maximum daily loss as fraction of portfolio"
    )
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.04, description="Take profit percentage")
    max_open_positions: int = Field(
        default=5, description="Maximum number of open positions"
    )
    risk_per_trade: float = Field(
        default=0.01, description="Risk per trade as fraction of portfolio"
    )
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    cooldown_period: int = Field(
        default=60, description="Cooldown period in seconds after loss"
    )


class GlobalConfig(BaseModel):
    """Global configuration container."""

    model_config = ConfigDict(extra="ignore")

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    # Additional global settings
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    log_level: str = Field(default="INFO", description="Logging level")
    output_dir: str = Field(default="artifacts", description="Output directory")

    # Venue precision policies
    venue_precision: Dict[str, Dict[str, VenuePrecisionConfig]] = Field(
        default_factory=dict, description="Venue-specific precision policies"
    )

    # Risk profile presets
    risk_profiles: Dict[str, RiskProfileConfig] = Field(
        default_factory=lambda: {
            "conservative": RiskProfileConfig(
                name="conservative",
                max_position_size=0.05,
                max_daily_loss=0.02,
                stop_loss_pct=0.01,
                take_profit_pct=0.02,
                max_open_positions=3,
                risk_per_trade=0.005,
                max_leverage=1.0,
                cooldown_period=120,
            ),
            "moderate": RiskProfileConfig(
                name="moderate",
                max_position_size=0.1,
                max_daily_loss=0.05,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                max_open_positions=5,
                risk_per_trade=0.01,
                max_leverage=1.0,
                cooldown_period=60,
            ),
            "aggressive": RiskProfileConfig(
                name="aggressive",
                max_position_size=0.2,
                max_daily_loss=0.1,
                stop_loss_pct=0.05,
                take_profit_pct=0.1,
                max_open_positions=10,
                risk_per_trade=0.02,
                max_leverage=2.0,
                cooldown_period=30,
            ),
        },
        description="Risk profile presets for different trading strategies",
    )


# Convenience exports
__all__ = [
    "TrainingConfig",
    "CheckpointConfig",
    "StreamingConfig",
    "EvalConfig",
    "VenuePrecisionConfig",
    "RiskProfileConfig",
    "GlobalConfig",
]
