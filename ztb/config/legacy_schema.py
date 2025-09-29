"""
Configuration schemas for ZTB trading system.

Defines default configurations for training and risk profiles.
"""

from dataclasses import dataclass, field


@dataclass
class RiskProfileConfig:
    """Risk profile configuration."""

    name: str = "aggressive"
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_daily_loss_pct: float = 0.20
    circuit_breaker_threshold: float = 0.15


@dataclass
class TrainingConfig:
    """Training configuration with BTC/JPY defaults."""

    symbol: str = "BTC_JPY"
    venue: str = "coincheck"
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    risk_profile: RiskProfileConfig = field(
        default_factory=lambda: RiskProfileConfig(name="aggressive")
    )


# Default configurations
DEFAULT_RISK_PROFILE = RiskProfileConfig(name="aggressive")
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    symbol="BTC_JPY", venue="coincheck", risk_profile=DEFAULT_RISK_PROFILE
)
