"""
Common PPO training configurations and utilities.

This module provides standardized PPO configurations to reduce duplication
across training scripts and improve consistency.
"""

from typing import Dict, Any, List, Optional, TypedDict, cast

# Common constants used across training modules
DEFAULT_REWARD_SCALING = 6.0     # Optimized value from hyperparameter search
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
DEFAULT_INITIAL_PORTFOLIO_VALUE = 1_000_000.0 # in JPY
DEFAULT_TRAINING_STEPS = 100_000

# === Long-Run Staging Configuration ===
# Staging design with flexible boundaries that adapt to total training steps
# Default ratios optimized for 1M training, but work for any total_timesteps

# Stage boundary ratios (as fraction of total_timesteps)
STAGE_WARMUP_RATIO = 0.05        # 0-5%: Warmup (weights=1.0, λ=0)
STAGE_TRANSITION_RATIO = 0.20    # 5-20%: Cosine warmup for weights/λ
STAGE_MAIN_RATIO = 0.80          # 20-80%: Main training (standard settings)
STAGE_FINAL_RATIO = 1.00         # 80-100%: Cosine annealing LR, early stop with 3 conditions

# Helper functions to calculate stage boundaries based on total steps
def get_stage_boundaries(total_timesteps: int) -> Dict[str, int]:
    """
    Calculate stage boundaries based on total training steps.
    
    Args:
        total_timesteps: Total number of training steps
        
    Returns:
        Dict with stage boundaries (warmup_end, transition_end, main_end, final_end)
        
    Example:
        For 1M steps:
        - warmup_end: 50,000 (5%)
        - transition_end: 200,000 (20%)
        - main_end: 800,000 (80%)
        - final_end: 1,000,000 (100%)
    """
    return {
        'warmup_end': int(total_timesteps * STAGE_WARMUP_RATIO),
        'transition_end': int(total_timesteps * STAGE_TRANSITION_RATIO),
        'main_end': int(total_timesteps * STAGE_MAIN_RATIO),
        'final_end': int(total_timesteps * STAGE_FINAL_RATIO),
    }

# Checkpoint and evaluation (adaptive)
CHECKPOINT_INTERVAL_RATIO = 0.025  # Save checkpoint every 2.5% of total steps (1M → 25k)
ROLLING_OOS_STEPS = 500            # Paper trade 500 steps for rolling OOS eval (extended from 300)

def get_checkpoint_interval(total_timesteps: int) -> int:
    """
    Calculate checkpoint interval based on total training steps.
    
    Args:
        total_timesteps: Total number of training steps
        
    Returns:
        Checkpoint interval (typically 2.5% of total steps)
        
    Example:
        - 1M steps → 25,000
        - 500k steps → 12,500
        - 2M steps → 50,000
    """
    return int(total_timesteps * CHECKPOINT_INTERVAL_RATIO)

# Monitoring thresholds (early stop conditions)
MIN_LEGAL_SELL_RATE = 0.12              # legal_sell_rate < 0.12 for patience period → stop
SELL_RATE_PATIENCE_RATIO = 0.005        # Patience as ratio of total steps (1M → 5k)

GRAD_NORM_SELL_MIN = 1e-6               # grad_norm(SELL) ≈ 0 → stop (gradient collapse)
SHARPE_PROXY_THRESHOLD = 0.0            # Sharpe_proxy ≤ 0 for patience evals → branch stop
SHARPE_PATIENCE_EVALS = 2               # Patience for low Sharpe (in evaluation counts)

def get_sell_rate_patience(total_timesteps: int) -> int:
    """
    Calculate patience steps for low sell rate based on total training steps.
    
    Args:
        total_timesteps: Total number of training steps
        
    Returns:
        Patience steps (typically 0.5% of total steps)
        
    Example:
        - 1M steps → 5,000
        - 500k steps → 2,500
        - 2M steps → 10,000
    """
    return int(total_timesteps * SELL_RATE_PATIENCE_RATIO)

# KL divergence monitoring
KL_VIOLATION_THRESHOLD = 0.5     # KL > 0.5 → potential policy collapse
KL_CRITICAL_THRESHOLD = 1.0      # KL > 1.0 → critical, emergency entropy boost

# Entropy target (H* = 0.7 * log(3) ≈ 0.769)
TARGET_ENTROPY_RATIO = 0.7       # Target entropy as ratio of max entropy
MAX_ENTROPY_3_ACTIONS = 1.0986   # log(3) for 3 actions (HOLD/BUY/SELL)

# Environment configuration constants
DEFAULT_RISK_FREE_RATE = 0.0    # Risk-free rate for Sharpe ratio calculation
DEFAULT_STOP_LOSS_THRESHOLD = 0.05  # 5% stop-loss threshold
DEFAULT_MAX_CONSECUTIVE_TRADES = 5  # Maximum number of consecutive trades
DEFAULT_MIN_HOLDING_PERIOD = 3  # Minimum holding period between trades

# Reward configuration constants
DEFAULT_REWARD_POSITION_SOFT_CAP = 0.8
DEFAULT_REWARD_POSITION_PENALTY_SCALE = 0.5
DEFAULT_REWARD_POSITION_PENALTY_EXPONENT = 4.0
DEFAULT_REWARD_INVENTORY_WINDOW = 128
DEFAULT_REWARD_INVENTORY_PENALTY_SCALE = 0.1
DEFAULT_REWARD_TRADE_FREQUENCY_PENALTY = 0.2
DEFAULT_REWARD_TRADE_FREQUENCY_HALFLIFE = 8.0
DEFAULT_REWARD_TRADE_COOLDOWN_STEPS = 2
DEFAULT_REWARD_TRADE_COOLDOWN_PENALTY = 0.2
DEFAULT_REWARD_MAX_CONSECUTIVE_TRADES = 5
DEFAULT_REWARD_CONSECUTIVE_TRADE_PENALTY = 0.1
DEFAULT_REWARD_VOLATILITY_WINDOW = 32
DEFAULT_REWARD_VOLATILITY_PENALTY_SCALE = 0.05
DEFAULT_REWARD_SHARPE_BONUS_SCALE = 0.02
DEFAULT_REWARD_CLIP_VALUE = 2.0


class PPOConfig(TypedDict, total=False):
    """Type definition for PPO configuration."""
    # Core PPO parameters
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: Optional[float]
    normalize_advantage: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    use_sde: bool
    sde_sample_freq: int
    target_kl: Optional[float]
    verbose: int

    # Trading-specific parameters
    reward_scaling: float
    transaction_cost: float
    position_penalty_scale: float
    inventory_penalty_scale: float
    trade_frequency_penalty: float
    total_timesteps: int

    # Environment parameters
    max_position_size: float
    fee_model: str
    fee_rate: float
    features: List[str]


# Default PPO configuration optimized for trading environments
DEFAULT_PPO_CONFIG: PPOConfig = {
    # Core PPO parameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "verbose": 1,

    # Trading-specific parameters
    "reward_scaling": DEFAULT_REWARD_SCALING,  # Optimized value from hyperparameter search
    "transaction_cost": 0.001,
    "position_penalty_scale": 0.01,
    "inventory_penalty_scale": 0.001,
    "trade_frequency_penalty": 0.0001,
    "total_timesteps": DEFAULT_TOTAL_TIMESTEPS,

    # Environment parameters
    "max_position_size": 1.0,
    "fee_model": "percentage",
    "fee_rate": 0.001,
    "features": [
        "close", "volume", "returns", "sma_20", "sma_50", "rsi_14",
        "macd", "bb_upper", "bb_lower", "atr_14", "stoch_k", "stoch_d"
    ],
}


def get_ppo_config(overrides: Optional[Dict[str, Any]] = None) -> PPOConfig:
    """
    Get PPO configuration with optional parameter overrides.

    Creates a PPO configuration dictionary starting from DEFAULT_PPO_CONFIG
    and applying any provided overrides. This allows for flexible configuration
    while maintaining sensible defaults.

    Args:
        overrides: Optional dictionary of configuration parameters to override.
                  Keys should match PPOConfig field names.

    Returns:
        PPOConfig: Complete PPO configuration dictionary with applied overrides.

    Example:
        >>> config = get_ppo_config({"learning_rate": 1e-4, "batch_size": 128})
        >>> print(config["learning_rate"])  # 0.0001
        >>> print(config["batch_size"])     # 128
    """
    config: Dict[str, Any] = dict(DEFAULT_PPO_CONFIG)
    if overrides:
        config.update(overrides)
    return cast(PPOConfig, config)


def get_conservative_ppo_config() -> PPOConfig:
    """
    Get conservative PPO configuration for stable training.

    Returns a PPO configuration optimized for stable, reliable training
    with lower learning rates and more conservative hyperparameters.
    Suitable for production training or when stability is prioritized
    over training speed.

    Returns:
        PPOConfig: Conservative PPO configuration with:
            - Lower learning rate (1e-4)
            - Smaller clip range (0.1)
            - Moderate entropy coefficient (0.01)
            - Lower max gradient norm (0.3)
    """
    return get_ppo_config({
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "max_grad_norm": 0.3,
    })


def get_aggressive_ppo_config() -> PPOConfig:
    """
    Get aggressive PPO configuration for faster learning.

    Returns a PPO configuration optimized for faster learning and exploration
    with higher learning rates and more aggressive hyperparameters.
    Suitable for initial experimentation or when training speed is prioritized
    over stability.

    Returns:
        PPOConfig: Aggressive PPO configuration with:
            - Higher learning rate (1e-3)
            - Larger clip range (0.3)
            - Higher entropy coefficient (0.1)
            - Higher max gradient norm (1.0)
    """
    return get_ppo_config({
        "learning_rate": 1e-3,
        "clip_range": 0.3,
        "ent_coef": 0.1,
        "max_grad_norm": 1.0,
    })