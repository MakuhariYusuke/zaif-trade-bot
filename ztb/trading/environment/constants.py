"""
Trading environment constants.

This module defines all magic numbers and constants used throughout the trading environment.
Centralizing constants improves maintainability and reduces errors.
"""

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    SAC_CONTINUOUS_THRESHOLD,
    SAC_CONTINUOUS_THRESHOLD_NEG,
)
from ztb.utils.exceptions.custom_exceptions import ConfigurationError

# ============================================================================
# Common Numeric Constants
# ============================================================================

# Standard fee rates and percentages
DEFAULT_FEE_RATE = 0.001  # 0.1% - Standard transaction fee
MINIMUM_UNIT_BUFFER = 0.001  # 0.1% - Minimum unit buffer for position sizing
MAXIMUM_FEE_RATE = 0.01  # 1% - Maximum allowable fee rate

# BTC and trading unit constants
BTC_MIN_UNIT = 0.0001  # Minimum BTC trading unit
EPSILON = (
    1e-5  # Small epsilon value for numerical stability (prevents division by zero)
)

# Position and balance constants
DEFAULT_INITIAL_BALANCE = 200000  # JPY - Default starting balance
DEFAULT_INITIAL_BALANCE_SMALL = 10000.0  # JPY - Small balance for testing
DEFAULT_TOTAL_CAPITAL = 100000.0  # JPY - Default total capital
DEFAULT_MAX_TRADE_SIZE_JPY = 100000.0  # JPY - Maximum trade size

# Action history sizes
DEFAULT_MAX_ACTION_HISTORY = 256  # Standard action history buffer
MAX_ACTION_HISTORY_LARGE = 512  # Large action history buffer for complex strategies

# Common batch sizes for training
# BATCH_SIZE_* constants imported from training.constants
from ztb.training.constants import (
    BATCH_SIZE_MEDIUM,
    BATCH_SIZE_SMALL,
    BATCH_SIZE_STANDARD,
)

# Common buffer/replay buffer sizes
BUFFER_SIZE_SMALL = 50000
BUFFER_SIZE_MEDIUM = 100000
BUFFER_SIZE_LARGE = 200000
BUFFER_SIZE_XLARGE = 500000

# Training and evaluation constants
DEFAULT_CHECKPOINT_INTERVAL = 10000  # Default checkpoint interval
DEFAULT_MAX_STEPS_PER_EPISODE = 10000  # Default max steps per episode
DEFAULT_ANALYSIS_SAMPLES = 10000  # Default number of analysis samples
DEFAULT_MIN_SAMPLES = 10000  # Default minimum samples for metrics
PPO_DEFAULT_N_STEPS = 2048  # Default n_steps for PPO training

# Volume and data generation constants
DEFAULT_VOLUME_RANGE_MIN = 1000  # Minimum volume range
DEFAULT_VOLUME_RANGE_MAX = 10000  # Maximum volume range

# Memory conversion constants
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3

# Statistical thresholds
DEFAULT_VOLATILITY_THRESHOLD = 0.001  # 0.1% volatility threshold
DEFAULT_CORRELATION_THRESHOLD = 0.95  # 95% correlation threshold
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.01  # 1% significance level
MINIMUM_STANDARD_DEVIATION = 0.01  # 1% minimum std dev

# Transaction costs and fees
DEFAULT_TRANSACTION_COST = 1e-5  # Default transaction cost for training
BASIS_POINTS = 10000  # Basis points conversion factor (1% = 100 basis points)

# Learning rates and optimization parameters
# DEFAULT_LEARNING_RATE imported from training.constants

DEFAULT_LAGRANGE_ETA_MIN = 0.001  # Minimum Lagrange multiplier
DEFAULT_LAGRANGE_ETA_LR = 0.001  # Lagrange learning rate

# Performance thresholds
GRADIENT_NORM_THRESHOLD = 1e-6
ADVANTAGE_THRESHOLD = 0.0
TIME_THRESHOLD_MS = 1.0  # 1ms
MEMORY_THRESHOLD_MB = 10.0  # 10MB

# Memory optimization
MEMORY_LEAK_THRESHOLD_PERCENT = 50.0
OBJECT_LEAK_THRESHOLD = 10000
PYTORCH_CUDA_ALLOC_MB = 512  # 512MB for CUDA allocation
PYTORCH_CUDA_ALLOC_MB = 512  # 512MB for CUDA allocation

# ============================================================================
# Action Space Constants
# ============================================================================

# Discrete action space (for PPO with Masked actions)
NUM_DISCRETE_ACTIONS = 3
# ACTION_HOLD = 0
# ACTION_BUY = 1
# ACTION_SELL = -1

# Action names for logging and display
ACTION_NAMES = ["HOLD", "BUY", "SELL"]

# Continuous action space (for SAC and other continuous algorithms)
# Action value range: [-1.0, 1.0]
# Mapping: negative values = SELL intensity, 0 = HOLD, positive = BUY intensity
CONTINUOUS_ACTION_MIN = -1.0
CONTINUOUS_ACTION_MAX = 1.0
CONTINUOUS_ACTION_DIM = 1  # Single continuous value

# Thresholds for converting continuous actions to discrete
# If continuous action is in [-threshold, threshold], it's HOLD
CONTINUOUS_TO_DISCRETE_THRESHOLD = SAC_CONTINUOUS_THRESHOLD
CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG = SAC_CONTINUOUS_THRESHOLD_NEG

# Holding period constraints
DEFAULT_MIN_HOLDING_PERIOD = 0  # No restriction
RECOMMENDED_MIN_HOLDING_PERIOD = 3  # Prevent rapid flip-flopping

# Volatility and correlation thresholds
DEFAULT_CORRELATION_THRESHOLD = DEFAULT_CORRELATION_THRESHOLD
DEFAULT_VOLATILITY_THRESHOLD = DEFAULT_VOLATILITY_THRESHOLD


# ============================================================================
# Reward Function Constants
# ============================================================================

# HOLD penalty weights
DEFAULT_HOLD_PENALTY_WEIGHT = 0.0
AGGRESSIVE_HOLD_PENALTY_WEIGHT = 0.1

# Consecutive action penalties
DEFAULT_CONSECUTIVE_HOLD_PENALTY = 0.0
MODERATE_CONSECUTIVE_HOLD_PENALTY = 0.05
AGGRESSIVE_CONSECUTIVE_HOLD_PENALTY = 0.01

# Trading frequency bonuses
DEFAULT_TRADING_FREQUENCY_BONUS = 0.0
MODERATE_TRADING_FREQUENCY_BONUS = 0.3

# Profit multipliers
DEFAULT_PROFIT_REWARD_MULTIPLIER = 1.0
AGGRESSIVE_PROFIT_REWARD_MULTIPLIER = 10.0

# Action diversity bonuses
DEFAULT_ACTION_DIVERSITY_BONUS = 0.0
MODERATE_ACTION_DIVERSITY_BONUS = 0.1

# Balance penalty constants
DEFAULT_BALANCE_PENALTY_SCALE = 1000.0  # Strong penalty for action imbalance (increased from 10.0)
DEFAULT_ACTION_BALANCE_TARGET = 0.333  # Target 33.3% for each action
DEFAULT_REDUNDANT_TRADE_PENALTY = 10.0  # Penalty for redundant trades at max position (increased from 5.0)

# Successful trade bonuses
DEFAULT_SUCCESSFUL_TRADE_BONUS = 0.0
AGGRESSIVE_SUCCESSFUL_TRADE_BONUS = 5.0

# Consecutive hold thresholds
DEFAULT_CONSECUTIVE_HOLD_THRESHOLD = 0
MODERATE_CONSECUTIVE_HOLD_THRESHOLD = 5

# Range market hold tolerance
DEFAULT_RANGE_MARKET_HOLD_TOLERANCE = 0.0
MODERATE_RANGE_MARKET_HOLD_TOLERANCE = 0.5


# ============================================================================
# Lagrange Constraint Constants (SELL Mitigation)
# ============================================================================

# Target SELL rate for Lagrange constraint
DEFAULT_LAGRANGE_R_TARGET = 0.175  # 17.5% SELL actions
MIN_SELL_RATE = 0.15  # Minimum 15% SELL actions

# Lagrange multiplier parameters
DEFAULT_LAGRANGE_TOLERANCE = 0.042625
DEFAULT_LAGRANGE_ETA = 0.062875
DEFAULT_LAGRANGE_ETA_MIN = DEFAULT_LAGRANGE_ETA_MIN
DEFAULT_LAGRANGE_ETA_MAX = 100.0
DEFAULT_LAGRANGE_ETA_LR = DEFAULT_LAGRANGE_ETA_LR


# ============================================================================
# Curriculum Learning Constants
# ============================================================================

# Hold restriction levels
HOLD_RESTRICTION_NONE = "none"
HOLD_RESTRICTION_LIMITED_20 = "limited_20"  # Max 20% HOLD
HOLD_RESTRICTION_FORBIDDEN = "forbidden"  # No HOLD allowed

# Diversity thresholds (minimum percentage for each action)
MIN_DIVERSITY_THRESHOLD_STRICT = 0.4  # 40% each BUY/SELL
MIN_DIVERSITY_THRESHOLD_MODERATE = 0.3  # 30% each action
MIN_DIVERSITY_THRESHOLD_RELAXED = 0.2  # 20% active trading

# Consecutive action penalties for curriculum
CONSECUTIVE_ACTION_PENALTY_STRONG = 0.01
CONSECUTIVE_ACTION_PENALTY_MODERATE = 0.005
CONSECUTIVE_ACTION_PENALTY_LIGHT = 0.002


# ============================================================================
# Model Performance Thresholds
# ============================================================================

# Action weighting parameters
ACTION_WEIGHTING_BETA = 3.0
ACTION_WEIGHTING_EMA_ALPHA = 0.1

# Gradient probe guards
GRAD_NORM_THRESHOLD = 1e-6
ADVANTAGE_THRESHOLD = 0.0

# Stratified sampling
MIN_SAMPLES_PER_ACTION = 1

# Regime detection
REGIME_THRESHOLD = DEFAULT_VOLATILITY_THRESHOLD

# Evaluation gates
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 0.15


# ============================================================================
# Memory Optimization Constants
# ============================================================================

# Data rows limits for memory optimization
DEFAULT_DATA_ROWS_LIMIT = None  # No limit
MEMORY_OPTIMIZED_DATA_ROWS = 10000

# Feature limits
DEFAULT_MAX_FEATURES = None  # No limit
MEMORY_OPTIMIZED_MAX_FEATURES = 50

# Feature quality thresholds
FEATURE_ZERO_RATE_THRESHOLD = 0.999  # 99.9% max zero rate for feature filtering (relaxed from 99%)
FEATURE_MIN_COUNT = 30  # Minimum features after quality filtering (reduced from 50)
FEATURE_VARIANCE_THRESHOLD = 1e-6  # Minimum variance threshold (relaxed from 1e-8)
FEATURE_NAN_RATE_THRESHOLD = 0.10  # 10% max NaN rate
FEATURE_OUTLIER_RATE_THRESHOLD = 0.30  # 30% max outlier rate
FEATURE_CORRELATION_THRESHOLD = 0.95  # 95% max correlation


# ============================================================================
# Statistical Thresholds
# ============================================================================

# Z-score thresholds for outlier detection
Z_SCORE_THRESHOLD_STRICT = 3.0
Z_SCORE_THRESHOLD_RELAXED = 5.0

# Calibration
DEFAULT_NUM_CALIBRATION_BINS = 10

# Performance profiling
DEFAULT_TIME_THRESHOLD_MS = 1.0
DEFAULT_MEMORY_THRESHOLD_MB = 10.0

# Memory leak detection
MEMORY_LEAK_THRESHOLD_PERCENT = 50.0
OBJECT_LEAK_THRESHOLD = 10000


# ============================================================================
# Utility Functions
# ============================================================================


def get_action_name(action: int) -> str:
    """
    Get human-readable name for action.

    Args:
        action: Action index (0=HOLD, 1=BUY, 2=SELL)

    Returns:
        Action name string
    """
    if 0 <= action < NUM_DISCRETE_ACTIONS:
        return ACTION_NAMES[action]
    return f"UNKNOWN_ACTION_{action}"


def continuous_to_discrete_action(
    continuous_action: float,
    threshold: float = CONTINUOUS_TO_DISCRETE_THRESHOLD,
    negative_threshold: float = CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG,
) -> int:
    """
    Convert continuous action [-1, 1] to discrete action.

    Args:
        continuous_action: Continuous action value
            - < -threshold: SELL
            - in [-threshold, threshold]: HOLD
            - > threshold: BUY
        threshold: Threshold for action conversion (default: CONTINUOUS_TO_DISCRETE_THRESHOLD)
            - Lower threshold = more BUY/SELL actions, less HOLD
            - Higher threshold = more HOLD actions, less BUY/SELL

    Returns:
        Discrete action (0=HOLD, 1=BUY, -1=SELL)
    """
    if continuous_action > threshold:
        return ACTION_BUY
    elif continuous_action < negative_threshold:
        return ACTION_SELL
    else:
        return ACTION_HOLD


def discrete_to_continuous_action(discrete_action: int) -> float:
    """
    Convert discrete action to continuous action.

    Args:
        discrete_action: Discrete action (0=HOLD, 1=BUY, 2=SELL)

    Returns:
        Continuous action value in [-1, 1]
    """
    if discrete_action == ACTION_BUY:
        return 1.0
    elif discrete_action == ACTION_SELL:
        return -1.0
    else:  # HOLD
        return 0.0


# ============================================================================
# Environment-Specific Constants
# ============================================================================

import os
from typing import Any, Dict

# Environment detection
ENVIRONMENT = os.getenv("TRADING_ENV", "dev").lower()

# Environment-specific configurations
ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dev": {
        "fee_rate": 0.001,  # 0.1% for development testing
        "initial_balance": 10000.0,  # Small balance for quick testing
        "max_trade_size": 1000.0,  # Small trades for dev
        "buffer_size": BUFFER_SIZE_SMALL,
        "batch_size": BATCH_SIZE_SMALL,
        "max_action_history": DEFAULT_MAX_ACTION_HISTORY,
        "enable_logging": True,
        "enable_memory_profiling": True,
    },
    "staging": {
        "fee_rate": 0.001,  # 0.1% for staging
        "initial_balance": 50000.0,  # Medium balance for staging
        "max_trade_size": 50000.0,  # Medium trades for staging
        "buffer_size": BUFFER_SIZE_MEDIUM,
        "batch_size": BATCH_SIZE_MEDIUM,
        "max_action_history": DEFAULT_MAX_ACTION_HISTORY,
        "enable_logging": True,
        "enable_memory_profiling": False,
    },
    "prod": {
        "fee_rate": 0.001,  # 0.1% for production (actual exchange fees)
        "initial_balance": 200000.0,  # Full balance for production
        "max_trade_size": 100000.0,  # Full trade sizes for production
        "buffer_size": BUFFER_SIZE_MEDIUM,
        "batch_size": BATCH_SIZE_STANDARD,
        "max_action_history": MAX_ACTION_HISTORY_LARGE,
        "enable_logging": False,  # Minimal logging in production
        "enable_memory_profiling": False,
    },
}

# Current environment config
CURRENT_ENV_CONFIG = ENV_CONFIGS.get(ENVIRONMENT, ENV_CONFIGS["dev"])

# Environment-specific constants (dynamically set based on environment)
ENV_FEE_RATE = CURRENT_ENV_CONFIG["fee_rate"]
ENV_INITIAL_BALANCE = CURRENT_ENV_CONFIG["initial_balance"]
ENV_MAX_TRADE_SIZE = CURRENT_ENV_CONFIG["max_trade_size"]
ENV_BUFFER_SIZE = CURRENT_ENV_CONFIG["buffer_size"]
ENV_BATCH_SIZE = CURRENT_ENV_CONFIG["batch_size"]
ENV_MAX_ACTION_HISTORY = CURRENT_ENV_CONFIG["max_action_history"]
ENV_ENABLE_LOGGING = CURRENT_ENV_CONFIG["enable_logging"]
ENV_ENABLE_MEMORY_PROFILING = CURRENT_ENV_CONFIG["enable_memory_profiling"]


# Validation functions for environment configs
def validate_environment_config() -> None:
    """
    Validate current environment configuration.

    Raises:
        ConfigurationError: If configuration is invalid
    """
    if ENVIRONMENT not in ENV_CONFIGS:
        raise ConfigurationError(
            f"Invalid environment '{ENVIRONMENT}'. Must be one of: {list(ENV_CONFIGS.keys())}"
        )

    config = CURRENT_ENV_CONFIG

    if config["fee_rate"] <= 0 or config["fee_rate"] > 0.1:
        raise ConfigurationError(
            f"Invalid fee_rate: {config['fee_rate']}. Must be between 0 and 0.1"
        )

    if config["initial_balance"] <= 0:
        raise ConfigurationError(
            f"Invalid initial_balance: {config['initial_balance']}. Must be positive"
        )

    if config["max_trade_size"] <= 0:
        raise ConfigurationError(
            f"Invalid max_trade_size: {config['max_trade_size']}. Must be positive"
        )

    if config["buffer_size"] <= 0:
        raise ConfigurationError(
            f"Invalid buffer_size: {config['buffer_size']}. Must be positive"
        )

    if config["batch_size"] <= 0:
        raise ConfigurationError(
            f"Invalid batch_size: {config['batch_size']}. Must be positive"
        )


# Validate on import
validate_environment_config()

# ============================================================================
# Live Trading Constants
# ============================================================================

# Price history buffer size for live trading
DEFAULT_PRICE_HISTORY_SIZE = 100

# Cleanup intervals for live trading
DEFAULT_CLEANUP_INTERVAL = 100  # iterations

# ============================================================================
# Heavy Trading Environment Constants
# ============================================================================

# Random start buffer parameters
RANDOM_START_MIN_BUFFER = 10  # Minimum buffer size for random start
RANDOM_START_MAX_BUFFER = 100  # Maximum buffer size for random start
RANDOM_START_BUFFER_RATIO = 0.1  # Ratio of data length to use as buffer

# Position calculation epsilon to prevent division by zero
POSITION_EPSILON = 1e-8

# Initial action counts for episode tracking
ACTION_COUNTS_INITIAL = [0, 0, 0]  # [HOLD, BUY, SELL] counts
