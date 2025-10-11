"""
Trading environment constants.

This module defines all magic numbers and constants used throughout the trading environment.
Centralizing constants improves maintainability and reduces errors.
"""

# ============================================================================
# Action Space Constants
# ============================================================================

# Discrete action space (for PPO with Masked actions)
NUM_DISCRETE_ACTIONS = 3
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

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
CONTINUOUS_TO_DISCRETE_THRESHOLD = 0.33


# ============================================================================
# Transaction Cost Constants (in basis points, 0.0001 = 0.01%)
# ============================================================================

# Default transaction costs
DEFAULT_TRANSACTION_COST = 0.001  # 0.1% (10 basis points)
DEFAULT_BUY_FEE_RATE = 0.001     # 0.1%
DEFAULT_SELL_FEE_RATE = 0.001    # 0.1%

# Exchange-specific transaction costs
EXCHANGE_FEES = {
    "coincheck": {
        "buy": 0.0,      # 0% taker fee
        "sell": 0.0,     # 0% taker fee
    },
    "bitflyer": {
        "buy": 0.001,    # 0.1%
        "sell": 0.001,   # 0.1%
    },
    "binance": {
        "buy": 0.001,    # 0.1%
        "sell": 0.001,   # 0.1%
    },
}

# High-frequency trading
HFT_TRANSACTION_COST = 0.002  # 0.2% for aggressive strategies


# ============================================================================
# Environment Configuration Constants
# ============================================================================

# Default balance and position limits
DEFAULT_INITIAL_BALANCE = 200000    # JPY
DEFAULT_MAX_POSITION_SIZE = 0.01    # 1% of balance per trade

# Action history tracking
DEFAULT_MAX_ACTION_HISTORY = 256
MAX_ACTION_HISTORY_LARGE = 512

# Holding period constraints
DEFAULT_MIN_HOLDING_PERIOD = 0      # No restriction
RECOMMENDED_MIN_HOLDING_PERIOD = 3  # Prevent rapid flip-flopping

# Volatility and correlation thresholds
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_VOLATILITY_THRESHOLD = 0.001


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
DEFAULT_LAGRANGE_R_TARGET = 0.175    # 17.5% SELL actions
MIN_SELL_RATE = 0.15                 # Minimum 15% SELL actions

# Lagrange multiplier parameters
DEFAULT_LAGRANGE_TOLERANCE = 0.042625
DEFAULT_LAGRANGE_ETA = 0.062875
DEFAULT_LAGRANGE_ETA_MIN = 0.001
DEFAULT_LAGRANGE_ETA_MAX = 100.0
DEFAULT_LAGRANGE_ETA_LR = 0.001


# ============================================================================
# Curriculum Learning Constants
# ============================================================================

# Hold restriction levels
HOLD_RESTRICTION_NONE = "none"
HOLD_RESTRICTION_LIMITED_20 = "limited_20"  # Max 20% HOLD
HOLD_RESTRICTION_FORBIDDEN = "forbidden"    # No HOLD allowed

# Diversity thresholds (minimum percentage for each action)
MIN_DIVERSITY_THRESHOLD_STRICT = 0.4   # 40% each BUY/SELL
MIN_DIVERSITY_THRESHOLD_MODERATE = 0.3  # 30% each action
MIN_DIVERSITY_THRESHOLD_RELAXED = 0.2   # 20% active trading

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
REGIME_THRESHOLD = 0.001

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


def continuous_to_discrete_action(continuous_action: float) -> int:
    """
    Convert continuous action [-1, 1] to discrete action.
    
    Args:
        continuous_action: Continuous action value
            - < -threshold: SELL
            - in [-threshold, threshold]: HOLD
            - > threshold: BUY
            
    Returns:
        Discrete action (0=HOLD, 1=BUY, 2=SELL)
    """
    if continuous_action > CONTINUOUS_TO_DISCRETE_THRESHOLD:
        return ACTION_BUY
    elif continuous_action < -CONTINUOUS_TO_DISCRETE_THRESHOLD:
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
