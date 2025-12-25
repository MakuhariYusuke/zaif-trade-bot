"""
Training Constants - Constants specific to training operations.

This module defines constants used throughout the training system,
organized by functional area for better maintainability.
"""

# ============================================================================
# Training Algorithm Constants
# ============================================================================

# Default learning rates for different algorithms
DEFAULT_LEARNING_RATE_PPO = 3e-4
DEFAULT_LEARNING_RATE_SAC = 3e-4
DEFAULT_LEARNING_RATE_AGGRESSIVE = (
    5e-4  # Higher learning rate for aggressive exploration
)
DEFAULT_LEARNING_RATE_DQN = 1e-3
DEFAULT_LEARNING_RATE = 0.001  # General default for environments

# Training batch sizes
DEFAULT_BATCH_SIZE_PPO = 64
DEFAULT_BATCH_SIZE_SAC = 256
DEFAULT_BATCH_SIZE_DQN = 32

# Buffer sizes for replay buffers
DEFAULT_BUFFER_SIZE_SAC = 1000000
DEFAULT_BUFFER_SIZE_MEDIUM = 100000  # Medium buffer size for standard training
DEFAULT_BUFFER_SIZE_AGGRESSIVE = 10000  # Smaller buffer for aggressive exploration

# General batch sizes for environments
BATCH_SIZE_SMALL = 32
BATCH_SIZE_MEDIUM = 64
BATCH_SIZE_STANDARD = 128
BATCH_SIZE_LARGE = 256
BATCH_SIZE_XLARGE = 512

# Training timesteps
DEFAULT_TOTAL_TIMESTEPS_PPO = 100000
DEFAULT_TOTAL_TIMESTEPS_SAC = 100000
DEFAULT_TOTAL_TIMESTEPS_DQN = 50000

# Learning starts
DEFAULT_LEARNING_STARTS_SAC = 1000
DEFAULT_LEARNING_STARTS_MINIMAL = 100

# Discount factor
DEFAULT_GAMMA = 0.99

# Target network update rate
DEFAULT_TAU = 0.005

# PPO clip range
DEFAULT_CLIP_RANGE = 0.2

# Entropy coefficient for SAC
DEFAULT_ENT_COEF_SAC = 0.1
DEFAULT_ENT_COEF_AUTO = "auto_1.0"

# Entropy coefficient for PPO
DEFAULT_ENT_COEF_PPO = 0.01

# PPO training parameters
DEFAULT_N_EPOCHS_PPO = 10
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_VF_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5

# SAC training parameters
DEFAULT_TRAIN_FREQ = 1
DEFAULT_GRADIENT_STEPS = 1

# Target update interval
DEFAULT_TARGET_UPDATE_INTERVAL = 1

# Verbosity level
DEFAULT_VERBOSE = 1

# Checkpoint/check frequency
DEFAULT_CHECK_FREQ = 1000

# Maximum training steps for configurable training
DEFAULT_MAX_TRAIN_STEPS = 2000

# Buffer steps to leave in data
DEFAULT_BUFFER_STEPS = 10

# Memory limits
DEFAULT_MAX_MEMORY_GB = 4.0

# Data processing
DEFAULT_CHUNK_SIZE = 1000

# ============================================================================
# Training Environment Constants
# ============================================================================

# Environment update frequencies
ENV_RESET_FREQUENCY = 1000  # Reset environment every N steps
ENV_EVAL_FREQUENCY = 5000  # Evaluate during training every N steps

# Reward scaling
DEFAULT_REWARD_SCALING = 1.0
REWARD_CLIP_VALUE = 10.0

# Observation normalization
OBSERVATION_NORMALIZATION_EPS = 1e-8

# ============================================================================
# Training Monitoring Constants
# ============================================================================

# Logging intervals
LOG_INTERVAL_STEPS = 1000
LOG_INTERVAL_SECONDS = 60

# Memory monitoring
MEMORY_LOG_INTERVAL = 2000
MEMORY_WARNING_THRESHOLD = 0.8  # 80% memory usage

# Performance profiling
PROFILE_INTERVAL = 5000
PROFILE_DURATION = 10  # seconds

# ============================================================================
# Training Validation Constants
# ============================================================================

# Validation parameters
VALIDATION_EPISODES = 10
VALIDATION_MAX_STEPS = 1000

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# Model saving
SAVE_INTERVAL = 10000
KEEP_CHECKPOINTS = 5

# ============================================================================
# Distributed Training Constants
# ============================================================================

# Distributed training settings
DEFAULT_WORLD_SIZE = 1
DEFAULT_MASTER_PORT = 12355
DEFAULT_BACKEND = "gloo"  # or "nccl"

# Gradient accumulation
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRADIENT_ACCUMULATION_STEPS = 8

# ============================================================================
# Training Error Handling Constants
# ============================================================================

# Retry configurations
MAX_TRAINING_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Timeout settings
TRAINING_TIMEOUT_HOURS = 24
EVALUATION_TIMEOUT_MINUTES = 30

# Error thresholds
MAX_CONSECUTIVE_ERRORS = 5
ERROR_RATE_THRESHOLD = 0.1  # 10% error rate

# ============================================================================
# Training Optimization Constants
# ============================================================================

# Learning rate scheduling
LR_DECAY_FACTOR = 0.5
LR_DECAY_PATIENCE = 5
MIN_LEARNING_RATE = 1e-6

# Gradient clipping
GRADIENT_CLIP_VALUE = 0.5
GRADIENT_CLIP_ALGORITHM = "norm"

# Regularization
L2_REGULARIZATION = 1e-4
DROPOUT_RATE = 0.1

# ============================================================================
# Training Metrics Constants
# ============================================================================

# Performance metrics
TARGET_SHARPE_RATIO = 1.0
TARGET_WIN_RATE = 0.55
TARGET_PROFIT_FACTOR = 1.2

# Risk metrics
MAX_DRAWDOWN_LIMIT = 0.2  # 20% max drawdown
VAR_CONFIDENCE_LEVEL = 0.95  # 95% VaR

# Convergence criteria
CONVERGENCE_THRESHOLD = 1e-5
CONVERGENCE_WINDOW = 100

# ============================================================================
# Training UI Constants
# ============================================================================

# Progress display
PROGRESS_BAR_WIDTH = 80
PROGRESS_UPDATE_INTERVAL = 1  # seconds

# Display formatting
FLOAT_PRECISION = 4
PERCENTAGE_PRECISION = 2

# Color schemes
COLOR_SUCCESS = "green"
COLOR_WARNING = "yellow"
COLOR_ERROR = "red"
COLOR_INFO = "blue"

# ============================================================================
# SAC Suite Constants
# ============================================================================

# Print formatting
SAC_PRINT_SEPARATOR_WIDTH = 60

# Default analysis parameters
SAC_DEFAULT_SAMPLES = 1000
SAC_DEFAULT_EPISODES = 100

# Exit codes
SAC_ERROR_EXIT_CODE = 130

# ============================================================================
# Training Component Constants
# ============================================================================

# TrainingStateManager constants
DEFAULT_REWARD_WINDOW_SIZE = 100
DEFAULT_LOSS_WINDOW_SIZE = 50
MAX_TRAINING_HISTORY_SIZE = 10000

# TrainingValidationManager constants
MIN_REWARD_THRESHOLD = -1000.0
MAX_REWARD_THRESHOLD = 1000.0
MIN_LOSS_THRESHOLD = 0.0
MAX_LOSS_THRESHOLD = 1000.0
NAN_TOLERANCE_FRACTION = 0.01  # 1% NaN tolerance

# TrainingRiskManager constants
EARLY_STOPPING_PATIENCE_DEFAULT = 20
OVERFITTING_THRESHOLD_DEFAULT = 0.1  # 10% performance drop
VALIDATION_WINDOW_DEFAULT = 50
LOSS_EXPLOSION_THRESHOLD = 10.0
REWARD_EXPLOSION_THRESHOLD = 1000.0
CONSECUTIVE_NAN_LOSS_THRESHOLD = 5

# ============================================================================
# Training Stability Constants
# ============================================================================

# Training stability thresholds
MAX_REWARD_VARIANCE_RATIO = 0.5  # Max std/mean ratio for rewards
MIN_TRAINING_HISTORY_FOR_TREND = 10
MIN_TRAINING_HISTORY_FOR_OVERFITTING = 100
MAX_TRAINING_TIME_HOURS = 24

# Performance monitoring
PERFORMANCE_CHECK_INTERVAL = 1000  # Check performance every N steps
CRITICAL_PERFORMANCE_DROP = 0.5  # 50% drop triggers warning
