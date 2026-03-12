"""
Optimization Constants - Constants specific to reward function optimization operations.

This module defines constants used throughout the optimization system,
organized by functional area for better maintainability.
"""

# ============================================================================
# Optimization Algorithm Constants
# ============================================================================

# Optuna settings
DEFAULT_OPTIMIZATION_TRIALS = 100
MAX_OPTIMIZATION_TRIALS = 1000
MIN_OPTIMIZATION_TRIALS = 10

# Study configuration
STUDY_NAME_PREFIX = "reward_optimization_v"
DEFAULT_STUDY_DIRECTION = "maximize"
STUDY_STORAGE_TYPE = "sqlite"

# Sampler settings
DEFAULT_SAMPLER = "TPESampler"
SAMPLER_SEED = 42
SAMPLER_N_STARTUP_TRIALS = 10

# Pruner settings
DEFAULT_PRUNER = "MedianPruner"
PRUNER_N_STARTUP_TRIALS = 5
PRUNER_N_WARMUP_STEPS = 10

# ============================================================================
# Optimization Evaluation Constants
# ============================================================================

# Evaluation settings
DEFAULT_EVALUATION_EPISODES = 50
MAX_EVALUATION_EPISODES = 200
MIN_EVALUATION_EPISODES = 10

# Episode length
DEFAULT_MAX_EPISODE_LENGTH = 1000
MIN_EPISODE_LENGTH = 100
MAX_EPISODE_LENGTH = 5000

# Evaluation timeout
EVALUATION_TIMEOUT_SECONDS = 300      # 5 minutes
MAX_EVALUATION_TIMEOUT = 1800         # 30 minutes

# ============================================================================
# Optimization Performance Constants
# ============================================================================

# Parallel processing
DEFAULT_N_JOBS = -1                   # Use all available cores
MAX_N_JOBS = 8
MIN_N_JOBS = 1

# Memory management
MAX_MEMORY_USAGE_MB = 4096            # 4GB
MEMORY_CHECK_INTERVAL = 60            # seconds

# Performance monitoring
OPTIMIZATION_TIMEOUT_HOURS = 24       # 24 hours max
CHECKPOINT_SAVE_INTERVAL = 10         # trials

# ============================================================================
# Optimization Parameter Constants
# ============================================================================

# Reward function parameters
REWARD_WEIGHT_RANGE = (0.0, 2.0)
REWARD_SCALE_RANGE = (0.1, 10.0)
PENALTY_MULTIPLIER_RANGE = (0.1, 5.0)

# Risk parameters
RISK_AVERSION_RANGE = (0.0, 1.0)
VOLATILITY_PENALTY_RANGE = (0.0, 1.0)

# Time-based parameters
TIME_DECAY_RANGE = (0.8, 1.0)
HORIZON_WEIGHT_RANGE = (0.1, 1.0)

# ============================================================================
# Optimization Validation Constants
# ============================================================================

# Validation settings
VALIDATION_RATIO = 0.2                # 20% of data for validation
CROSS_VALIDATION_FOLDS = 5

# Statistical significance
SIGNIFICANCE_LEVEL = 0.05
MIN_SAMPLE_SIZE = 30

# Stability checks
STABILITY_WINDOW_SIZE = 10
MIN_STABILITY_THRESHOLD = 0.8

# ============================================================================
# Optimization Convergence Constants
# ============================================================================

# Convergence criteria
CONVERGENCE_TOLERANCE = 1e-4
MAX_CONSECUTIVE_NO_IMPROVEMENT = 20
MIN_IMPROVEMENT_THRESHOLD = 0.001

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.01

# ============================================================================
# Optimization Logging Constants
# ============================================================================

# Logging intervals
LOG_INTERVAL = 5                      # trials
DETAILED_LOG_INTERVAL = 25

# Log levels
OPTIMIZATION_LOG_LEVEL = "INFO"
ERROR_LOG_LEVEL = "ERROR"
DEBUG_LOG_LEVEL = "DEBUG"

# Report generation
REPORT_UPDATE_INTERVAL = 50           # trials
FINAL_REPORT_GENERATION = True

# ============================================================================
# Optimization Error Handling Constants
# ============================================================================

# Error recovery
MAX_OPTIMIZATION_RETRIES = 3
RETRY_DELAY_SECONDS = 10
CIRCUIT_BREAKER_TIMEOUT = 600         # 10 minutes

# Error classification
CRITICAL_OPTIMIZATION_ERRORS = [
    "evaluation_failed",
    "study_creation_failed",
    "parameter_validation_failed",
]

WARNING_OPTIMIZATION_ERRORS = [
    "convergence_slow",
    "memory_warning",
    "timeout_warning",
]

# ============================================================================
# Optimization Storage Constants
# ============================================================================

# Database settings
DATABASE_TIMEOUT = 30                 # seconds
DATABASE_CHECK_SAME_THREAD = False

# File paths
OPTIMIZATION_RESULTS_DIR = "optimization_results"
CHECKPOINT_FILE_SUFFIX = "_checkpoint.db"
RESULTS_FILE_SUFFIX = "_results.json"

# Backup settings
AUTO_BACKUP_INTERVAL = 100            # trials
MAX_BACKUP_FILES = 5

# ============================================================================
# Optimization UI Constants
# ============================================================================

# Progress display
PROGRESS_BAR_UPDATE_INTERVAL = 1      # second
STATUS_UPDATE_INTERVAL = 5            # seconds

# Visualization settings
PLOT_UPDATE_INTERVAL = 20             # trials
MAX_PLOTS_PER_FIGURE = 4

# Dashboard settings
DASHBOARD_REFRESH_INTERVAL = 30       # seconds
METRICS_DISPLAY_PRECISION = 4

# ============================================================================
# Optimization Benchmark Constants
# ============================================================================

# Benchmark comparisons
BASELINE_REWARD_THRESHOLD = 0.0
IMPROVEMENT_THRESHOLD = 0.05          # 5% improvement required

# Performance targets
TARGET_OPTIMIZATION_TIME_HOURS = 12
TARGET_IMPROVEMENT_PERCENTAGE = 10

# Statistical benchmarks
MIN_SHARPE_RATIO = 1.0
MAX_DRAWDOWN_LIMIT = 0.1
