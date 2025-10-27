"""
Live Trading Constants - Constants specific to live trading operations.

This module defines constants used throughout the live trading system,
organized by functional area for better maintainability.
"""

# ============================================================================
# Live Trading Execution Constants
# ============================================================================

# Trading frequencies
DEFAULT_TRADING_INTERVAL_SECONDS = 60  # 1 minute
MIN_TRADING_INTERVAL_SECONDS = 10
MAX_TRADING_INTERVAL_SECONDS = 3600  # 1 hour

# Order execution
ORDER_TIMEOUT_SECONDS = 30
MAX_ORDER_RETRIES = 3
ORDER_RETRY_DELAY_SECONDS = 2

# Position limits
DEFAULT_MAX_POSITION_SIZE = 100000  # JPY
DEFAULT_MIN_POSITION_SIZE = 1000    # JPY
POSITION_SIZE_INCREMENT = 1000      # JPY

# ============================================================================
# Live Trading Risk Management Constants
# ============================================================================

# Stop loss settings
DEFAULT_STOP_LOSS_PERCENTAGE = 0.02  # 2%
MAX_STOP_LOSS_PERCENTAGE = 0.05      # 5%
MIN_STOP_LOSS_PERCENTAGE = 0.005     # 0.5%

# Take profit settings
DEFAULT_TAKE_PROFIT_RATIO = 2.0      # 2:1 reward-to-risk
MIN_TAKE_PROFIT_RATIO = 1.5
MAX_TAKE_PROFIT_RATIO = 5.0

# Risk per trade
DEFAULT_RISK_PER_TRADE = 0.01        # 1% of portfolio
MAX_RISK_PER_TRADE = 0.05            # 5% of portfolio

# ============================================================================
# Live Trading Monitoring Constants
# ============================================================================

# Health check intervals
HEALTH_CHECK_INTERVAL_SECONDS = 300  # 5 minutes
CONNECTION_CHECK_INTERVAL_SECONDS = 60  # 1 minute

# Performance monitoring
PERFORMANCE_LOG_INTERVAL = 3600      # 1 hour
METRICS_UPDATE_INTERVAL = 60         # 1 minute

# Alert thresholds
MAX_CONNECTION_FAILURES = 5
MIN_SUCCESS_RATE = 0.95              # 95% success rate
MAX_LATENCY_MS = 5000                # 5 seconds

# ============================================================================
# Live Trading Data Constants
# ============================================================================

# Market data buffers
PRICE_HISTORY_SIZE = 1000
ORDER_BOOK_DEPTH = 20
TRADE_HISTORY_SIZE = 100

# Data validation
MAX_PRICE_DEVIATION = 0.1            # 10% max deviation from last price
MIN_VOLUME_THRESHOLD = 1000          # Minimum volume for valid data

# Data refresh rates
MARKET_DATA_REFRESH_SECONDS = 1
ORDER_BOOK_REFRESH_SECONDS = 5

# ============================================================================
# Live Trading Model Constants
# ============================================================================

# Model prediction settings
PREDICTION_TIMEOUT_SECONDS = 5
MAX_PREDICTION_RETRIES = 2
MODEL_WARMUP_TIME_SECONDS = 30

# Feature computation
FEATURE_COMPUTATION_TIMEOUT = 2      # seconds
MAX_FEATURE_COMPUTATION_RETRIES = 2

# Action validation
ACTION_CONFIDENCE_THRESHOLD = 0.6    # Minimum confidence for action execution
MAX_CONSECUTIVE_HOLD_ACTIONS = 10

# ============================================================================
# Live Trading Communication Constants
# ============================================================================

# API rate limits
API_RATE_LIMIT_REQUESTS = 100         # requests per minute
API_RATE_LIMIT_WINDOW = 60            # seconds

# Notification settings
NOTIFICATION_COOLDOWN_SECONDS = 300   # 5 minutes between similar notifications
MAX_NOTIFICATIONS_PER_HOUR = 10

# Logging levels
DEFAULT_LOG_LEVEL = "INFO"
ERROR_LOG_LEVEL = "ERROR"
DEBUG_LOG_LEVEL = "DEBUG"

# ============================================================================
# Live Trading Error Handling Constants
# ============================================================================

# Error recovery
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_DELAY_SECONDS = 10
CIRCUIT_BREAKER_TIMEOUT = 300         # 5 minutes

# Error classification
CRITICAL_ERRORS = [
    "connection_failed",
    "authentication_failed",
    "insufficient_funds",
    "order_rejection",
]

WARNING_ERRORS = [
    "high_latency",
    "data_stale",
    "prediction_timeout",
]

# ============================================================================
# Live Trading Performance Constants
# ============================================================================

# Performance targets
TARGET_PROFITABILITY = 0.02          # 2% daily target
TARGET_SHARPE_RATIO = 1.5
TARGET_MAX_DRAWDOWN = 0.05           # 5% max drawdown

# Performance windows
DAILY_PERFORMANCE_WINDOW = 86400     # 24 hours in seconds
WEEKLY_PERFORMANCE_WINDOW = 604800   # 7 days in seconds

# Benchmark comparisons
BENCHMARK_RETURN_THRESHOLD = 0.01    # 1% vs benchmark

# ============================================================================
# Live Trading UI Constants
# ============================================================================

# Display settings
STATUS_UPDATE_INTERVAL = 5            # seconds
DASHBOARD_REFRESH_INTERVAL = 30       # seconds

# Alert levels
INFO_ALERT_LEVEL = "info"
WARNING_ALERT_LEVEL = "warning"
ERROR_ALERT_LEVEL = "error"
CRITICAL_ALERT_LEVEL = "critical"

# Display formatting
CURRENCY_PRECISION = 2
PERCENTAGE_PRECISION = 3
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\ztb\trading\live_trader\constants.py