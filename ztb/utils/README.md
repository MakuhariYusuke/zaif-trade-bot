# ZTB Utils - Core Utilities Library

This directory contains core utility modules for the ZTB (Zaif Trade Bot)
system, providing essential functionality for configuration management,
reliability, observability, and data processing.

## Overview

The ZTB Utils library provides:

- **Configuration Management**: Centralized environment variable handling
- **Reliability**: Rate limiting
- **Observability**: Structured logging and monitoring
- **Data Processing**: Memory optimization and validation

## Core Modules

### Configuration Management

#### `config.py` - ZTBConfig

Centralized configuration management using environment variables.

```python
from ztb.utils.config import ZTBConfig

config = ZTBConfig()

# Get configuration values with type safety
api_key = config.get('API_KEY')
timeout = config.get_int('TIMEOUT', 30)
enabled = config.get_bool('FEATURE_ENABLED', False)
rate = config.get_float('SUCCESS_RATE', 0.95)
```

**Features:**

- Type-safe configuration retrieval
- Environment variable integration
- Default value support
- Validation and error handling

### Reliability & Resilience

#### `rate_limiter.py` - Rate Limiting

Token bucket rate limiting for API calls and resource protection.

```python
from ztb.utils.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

config = RateLimitConfig(requests_per_second=5.0, burst_limit=10)
limiter = TokenBucketRateLimiter(config)

# Check rate limit before API call
if await limiter.acquire():
    # Make API call
    pass
```

### Observability & Monitoring

#### `observability.py` - Structured Logging

JSON-formatted logging with correlation IDs and metrics.

```python
from ztb.utils.observability import get_logger, generate_correlation_id

logger = get_logger(__name__)
correlation_id = generate_correlation_id()

logger.info("Operation started", extra={
    "correlation_id": correlation_id,
    "operation": "trade_execution"
})
```

### Trading Metrics & Analytics

#### `metrics/trading_metrics.py` - Performance Metrics

Advanced trading performance metrics calculation (Sharpe/Sortino/Calmar ratios).

```python
from ztb.utils.metrics.trading_metrics import sharpe_ratio, sharpe_with_stats

# Calculate Sharpe ratio
returns = [0.01, 0.02, -0.005, 0.015, -0.01]
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)

# Calculate Sharpe statistics across multiple runs
sharpes = [1.2, 1.5, 0.8, 1.8, 1.1]
stats = sharpe_with_stats(sharpes)
print(f"Mean Sharpe: {stats['mean']:.3f} ± {stats['std']:.3f}")
```

**Features:**

- Sharpe ratio with annualization
- Sortino ratio (downside deviation)
- Calmar ratio (drawdown-adjusted return)
- Statistical analysis across multiple runs
- Safe operation with error handling

### Data Processing & Optimization

#### `data/data_generation.py` - Synthetic Data Generation

Generate synthetic market data for testing and experimentation.

```python
from ztb.utils.data.data_generation import generate_synthetic_data

# Generate synthetic OHLCV data
df = generate_synthetic_data(n_rows=5000, freq="1H")
print(f"Generated {len(df)} rows of synthetic data")
```

**Features:**

- Configurable data size and frequency
- Realistic price movements with volatility
- Episode ID generation for reinforcement learning
- Safe operation with error handling

#### `data/outlier_detection.py` - Outlier Detection

Statistical outlier detection using IQR and Z-score methods.

```python
from ztb.utils.data.outlier_detection import detect_outliers_iqr, detect_outliers_zscore

# Detect outliers using IQR method
df_clean, q1, q3 = detect_outliers_iqr(df, "price")

# Detect outliers using Z-score method
df_outliers = detect_outliers_zscore(df, "volume", threshold=3.0)
```

**Features:**

- IQR (Interquartile Range) method
- Z-score method with configurable threshold
- Safe operation with error handling
- Integration with pandas DataFrames

#### `data/report_generator.py` - Report Generation

Generate analysis reports from data processing results.

#### `memory/dtypes.py` - Memory Optimization

Pandas DataFrame dtype optimization for reduced memory footprint.

```python
from ztb.utils.memory.dtypes import optimize_dtypes

# Optimize DataFrame memory usage
df_optimized, report = optimize_dtypes(df)

print(f"Memory saved: {report.memory_saved_mb:.2f} MB")
print(f"Reduction: {report.percent_reduction:.1f}%")
```

## Directory Structure

```text
ztb/utils/
├── cache/              # Caching utilities
├── ci_utils.py         # CI utilities
├── cli_common.py       # CLI utilities
├── config.py           # Configuration management
├── config_loader.py    # Configuration file loading
├── core/               # Core utilities
├── data/               # Data processing utilities
├── data_utils.py       # Data utilities
├── errors.py           # Error handling
├── fees/               # Fee calculation utilities
├── indicators/         # Technical indicators
├── io/                 # I/O utilities
├── logging/            # Logging utilities
├── logging_utils.py    # Logging utilities
├── memory/             # Memory optimization
│   └── memory_monitor.py # Memory monitoring
├── metrics/            # Trading metrics & analytics
├── notify/             # Notification utilities
├── observability.py    # Structured logging & monitoring
├── perf/               # Performance utilities
├── quality.py          # Quality assurance
├── rate_limiter.py     # Rate limiting
├── README.md           # This file
├── resource/           # Resource management
├── thresholds.py       # Threshold management
└── __init__.py
```

## Usage Examples

### Basic Setup

```python
from ztb.utils.config import ZTBConfig
from ztb.utils.observability import get_logger
from ztb.utils.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

# Initialize core components
config = ZTBConfig()
logger = get_logger(__name__)

# Setup rate limiting
rate_config = RateLimitConfig(requests_per_second=10.0)
limiter = TokenBucketRateLimiter(rate_config)

logger.info("ZTB Utils initialized successfully")
```

## Best Practices

### Configuration

- Use `ZTBConfig` for all environment variable access
- Provide sensible defaults for all configuration values
- Validate configuration values at startup

### Reliability

- Use rate limiting to prevent API abuse

### Observability

- Use structured logging with correlation IDs
- Include relevant context in log messages

### Performance

- Apply memory optimization to large DataFrames
- Use rate limiting to prevent resource exhaustion

## Testing

Most utilities include comprehensive test suites.

## Dependencies

Core dependencies (automatically managed):

- `pandas` - Data processing
- `numpy` - Numerical operations

Optional dependencies for extended functionality:

- Monitoring and observability integrations (e.g., `prometheus_client` for metrics, `sentry-sdk` for error tracking, `opentelemetry` for distributed tracing)

## Contributing

When adding new utilities:

1. Follow the established patterns in existing modules
2. Include comprehensive type hints
3. Add unit tests with good coverage
4. Update this README with new functionality
5. Ensure compatibility with the existing architecture

## Migration Guide

### From Custom Implementations

Replace custom utility code with standardized implementations:

```python
# Before
import os
api_key = os.getenv('API_KEY', 'default')

# After
from ztb.utils.config import ZTBConfig
config = ZTBConfig()
api_key = config.get('API_KEY', 'default')
```
