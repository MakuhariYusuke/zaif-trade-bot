# ZTB Utils API Documentation

## Overview

The `ztb.utils` package provides common utilities used throughout the ZTB trading system. This document describes the key modules and their APIs.

## Modules

### data_validation.py

Data validation utilities for ensuring data quality and integrity.

#### Functions

##### `validate_dataframe(df, required_columns, column_types=None, min_rows=1)`

Validates DataFrame structure and content.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `required_columns` (List[str]): Required column names
- `column_types` (Optional[Dict[str, str]]): Expected column dtypes
- `min_rows` (int): Minimum number of rows required

**Returns:** `bool` - True if validation passes

##### `validate_numeric_array(data, name, allow_nan=True, min_value=None, max_value=None)`

Validates numeric array data.

**Parameters:**
- `data` (ArrayLike): Numeric data to validate
- `name` (str): Name for logging
- `allow_nan` (bool): Whether NaN values are allowed
- `min_value` (Optional[float]): Minimum allowed value
- `max_value` (Optional[float]): Maximum allowed value

**Returns:** `bool` - True if validation passes

##### `validate_config_dict(config, required_keys, validators=None)`

Validates configuration dictionary.

**Parameters:**
- `config` (Dict[str, Any]): Configuration to validate
- `required_keys` (List[str]): Required keys
- `validators` (Optional[Dict[str, Callable]]): Validation functions

**Returns:** `bool` - True if validation passes

### types.py

Type definitions and protocols used across the codebase.

#### Type Aliases

- `NumericType`: `Union[int, float, np.number]`
- `ArrayLike`: `Union[np.ndarray, pd.Series, List[NumericType]]`
- `ActionType`: `int` (0: HOLD, 1: BUY, 2: SELL)
- `ActionMask`: `np.ndarray` (Boolean array for valid actions)

#### TypedDict Classes

##### `TrainingConfig`

Training configuration dictionary with optional fields for PPO parameters.

##### `EnvironmentConfig`

Environment configuration dictionary for trading environments.

##### `ModelConfig`

Model configuration dictionary for neural network architecture.

#### Protocols

##### `TradingEnvironment`

Protocol for trading environments with standard RL interface.

##### `FeatureCalculator`

Protocol for feature calculation classes.

##### `CallbackProtocol`

Protocol for training callbacks.

### talib_wrapper.py

Ta-Lib integration with fallback implementations.

#### Class: `TaLibWrapper`

##### Static Methods

###### `check_talib_availability()`

Check if Ta-Lib is available.

**Returns:** `bool`

###### `sma(data, period)`

Simple Moving Average.

**Parameters:**
- `data` (ArrayLike): Input price data
- `period` (int): Period for calculation

**Returns:** `np.ndarray`

###### `ema(data, period)`

Exponential Moving Average.

**Parameters:**
- `data` (ArrayLike): Input price data
- `period` (int): Period for calculation

**Returns:** `np.ndarray`

###### `rsi(data, period=14)`

Relative Strength Index.

**Parameters:**
- `data` (ArrayLike): Input price data
- `period` (int): Period for calculation

**Returns:** `np.ndarray`

###### `macd(data, fast_period=12, slow_period=26, signal_period=9)`

MACD (Moving Average Convergence Divergence).

**Parameters:**
- `data` (ArrayLike): Input price data
- `fast_period`, `slow_period`, `signal_period` (int): MACD parameters

**Returns:** `Tuple[np.ndarray, np.ndarray, np.ndarray]` - (MACD, Signal, Histogram)

###### `bbands(data, period=20, nbdevup=2.0, nbdevdn=2.0)`

Bollinger Bands.

**Parameters:**
- `data` (ArrayLike): Input price data
- `period` (int): Period for calculation
- `nbdevup`, `nbdevdn` (float): Standard deviation multipliers

**Returns:** `Tuple[np.ndarray, np.ndarray, np.ndarray]` - (Upper, Middle, Lower)

### config.py

Central configuration management.

#### Class: `ZTBConfig`

##### Methods

###### `get(key, default=None)`

Get configuration value from environment variables.

###### `get_bool(key, default=False)`

Get boolean configuration value.

###### `get_int(key, default=0)`

Get integer configuration value.

###### `get_float(key, default=0.0)`

Get float configuration value.

### file_utils.py

File I/O utilities.

#### Functions

##### `safe_json_load(file_path, default=None)`

Safely load JSON from file with error handling.

##### `safe_json_dump(data, file_path, indent=2, default=None)`

Safely dump data to JSON file with error handling.

### logging_utils.py

Logging utilities.

#### Functions

##### `setup_logging(level=INFO, format_string=None)`

Set up basic logging configuration.

##### `get_logger(name)`

Get a configured logger instance.

## Usage Examples

### Data Validation

```python
from ztb.utils.data_validation import validate_dataframe
import pandas as pd

df = pd.read_csv('market_data.csv')
if validate_dataframe(df, ['open', 'high', 'low', 'close', 'volume']):
    print("Data validation passed")
```

### Ta-Lib Integration

```python
from ztb.utils.talib_wrapper import TaLibWrapper
import numpy as np

prices = np.random.rand(100)
sma_values = TaLibWrapper.sma(prices, period=20)
rsi_values = TaLibWrapper.rsi(prices, period=14)
```

### Type-Safe Configuration

```python
from ztb.utils.config import ZTBConfig
from ztb.utils.types import TrainingConfig

config = ZTBConfig()
learning_rate = config.get_float('LEARNING_RATE', 0.0003)

train_config: TrainingConfig = {
    'total_timesteps': 1000000,
    'learning_rate': learning_rate,
    'batch_size': 64
}
```

## Error Handling

All utility functions include comprehensive error handling and logging. Check logs for detailed error information when operations fail.
