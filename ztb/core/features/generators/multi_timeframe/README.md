# Multi-Timeframe Feature Engineering System

## Overview

The Multi-Timeframe Feature Engineering System provides comprehensive feature generation across multiple timeframes for enhanced reinforcement learning in trading environments. This system integrates features from 1 minute, 5 minute, 15 minute, 1 hour, 4 hour, and 1 day timeframes to provide richer market context.

## Supported Timeframes

- **1min**: High-frequency features for scalping and short-term analysis
- **5min**: Balanced feature set for primary training timeframe (default)
- **15min**: Medium-term analysis with trend confirmation
- **1hour**: Hourly analysis for swing trading patterns
- **4hour**: High-quality features for position trading
- **1day**: Daily analysis for long-term market structure

## Architecture

### Core Components

1. **MultiTimeframeFeatureSystem**: Main interface for the complete system
2. **MultiTimeframeFeatureEngineer**: Core feature generation engine
3. **MultiTimeframeConfig**: Configuration management system
4. **MultiTimeframeDataPipeline**: Data loading and synchronization

### Directory Structure

```
ztb/features/multi_timeframe/
├── __init__.py                 # Main interface and convenience functions
├── engine.py                   # Core feature engineering logic
├── config.py                   # Configuration management
├── data_pipeline.py           # Data loading and processing
├── config/
│   └── multi_timeframe_config.json  # Default configuration
└── test_system.py             # System testing and validation
```

## Usage

### Basic Usage

```python
from ztb.features.multi_timeframe import MultiTimeframeFeatureSystem
from ztb.features.timeframe import Timeframe

# Initialize the system
system = MultiTimeframeFeatureSystem()

# Process multi-timeframe features
features_df = system.process_multi_timeframe_data()
```

### Advanced Usage

```python
from ztb.features.multi_timeframe import process_multi_timeframe_features
from ztb.features.timeframe import Timeframe

# Define data files for each timeframe
data_files = {
    Timeframe.M1: "data/btc_jpy_1min.csv",
    Timeframe.M5: "data/btc_jpy_5min.csv",
    Timeframe.H1: "data/btc_jpy_1hour.csv",
    Timeframe.D1: "data/btc_jpy_1day.csv",
}

# Process features with custom configuration
features_df = process_multi_timeframe_features(
    data_files=data_files,
    config_path="path/to/custom/config.json",
    feature_set="high_quality"
)
```

### Configuration

The system uses a JSON configuration file to control behavior:

```json
{
  "enabled_timeframes": ["1min", "5min", "15min", "1hour", "4hour", "1day"],
  "base_timeframe": "5min",
  "feature_sets": {
    "1min": {
      "feature_set": "minimal",
      "window_sizes": [3, 5, 7, 10],
      "max_features": 50
    }
  },
  "integration": {
    "include_timeframe_indicators": true,
    "feature_prefixing": true
  }
}
```

## Features

### Timeframe-Aware Feature Generation

- **Adaptive Window Sizes**: Different window sizes optimized for each timeframe
- **Feature Set Selection**: Choose from minimal, full, or high_quality feature sets
- **Quality Control**: Automatic feature filtering based on variance, correlation, and NaN rates

### Data Synchronization

- **Timestamp Alignment**: Synchronize data across timeframes to common timestamps
- **Missing Data Handling**: Forward-fill and interpolation for missing data points
- **Resampling**: Generate missing timeframes through intelligent resampling

### Integration Features

- **Timeframe Prefixing**: Features prefixed with timeframe (e.g., `5min_sma_20`)
- **Timeframe Indicators**: Hierarchy and relationship indicators
- **Weighted Integration**: Configurable weights for different timeframes

## Configuration Options

### Timeframe-Specific Settings

Each timeframe can be configured independently:

- **feature_set**: Feature complexity level (minimal/full/high_quality)
- **window_sizes**: Technical indicator window sizes
- **max_features**: Maximum number of features to generate

### Integration Settings

- **include_timeframe_indicators**: Add timeframe metadata features
- **timeframe_alignment_method**: Data synchronization method
- **feature_prefixing**: Prefix features with timeframe names

### Quality Control

- **max_nan_rate**: Maximum allowed NaN rate per feature
- **min_variance**: Minimum variance threshold
- **max_correlation**: Maximum correlation threshold for feature reduction

## Testing

Run the test suite to validate the system:

```bash
cd ztb/features/multi_timeframe
python test_system.py
```

## Integration with SAC v427

The multi-timeframe system is designed to integrate seamlessly with the SAC v427 training system:

1. Replace single timeframe feature generation with multi-timeframe features
2. Update environment configuration to handle increased feature dimensions
3. Adjust reward functions to leverage timeframe relationships

## Performance Considerations

- **Memory Usage**: Multi-timeframe features significantly increase memory requirements
- **Processing Time**: Feature generation time scales with number of timeframes
- **Parallel Processing**: Enable parallel processing for better performance
- **Caching**: Use feature caching for repeated processing

## Future Enhancements

- **Advanced Synchronization**: More sophisticated timestamp alignment
- **Cross-Timeframe Features**: Features that combine multiple timeframes
- **Adaptive Feature Selection**: Dynamic feature selection based on market conditions
- **Real-time Processing**: Streaming multi-timeframe feature generation
