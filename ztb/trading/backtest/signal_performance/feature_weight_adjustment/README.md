# Dynamic Feature Weight Adjustment System

## Overview

The Dynamic Feature Weight Adjustment System provides automatic optimization of feature weights for SAC (Soft Actor-Critic) learning-based trading systems. The system dynamically adjusts feature importance based on real-time performance analysis and market conditions to improve trading performance.

## Key Features

- **Multiple Adjustment Strategies**: Performance-driven, correlation-based, and reinforcement learning approaches
- **Real-time Performance Evaluation**: Continuous monitoring and analysis of weight adjustment effectiveness
- **Market Condition Adaptation**: Automatic adjustment based on volatility and trend strength
- **Modular Architecture**: Extensible design for easy addition of new strategies and data providers
- **Comprehensive Validation**: Built-in validation and error handling for robust operation

## Architecture

### Core Components

- **DynamicWeightAdjuster**: Main orchestrator for weight adjustments
- **Adjustment Strategies**: Different algorithms for weight optimization
  - `PerformanceDrivenStrategy`: Adjusts based on win rate, returns, and risk metrics
  - `CorrelationBasedStrategy`: Reduces weights for highly correlated features
  - `ReinforcementLearningStrategy`: Uses RL to learn optimal weight adjustments
- **PerformanceEvaluator**: Measures effectiveness of adjustments
- **Data Providers**: Interfaces for SAC learning and signal performance data

### Directory Structure

```
feature_weight_adjustment/
├── __init__.py                      # Main package exports
├── core/                           # Core implementation
│   ├── weight_adjuster.py          # Main adjustment logic
│   ├── adjustment_strategies.py    # Strategy implementations
│   └── performance_evaluator.py    # Performance analysis
├── interfaces/                     # Abstract interfaces
│   ├── adjustment_interface.py     # Strategy interface
│   └── data_provider_interface.py  # Data provider interface
├── config/                         # Configuration classes
│   └── adjustment_config.py        # Configuration management
├── utils/                          # Utility functions
│   ├── data_processor.py           # Data processing utilities
│   └── validation_utils.py         # Validation functions
└── tests/                          # Unit tests
    └── test_weight_adjuster.py     # Test cases
```

## Usage

### Basic Usage

```python
from ztb.trading.backtest.signal_performance.feature_weight_adjustment import (
    DynamicWeightAdjuster,
    AdjustmentConfig,
)

# Create configuration
config = AdjustmentConfig(
    strategy="performance_driven",
    adjustment_rate=0.05,
    min_weight=0.01,
    max_weight=1.0,
)

# Initialize adjuster
adjuster = DynamicWeightAdjuster(config)

# Current feature weights
current_weights = {
    "rsi": 0.3,
    "macd": 0.3,
    "volume": 0.2,
    "price": 0.2,
}

# Performance data
performance_data = {
    "win_rate": 0.65,
    "total_return": 0.045,
    "feature_performance": {
        "rsi": {"win_rate": 0.7, "return_contribution": 0.025},
        "macd": {"win_rate": 0.6, "return_contribution": 0.015},
        "volume": {"win_rate": 0.5, "return_contribution": 0.003},
        "price": {"win_rate": 0.55, "return_contribution": 0.002},
    }
}

# Feature importance scores
feature_importance = {
    "rsi": 0.8,
    "macd": 0.7,
    "volume": 0.5,
    "price": 0.6,
}

# Adjust weights
new_weights = adjuster.adjust_weights(
    current_weights,
    performance_data,
    feature_importance,
)

print("Adjusted weights:", new_weights)
```

### Using Different Strategies

```python
# Performance-driven strategy
adjuster_perf = DynamicWeightAdjuster({
    "strategy": "performance_driven",
    "adjustment_rate": 0.05,
})

# Correlation-based strategy
adjuster_corr = DynamicWeightAdjuster({
    "strategy": "correlation_based",
    "correlation_threshold": 0.8,
})

# Reinforcement learning strategy
adjuster_rl = DynamicWeightAdjuster({
    "strategy": "reinforcement_learning",
    "learning_rate": 0.01,
})
```

### Performance Evaluation

```python
from ztb.trading.backtest.signal_performance.feature_weight_adjustment import PerformanceEvaluator

evaluator = PerformanceEvaluator()

# Evaluate adjustment impact
impact = evaluator.evaluate_adjustment_impact(
    before_weights=current_weights,
    after_weights=new_weights,
    performance_data=performance_data,
    feature_importance=feature_importance,
)

print(f"Impact score: {impact['impact_score']}")
print(f"Recommendation: {impact['recommendation']}")

# Get overall effectiveness
effectiveness = evaluator.get_adjustment_effectiveness()
print(f"Success rate: {effectiveness['success_rate']:.2%}")
```

## Configuration

### AdjustmentConfig Parameters

- `strategy`: Adjustment strategy ("performance_driven", "correlation_based", "reinforcement_learning")
- `adjustment_rate`: Maximum weight change per adjustment (0.0-1.0)
- `min_weight`: Minimum allowed weight (0.0-1.0)
- `max_weight`: Maximum allowed weight (0.0-1.0)
- `performance_window`: Number of past adjustments to consider (int)
- `enable_market_adaptation`: Whether to adapt to market conditions (bool)

### Strategy-Specific Configuration

#### PerformanceDrivenStrategy
- `performance_weights`: Weights for different performance metrics
- `market_adaptation`: Market condition adjustment factors

#### CorrelationBasedStrategy
- `correlation_threshold`: Threshold for considering features correlated (0.0-1.0)
- `redundancy_penalty`: Penalty for correlated features (0.0-1.0)

#### ReinforcementLearningStrategy
- `learning_rate`: Q-learning learning rate (0.0-1.0)
- `discount_factor`: Future reward discount factor (0.0-1.0)
- `exploration_rate`: Probability of random actions (0.0-1.0)

## Testing

Run the unit tests:

```bash
cd ztb/trading/backtest/signal_performance/feature_weight_adjustment
python -m pytest tests/
```

## Integration

The system integrates with the existing SAC learning pipeline:

1. **Data Collection**: Gather performance metrics and feature correlations
2. **Weight Adjustment**: Apply selected strategy to optimize weights
3. **Performance Evaluation**: Measure impact of adjustments
4. **Feedback Loop**: Use results to improve future adjustments

## Performance Considerations

- **Memory Usage**: Strategies maintain history for learning
- **Computation Time**: RL strategy requires more computation
- **Convergence**: Different strategies have different convergence characteristics
- **Stability**: Rate limiting prevents extreme weight changes

## Extensibility

### Adding New Strategies

1. Implement `WeightAdjustmentInterface`
2. Register with `AdjustmentStrategyRegistry`
3. Add configuration support in `AdjustmentConfig`

### Adding Data Providers

1. Implement `DataProviderInterface`
2. Provide performance data and feature correlations
3. Integrate with existing data pipeline

## Troubleshooting

### Common Issues

1. **No weight changes**: Check performance data validity
2. **Unstable adjustments**: Increase rate limiting or change strategy
3. **Poor performance**: Verify feature importance scores
4. **Memory issues**: Reduce history window sizes

### Debugging

Enable logging to see adjustment details:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Future Enhancements

- **Ensemble Strategies**: Combine multiple strategies
- **Online Learning**: Continuous adaptation without resets
- **Feature Selection**: Automatic feature addition/removal
- **Multi-objective Optimization**: Balance multiple performance metrics