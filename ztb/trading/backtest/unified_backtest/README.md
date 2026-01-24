# Unified Backtest Framework

A comprehensive backtesting framework for trading strategies, designed to leverage SAC learning outcomes and support multiple strategy types including SAC models and Action Signal Guide.

## Features

- **Multiple Strategy Support**: SAC models, Action Signal Guide, and hybrid strategies
- **SAC Learning Integration**: Leverage SAC learning outcomes for enhanced analysis
- **Comprehensive Analysis**: Performance, risk, correlation, and signal quality analysis
- **Unified Interface**: Consistent API for all strategy types
- **Automated Reporting**: JSON and Markdown report generation
- **Statistical Testing**: Significance testing and comparative analysis

## Architecture

```
unified_backtest/
├── __init__.py                 # Package initialization
├── unified_backtester.py       # Main backtesting engine
├── strategy_base.py           # Strategy base classes and protocols
├── sac_strategy.py            # SAC-based trading strategy
├── action_signal_guide_strategy.py  # Action Signal Guide strategy
├── analyzer.py                 # Comprehensive analysis engine
└── run_unified_backtest.py     # CLI runner
```

## Quick Start

### Single Strategy Backtest

```python
from ztb.trading.backtest.unified_backtest import (
    UnifiedBacktester,
    SACStrategy,
    BacktestConfig
)

# Initialize backtester
backtester = UnifiedBacktester()

# Create SAC strategy
sac_strategy = SACStrategy(
    name="SAC_v444",
    model_path="models/sac_v444.zip"
)

# Register strategy
backtester.register_strategy("sac_v444", sac_strategy)

# Load market data
import pandas as pd
data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)

# Configure backtest
config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.001,
    enable_shorting=True
)

# Run backtest
result = backtester.run_backtest("sac_v444", data, config)

print(f"Total Return: {result.performance_metrics.total_return:.2%}")
print(f"Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.2f}")
```

### Strategy Comparison

```python
# Register multiple strategies
sac_strategy = SACStrategy(name="SAC_v444", model_path="models/sac_v444.zip")
asg_strategy = ActionSignalGuideStrategy(name="ActionSignalGuide")

backtester.register_strategy("sac", sac_strategy)
backtester.register_strategy("asg", asg_strategy)

# Compare strategies
results = backtester.compare_strategies(["sac", "asg"], data, config)

# Analyze comparison
from ztb.trading.backtest.unified_backtest import BacktestAnalyzer
analyzer = BacktestAnalyzer()
comparison = analyzer.compare_strategies(results)
```

### SAC Learning Outcome Analysis

```python
# Analyze SAC learning progression
sac_results = {
    "sac_v430": sac_v430_result,
    "sac_v435": sac_v435_result,
    "sac_v440": sac_v440_result,
    "sac_v444": sac_v444_result,
}

learning_analysis = analyzer.analyze_sac_learning_outcomes(sac_results)
print("Learning Progression:", learning_analysis["learning_progression"])
```

## CLI Usage

### Single Strategy Backtest

```bash
python -m ztb.trading.backtest.unified_backtest.run_unified_backtest \
    --strategy SAC_v444 \
    --data data/market_data.csv \
    --initial-capital 100000
```

### Strategy Comparison

```bash
python -m ztb.trading.backtest.unified_backtest.run_unified_backtest \
    --strategies SAC_v444 ActionSignalGuide \
    --data data/market_data.csv \
    --initial-capital 100000
```

## Strategy Types

### SAC Strategy

Leverages trained SAC models with regime adaptation:

```python
sac_strategy = SACStrategy(
    name="SAC_v444",
    model_path="models/sac_v444.zip",
    regime_classifier_path="models/regime_classifier.pkl"  # Optional
)
```

### Action Signal Guide Strategy

Pattern-based signal generation:

```python
asg_strategy = ActionSignalGuideStrategy(
    name="ActionSignalGuide",
    pattern_types=["candlestick", "fibonacci", "wave", "harmonic"]
)
```

## Analysis Features

### Performance Analysis
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Trade statistics
- Temporal patterns

### SAC-Specific Analysis
- Learning progression across versions
- Regime adaptation effectiveness
- Hyperparameter sensitivity
- Training outcome correlation

### Signal Quality Analysis
- Signal distribution and timing
- Pattern recognition accuracy
- Signal effectiveness metrics

### Comparative Analysis
- Strategy performance comparison
- Statistical significance testing
- Risk-adjusted performance ranking

## Advanced Analysis Features

The unified backtest framework integrates advanced analysis capabilities from archived scripts:

### Risk Analysis
- **Sharpe Ratio**: Risk-adjusted return calculation
- **Sortino Ratio**: Downside deviation-based risk adjustment
- **Value at Risk (VaR)**: Historical simulation-based risk measure
- **Expected Shortfall (CVaR)**: Conditional Value at Risk
- **Omega Ratio**: Probability-weighted ratio of gains vs losses

### Feature Analysis
- **Permutation Importance**: Model-agnostic feature importance
- **Correlation Analysis**: Feature-target relationships
- **Multicollinearity Detection**: Feature correlation filtering

### Market Regime Analysis
- **Volatility Clustering**: Volatility-based regime detection
- **Trend Following**: Trend-based regime classification
- **Regime Transition Analysis**: Market state change patterns

### Walkforward Analysis
- **Rolling Window Validation**: Time-series validation
- **Performance Stability**: Consistency across time periods
- **Sharpe Ratio Consistency**: Risk-adjusted return stability

### Temporal Analysis
- **Seasonal Patterns**: Monthly performance analysis
- **Drawdown Duration**: Loss period analysis
- **Recovery Patterns**: Post-drawdown recovery analysis

## Configuration

### BacktestConfig

```python
config = BacktestConfig(
    initial_capital=100000.0,      # Starting capital
    commission=0.001,              # Commission per trade (0.1%)
    slippage=0.0005,               # Slippage per trade (0.05%)
    max_position_size=1.0,         # Max position size (1.0 = 100%)
    enable_shorting=True,          # Allow short positions
    warmup_periods=100,            # Warmup periods for indicators
)
```

## Output Formats

### JSON Results
```json
{
  "strategy_name": "SAC_v444",
  "performance_metrics": {
    "total_return": 0.1567,
    "sharpe_ratio": 1.23,
    "max_drawdown": -0.089,
    "win_rate": 0.542
  },
  "trade_history": [...],
  "execution_time": 45.67
}
```

### Markdown Reports
- Performance summaries
- Risk analysis
- Trade analysis
- Strategy comparisons

## Integration with Existing Codebase

The unified backtest framework integrates with existing ztb components:

- **BacktestEngine**: Uses existing backtest engine for execution
- **MetricsCalculator**: Uses centralized metrics in `ztb.metrics.metrics` (backtest metrics module is deprecated)
- **ReportGenerator**: Uses `ztb.reporting.generators.backtest` (legacy module is deprecated)
- **Strategy Adapters**: Compatible with existing adapter interface

## SAC Learning Integration

### Learning Outcome Tracking
- Model performance across training versions
- Regime-specific performance analysis
- Hyperparameter impact assessment

### Signal Contribution Analysis
- Action Signal Guide impact on SAC decisions
- Correlation between signals and SAC actions
- Performance attribution analysis

## Best Practices

1. **Data Quality**: Ensure clean, properly formatted OHLCV data
2. **Strategy Validation**: Always validate strategies on out-of-sample data
3. **Risk Management**: Use appropriate position sizing and stop-loss rules
4. **Performance Attribution**: Analyze sources of returns and risk
5. **Statistical Significance**: Use statistical tests for meaningful comparisons

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Verify model file paths and formats
2. **Data Format Issues**: Ensure OHLCV columns are properly named
3. **Memory Issues**: For large datasets, consider data chunking
4. **Strategy Errors**: Check strategy initialization and parameter validation

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Parallel strategy execution
- Walk-forward optimization
- Monte Carlo simulation
- Advanced risk management
- Real-time signal integration
- Web-based visualization dashboard
