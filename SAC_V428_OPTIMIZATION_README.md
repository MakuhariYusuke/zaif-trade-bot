# SAC v428 Advanced Optimization Framework

## Overview

This framework provides comprehensive optimization capabilities for the SAC v428 trading model, addressing three critical optimization gaps:

1. **Extended Duration Backtesting** - Test model robustness over 2+ years of market data
2. **Market Condition Analysis** - Analyze performance across different market regimes
3. **Hyperparameter Optimization** - Automated parameter tuning with cross-validation

## Components

### Core Modules

- **`ztb/analysis/market_regime_classifier.py`** - Advanced market condition classification using technical indicators
- **`ztb/analysis/regime_performance_analyzer.py`** - Performance analysis by market regime with statistical validation
- **`ztb/optimization/hyperparameter_optimizer.py`** - Multi-method hyperparameter optimization framework
- **`ztb/optimization/run_optimization.py`** - Complete optimization pipeline orchestrator

### Configuration

- **`configs/sac_v428_extended_backtest.json`** - Extended backtest configuration with optimization parameters

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install optuna scikit-learn pandas numpy
```

### Run Complete Optimization Pipeline

```bash
python ztb/optimization/run_optimization.py \
  --config configs/sac_v428_extended_backtest.json \
  --full-pipeline
```

This will execute:
1. Extended duration backtest (732 days)
2. Market condition analysis
3. Hyperparameter optimization (20 trials)
4. Generate comprehensive report

### Individual Components

#### Extended Backtest Only
```bash
python ztb/optimization/run_optimization.py \
  --config configs/sac_v428_extended_backtest.json \
  --backtest-only
```

#### Market Analysis Only
```bash
python ztb/optimization/run_optimization.py \
  --config configs/sac_v428_extended_backtest.json \
  --analysis-only
```

#### Hyperparameter Optimization Only
```bash
python ztb/optimization/run_optimization.py \
  --config configs/sac_v428_extended_backtest.json \
  --optimize-only
```

#### Generate Report
```bash
python ztb/optimization/run_optimization.py \
  --config configs/sac_v428_extended_backtest.json \
  --generate-report
```

## Configuration Details

### Extended Backtest Configuration

The configuration file includes:

- **Data Source**: 732-day BTC/JPY dataset (2022-2024)
- **Evaluation Intervals**: Multiple market periods (full period, bull/bear markets, high volatility)
- **Memory Optimization**: Chunked processing for large datasets
- **Hyperparameter Space**: Bayesian optimization ranges for SAC parameters

### Market Regime Classification

Classifies market conditions using:
- Trend strength (SMA, EMA crossovers)
- Momentum (MACD, RSI)
- Volatility (Bollinger Bands, ATR)
- Volume analysis

### Hyperparameter Optimization

Supports multiple methods:
- **Bayesian Optimization** (Optuna) - Preferred for efficiency
- **Random Search** - Fallback when Optuna unavailable
- **Grid Search** - Deterministic parameter sweeps

## Output Structure

Results are saved in timestamped directories under `optimization_results/`:

```
optimization_results/20241201_143000/
├── extended_backtest_results.json
├── market_condition_analysis.json
├── regime_statistics.json
├── hyperparameter_optimization.json
├── optimization_summary.json
└── optimization_report.md
```

## Key Features

### Statistical Validation
- Bootstrap confidence intervals for performance metrics
- Regime transition impact analysis
- Comparative statistical testing between market conditions

### Memory Optimization
- Chunked data processing for extended backtests
- Efficient feature computation
- Optimized data structures

### Cross-Validation Support
- Time-series aware cross-validation
- Multiple evaluation metrics
- Overfitting detection

### Automated Reporting
- Comprehensive performance summaries
- Market regime insights
- Optimization recommendations
- Visual performance comparisons

## Performance Metrics

The framework evaluates models using:
- **Sharpe Ratio** - Risk-adjusted returns
- **Win Rate** - Trade success percentage
- **Profit Factor** - Gross profit / gross loss
- **Maximum Drawdown** - Peak-to-trough decline
- **Calmar Ratio** - Annual return / max drawdown

## Best Practices

1. **Start with Extended Backtest** - Validate model stability over long periods
2. **Analyze Market Conditions** - Understand regime-specific performance
3. **Optimize Hyperparameters** - Use insights from market analysis
4. **Validate on Holdout Data** - Test optimized parameters on unseen data
5. **Monitor in Production** - Track performance degradation signals

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk_size in configuration
2. **Optuna Not Available**: Framework automatically falls back to random search
3. **Dataset Not Found**: Verify data/btc_jpy_extended_dataset.csv exists
4. **Import Errors**: Ensure all ztb modules are properly installed

### Performance Tuning

- Increase `n_trials` for better optimization (default: 20)
- Enable `cross_validation` for more robust results
- Adjust `chunk_size` based on available memory
- Use `regime_specific_optimization` for targeted tuning

## Integration

The optimization framework integrates seamlessly with:
- **Unified Trainer** - SAC model training infrastructure
- **Config Manager** - Centralized configuration management
- **Logging Utils** - Comprehensive logging and monitoring

## Future Enhancements

- Multi-objective optimization (returns + risk)
- Ensemble model optimization
- Real-time adaptation capabilities
- Advanced feature selection
- Neural architecture search

---

*Generated for SAC v428 Advanced Optimization Framework*