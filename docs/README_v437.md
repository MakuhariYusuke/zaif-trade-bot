# SAC v437 Development

## Overview

SAC v437 is an enhanced version of the SAC trading model that addresses the over-trading issues found in v436. The key improvements include:

- **Expanded Feature Set**: Increased from 21 to 150+ dimensions using comprehensive feature engineering
- **Trading Frequency Control**: Built-in mechanisms to prevent excessive trading
- **Enhanced Feature Categories**: Regime awareness, correlation features, ensemble signals, and risk-adjusted indicators
- **Improved Directory Organization**: Better structure for experiments, models, and configurations

## Key Changes from v436

### Problems Addressed
- **Over-trading**: v436 showed 0.5 step intervals between trades leading to excessive transaction costs
- **Limited Features**: Only 21 features were being used effectively
- **Poor Risk Management**: Insufficient trading frequency controls

### Solutions Implemented
- **Feature Expansion**: 150+ dimensional feature space with multiple categories
- **Frequency Controls**: Minimum steps between trades, maximum trades per episode
- **Enhanced Indicators**: Technical, statistical, and market microstructure features
- **Better Organization**: Structured directories for v437 development

## Directory Structure

```
backtest_experiments/v437.1/    # Backtest results
checkpoints/v437/              # Model checkpoints
models/v437/                   # Saved models
config/v437/                   # v437 configurations
tensorboard/v437/              # Training logs
```

## Feature Categories

### 1. Price Features (10+)
- Close, Open, High, Low, Volume
- Returns, Log Returns, Price Changes
- Price Acceleration, Jerk

### 2. Momentum Features (20+)
- RSI variants (5, 10, 14, 20 periods)
- MACD, Stochastic, Williams %R
- CCI, ROC, Momentum oscillators

### 3. Volatility Features (15+)
- Bollinger Bands (multiple periods)
- ATR, NATR, Historical Volatility
- Volatility ratios and clustering

### 4. Trend Features (25+)
- Multiple MA types (SMA, EMA, DEMA, TEMA, HMA, WMA)
- ADX, Directional Movement
- Trend strength and direction

### 5. Volume Features (15+)
- Volume MAs and ratios
- OBV, VPT, VWAP
- Accumulation/Distribution, Chaikin Money Flow

### 6. Regime Features (20+)
- Trend/Bear/Sideways classification
- Volatility regime detection
- Momentum regimes

### 7. Correlation Features (25+)
- Price-volume correlations
- Autocorrelation analysis
- Volatility clustering, Leverage effect

### 8. Ensemble Features (20+)
- Multi-timeframe signals
- Ensemble confidence and diversity
- Prediction uncertainty

### 9. Risk Features (15+)
- VaR, CVaR calculations
- Sharpe, Sortino, Calmar ratios
- Risk-adjusted returns

### 10. Technical Indicators (20+)
- Ichimoku Cloud
- Fibonacci retracements
- Pivot points and support/resistance

### 11. Market Microstructure (15+)
- Spread analysis
- Order flow indicators
- Realized volatility measures

## Trading Frequency Controls

- **Minimum Steps Between Trades**: 5 steps minimum
- **Maximum Trades Per Episode**: 50 trades limit
- **Trading Penalty Coefficient**: 0.001 per excessive trade
- **Recent Volatility Check**: Prevents trading in high volatility
- **Trend Persistence**: Rewards sustained directional moves

## Usage

### Training

```bash
# Train v437 model with full feature set
python train_sac_v437.py --timesteps 100000 --feature-set full

# Train with custom configuration
python train_sac_v437.py --config config/v437/custom_config.json --feature-set high_quality
```

### Backtesting

```bash
# Backtest trained model
python backtest_sac_v437.py --model-path models/v437/sac_v437_final.zip

# Backtest with evaluation episodes
python backtest_sac_v437.py --model-path models/v437/sac_v437_final.zip --episodes 20
```

### Feature Management

```bash
# List available feature sets
python manage_features.py list

# Show feature set details
python manage_features.py show full
```

## Configuration

### Main Configuration (`config/v437/sac_v437_enhanced_config.json`)

```json
{
  "model_name": "sac_v437_enhanced_features",
  "version": "v437.1",
  "environment": {
    "enhanced_features": {
      "enabled": true,
      "target_dimensions": 150,
      "trading_frequency_control": {
        "min_steps_between_trades": 5,
        "max_trades_per_episode": 50,
        "trading_penalty_coefficient": 0.001
      }
    }
  }
}
```

### Feature Configuration (`config/features/feature_sets/v437_enhanced_features.json`)

Defines which feature categories are enabled and their parameters.

## Performance Expectations

### Improvements Over v436
- **Reduced Over-trading**: 50-70% reduction in trade frequency
- **Better Risk Management**: Improved Sharpe and Sortino ratios
- **Enhanced Returns**: More stable portfolio growth
- **Feature Utilization**: 150+ features vs 21 in v436

### Expected Metrics
- **Trades per Episode**: 20-40 (vs 100+ in v436)
- **Win Rate**: 55-65%
- **Profit Factor**: 1.2-1.5
- **Max Drawdown**: <15%

## Development Notes

### Feature Engineering Philosophy
- **Diversity over Complexity**: Multiple simple features beat single complex ones
- **Correlation Awareness**: Remove highly correlated features automatically
- **Domain Knowledge**: Incorporate trading domain expertise
- **Computational Efficiency**: Balance feature richness with training speed

### Training Considerations
- **Longer Training**: 100k+ timesteps recommended for feature convergence
- **Feature Normalization**: Critical for stable training
- **Regularization**: Entropy bonuses help exploration
- **Curriculum Learning**: Start simple, increase complexity

### Backtesting Validation
- **Multiple Episodes**: Test across different market conditions
- **Walk-Forward Analysis**: Validate on unseen data
- **Risk Metrics**: Focus on risk-adjusted returns
- **Overfitting Checks**: Compare training vs validation performance

## Future Enhancements

### Planned Features
- **Adaptive Feature Selection**: Dynamic feature importance during training
- **Market Regime Adaptation**: Automatic regime detection and switching
- **Ensemble Methods**: Multiple model combination strategies
- **Advanced Risk Management**: Dynamic position sizing

### Research Directions
- **Feature Importance Analysis**: Which features drive performance?
- **Temporal Patterns**: How do features evolve over time?
- **Market Condition Adaptation**: Different strategies for different regimes
- **Scalability**: How does performance scale with more features?

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce feature set to 'minimal' or 'high_quality'
2. **Training Instability**: Check feature normalization and learning rates
3. **Poor Performance**: Verify data quality and feature engineering
4. **Over-trading**: Increase trading frequency penalties

### Debug Steps

1. Test with minimal feature set first
2. Check feature distributions and correlations
3. Validate environment reset and step functions
4. Monitor training metrics and early stopping

## Contributing

When adding new features:
1. Add to appropriate category in feature configuration
2. Update documentation
3. Test with existing feature sets
4. Validate performance impact
5. Update configuration schemas
