# V4XX Series Configuration Guide

## Overview

The V4XX series represents the evolution of SAC (Soft Actor-Critic) models for trading, with each version introducing specific improvements and optimizations. This guide helps you understand and create configurations for V4XX models.

## Version History

### V435 Series
- **V435.5**: Basic ensemble approach
- **V435.6**: Majority voting ensemble
- **V435.7**: Enhanced reward functions with multiple variants (7a, 7b, 7c)

## Configuration Structure

### Main Configuration File Structure

```json
{
  "algorithm": "sac",
  "model_name": "sac_v435_7a",
  "version": "4.3.5.7a",
  "training": {
    "data_config": {
      "data_path": "data/btc_jpy_yahoo_real_20251021_featured.csv",
      "validation_split": 0.2,
      "test_split": 0.1
    },
    "total_timesteps": 50000,
    "sac_hyperparameters": {
      "learning_rate": 3e-4,
      "batch_size": 256,
      "buffer_size": 1000000,
      "learning_starts": 1000,
      "tau": 0.005,
      "gamma": 0.99,
      "ent_coef": "auto_1.0",
      "target_entropy": "auto"
    },
    "environment": {
      "initial_balance": 10000,
      "transaction_cost": 0.0,
      "max_position_size": 1.0,
      "random_start": true,
      "enable_correlation_reduction": true,
      "correlation_threshold": 0.85,
      "max_features": 100,
      "feature_adaptation": true,
      "market_regime_detection": true,
      "symmetric_thresholds": true,
      "action_threshold_buy": -0.3333,
      "action_threshold_sell": 0.3333
    },
    "reward_function": {
      "type": "v435_enhanced",
      "base_profit_bonus_atr_coeff": 5.0,
      "base_profit_bonus_portfolio_coeff": 10.0,
      "base_action_penalty": 0.15,
      "loss_penalty_coeff": -1.0,
      "action_frequency_penalty": 0.0001,
      "long_short_asymmetry": true,
      "risk_adjusted_bonus": true,
      "market_regime_penalty": true,
      "symmetric_thresholds": true,
      "description": "Variant description"
    }
  },
  "features": {
    "technical_indicators": [
      "rsi_14", "macd", "macd_signal", "macd_hist",
      "bb_upper", "bb_middle", "bb_lower", "bb_width",
      "stoch_k", "stoch_d", "williams_r",
      "sma_5", "sma_10", "sma_20", "sma_50",
      "ema_5", "ema_10", "ema_20", "ema_50",
      "atr_14", "cci_14", "mfi_14",
      "roc_12", "mom_10", "vwap",
      "price_volume_trend", "volatility_20",
      "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"
    ]
  }
}
```

## Key Parameters

### Reward Function Parameters

| Parameter | Description | Range | Example |
|-----------|-------------|-------|---------|
| `base_profit_bonus_atr_coeff` | ATR-based profit bonus multiplier | 1.0-30.0 | 5.0 |
| `base_profit_bonus_portfolio_coeff` | Portfolio-based profit bonus multiplier | 1.0-50.0 | 10.0 |
| `base_action_penalty` | Base penalty for taking actions | 0.01-1.0 | 0.15 |
| `loss_penalty_coeff` | Multiplier for loss penalties | -5.0-0.0 | -1.0 |
| `action_frequency_penalty` | Penalty for frequent actions | 0.0-0.01 | 0.0001 |
| `long_short_asymmetry` | Different rewards for long/short | true/false | true |
| `risk_adjusted_bonus` | Adjust bonuses by risk metrics | true/false | true |
| `market_regime_penalty` | Penalize based on market regime | true/false | true |
| `symmetric_thresholds` | Use symmetric action thresholds | true/false | true |

### Environment Parameters

| Parameter | Description | Range | Example |
|-----------|-------------|-------|---------|
| `initial_balance` | Starting portfolio balance | 1000-100000 | 10000 |
| `transaction_cost` | Trading fee percentage | 0.0-0.01 | 0.0 |
| `max_position_size` | Maximum position size (0-1) | 0.1-1.0 | 1.0 |
| `action_threshold_buy` | Threshold for buy actions | -1.0-0.0 | -0.3333 |
| `action_threshold_sell` | Threshold for sell actions | 0.0-1.0 | 0.3333 |

## V435.7 Variants

### 7a: Ultra-micro frequency penalty
```json
{
  "action_frequency_penalty": 0.0001,
  "description": "Ultra-micro frequency penalty with symmetric thresholds"
}
```

### 7b: Zero frequency penalty
```json
{
  "action_frequency_penalty": 0.0,
  "description": "Zero frequency penalty with symmetric thresholds"
}
```

### 7c: Enhanced victory bonuses
```json
{
  "base_profit_bonus_atr_coeff": 15.0,
  "base_profit_bonus_portfolio_coeff": 30.0,
  "action_frequency_penalty": 0.001,
  "description": "Enhanced victory bonuses with symmetric thresholds"
}
```

## Creating New Configurations

1. **Copy a base configuration** from existing V435 configs
2. **Modify reward parameters** based on your strategy
3. **Adjust environment settings** for your data
4. **Update feature list** if needed
5. **Test with simple_backtest** scripts

## Best Practices

1. **Start with existing variants** and modify incrementally
2. **Use symmetric thresholds** for balanced trading
3. **Tune frequency penalties** based on your time horizon
4. **Test on historical data** before live deployment
5. **Document your changes** in the description field

## Troubleshooting

### Common Issues

1. **High frequency trading**: Increase `action_frequency_penalty`
2. **Low profitability**: Adjust `base_profit_bonus_*` coefficients
3. **Over-conservative**: Lower action thresholds or penalties
4. **Data compatibility**: Check feature names match your dataset

---

# Unified Training System (v441+)

## Overview

Starting from v441, all V4XX series models use a unified training and analysis system that provides consistent interfaces, automatic configuration conversion, and improved maintainability.

## Key Features

- **Unified Configuration**: Automatic conversion between legacy and unified formats
- **Consistent Training**: Single trainer supports all V4XX versions
- **Unified Analysis**: Common analysis framework for all versions
- **PowerShell Support**: Easy command-line interface for Windows environments

## Quick Start

### Training a Model

```powershell
# Using PowerShell (recommended for Windows)
.\scripts\run_training.ps1 -Action train -Version v435

# Or using Python directly
python ztb/training/v4xx_unified_trainer.py --config config/sac_v435_7a_config.json
```

### Analyzing Results

```powershell
# Analyze backtest results
.\scripts\run_training.ps1 -Action analyze -Version v440

# Or using Python directly
python -c "from ztb.analysis.v4xx_unified_analyzer import analyze_v4xx_results; analyze_v4xx_results('results/v440/backtest_results_v440.json')"
```

### Converting Legacy Configurations

```powershell
# Convert old format to unified format
# Convert legacy configurations
.\scripts\run_training.ps1 -Action convert -Config config/sac_v427_default_config.json
```

## Version Support

| Version | Status | Training Script | Analysis Support |
|---------|--------|----------------|------------------|
| V427 | ✅ Unified | `train_sac_v437.py` | ✅ Full |
| V435 | ✅ Unified | `v435/train_sac_v435_7a.py` | ✅ Full |
| V437 | ✅ Unified | `train_sac_v437.py` | ✅ Full |
| V440 | ✅ Unified | `train_sac_v440.py` | ✅ Full |

## Configuration Conversion

The system automatically detects and converts legacy configurations:

### Legacy Format (v427)
```json
{
  "model_name": "sac_v427_market_adaptive_ensemble",
  "algorithm": "sac",
  "total_timesteps": 10000,
  "sac_hyperparameters": {...},
  "environment": {...}
}
```

### Unified Format (v435+)
```json
{
  "algorithm": "sac",
  "model_name": "sac_v427_converted",
  "version": "4.2.7",
  "training": {
    "total_timesteps": 10000,
    "sac_hyperparameters": {...},
    "environment": {...},
    "data_config": {...}
  }
}
```

## Migration Guide

### From Individual Scripts to Unified System

1. **Update your configuration** to use the unified format, or let the system auto-convert
2. **Replace training calls** with unified trainer:
   ```python
   # Old way
   from train_sac_v427 import train_sac_v427
   train_sac_v427(...)

   # New way
   from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
   trainer = V4XXUnifiedTrainer("config/sac_v427_config.json")
   trainer.train()
   ```

3. **Use unified analysis**:
   ```python
   # Old way - version-specific analysis
   # New way - universal analysis
   from ztb.analysis.v4xx_unified_analyzer import analyze_v4xx_results
   analyze_v4xx_results("results/backtest_results.json", version="427")
   ```

## Architecture Benefits

- **Reduced Code Duplication**: Common components shared across versions
- **Consistent Interfaces**: Same API for training and analysis regardless of version
- **Automatic Validation**: Configuration validation and error checking
- **Better Maintainability**: Changes to core functionality benefit all versions
- **PowerShell Integration**: Native Windows support with easy command interface

## Future Development

- **v441+**: All new versions will use this unified system by default
- **Plugin Architecture**: Easy addition of new algorithms and features
- **Configuration Templates**: Pre-built configurations for common use cases
- **Automated Testing**: Comprehensive test suite for all components
