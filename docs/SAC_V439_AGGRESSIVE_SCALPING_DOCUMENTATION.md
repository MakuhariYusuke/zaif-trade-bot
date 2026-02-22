# SAC v439 Development - Aggressive Scalping Model

## Overview

SAC v439 is an enhanced version of the SAC trading model specifically optimized for scalping trading with high-frequency execution. This version addresses the critical issue of zero trades by implementing aggressive trading parameters and action signal guidance.

## Key Problems Addressed

### 1. Zero Trade Issue
- **Problem**: Models were generating zero trades due to high action thresholds and conservative settings
- **Solution**: Reduced action threshold from 0.1 to 0.01, min_trade_size from 0.001 to 0.00001

### 2. Conservative Trading Parameters
- **Problem**: Settings were too conservative for active trading
- **Solution**: Implemented scalping-optimized parameters for high-frequency trading

### 3. Missing Action Signal Guidance
- **Problem**: Action signal guide was not integrated into training environment
- **Solution**: Integrated ActionSignalGuide into reward calculation with strong guidance level and explicit RSI/MACD weighting

## Technical Improvements

### Environment Optimizations
- **Action Threshold**: Reduced to 0.01 (from 0.1) for more responsive trading
- **Minimum Trade Size**: Reduced to 0.00001 (from 0.001) for smaller position changes
- **Reward Scaling**: Optimized for scalping frequency
- **Commission/Slippage**: Maintained realistic trading costs

### Action Signal Guidance Integration
- **Guidance Level**: Strong (prioritises high-confidence signals)
- **Signal Strength Threshold**: 0.2 for timely entries
- **Reward Integration**: Signal-aligned actions receive up to 2.5x bonus
- **Penalty Mitigation**: Action penalties scaled down by signal confidence

### Feature Set Optimization
- **Dimensions**: Reduced to 50 features (from 150+) for faster processing
- **Multi-timeframe Exclusion**: Removed multi-timeframe features as requested
- **Technical Focus**: RSI, MACD, Bollinger Bands, Stochastic, ATR, CCI
- **Scalping Indicators**: Short-term momentum and volatility measures

## Configuration Structure

```
config/v439/
├── sac_v439_scalping_config.json    # Main configuration
├── results/v439/                    # Training results
├── models/v439/                     # Saved models
├── checkpoints/v439/                # Training checkpoints
└── tensorboard/v439/                # Training logs
```

## Training Script

### `train_sac_v439_scalping.py`
- Optimized for scalping with reduced thresholds
- Integrated action signal guidance
- Comprehensive logging and checkpointing
- Metadata saving for reproducibility

### Key Parameters
```json
{
  "scalping_optimization": {
    "action_threshold": 0.01,
    "negative_action_threshold": -0.01,
    "min_position_change": 0.00001,
    "max_trades_per_episode": 600,
    "trading_frequency_bonus": 0.2
  },
  "signal_guidance": {
    "enabled": true,
    "guidance_level": "strong",
    "signal_strength_threshold": 0.2,
    "reward_bonus_multiplier": 2.5,
    "action_penalty_multiplier": 0.35
  }
}
```

## Reward Function Enhancements

### Scalping Mode
- **Reduced Action Penalties**: Base trade penalty lowered to 0.01
- **Frequency Bonuses**: Target action rate of 0.55 with inactivity penalties
- **Signal Integration**: Live signal strength logged every step
- **Hold Penalty Multiplier**: 1.2x penalty on HOLD to discourage stagnation

### Action Signal Guidance
- **Signal Strength Evaluation**: Technical signals guide action decisions
- **Reward Modification**: Signal-aligned actions receive multipliers
- **Penalty Mitigation**: Reduced penalties for guided actions
- **Learning Acceleration**: Faster convergence through guided exploration

## Expected Performance Improvements

### Trading Frequency
- **Target**: >500 trades per 10,000 timesteps (>50% action rate)
- **Previous**: 0-10 trades per episode
- **Improvement**: 5-20x increase in trading activity

### Win Rate Optimization
- **Signal-Guided Decisions**: Higher probability of profitable trades
- **Reduced Random Actions**: Action guidance improves decision quality
- **Risk Management**: Maintains position sizing and stop-loss logic

### Training Stability
- **Faster Convergence**: Action guidance accelerates learning
- **Reduced Exploration Time**: Guided actions reduce random exploration needs
- **Better Sample Efficiency**: Higher quality training samples

## Usage Examples

### Training
```bash
# Train v439 scalping model
python scripts/training/train_sac_v439_scalping.py \
  --config config/v439/sac_v439_scalping_config.json \
  --timesteps 100000

# Train with custom data
python scripts/training/train_sac_v439_scalping.py \
  --data data/btc_jpy_featured_dataset.csv \
  --output models/v439_custom
```

### Backtesting
```bash
# Backtest trained model
python scripts/training/backtest_sac_v439.py \
  --model models/v439/sac_v439_scalping_final.zip \
  --episodes 10
```

## Validation Metrics

### Success Criteria
- **Minimum Trades**: >50 trades per episode
- **Trading Frequency**: >10% of steps involve position changes
- **Signal Alignment**: >60% of actions align with technical signals
- **Profitability**: Positive total return in backtesting

### Monitoring
- **Trade Count**: Track trades per episode
- **Action Distribution**: Monitor BUY/SELL/HOLD ratios
- **Signal Strength**: Average signal strength per action
- **Reward Components**: Breakdown of reward sources

## Future Enhancements

### Phase 2: Advanced Scalping
- **Micro-position Sizing**: Even smaller position increments
- **High-Frequency Indicators**: Tick-level momentum measures
- **Real-time Signal Adaptation**: Dynamic signal weight adjustment

### Phase 3: Ensemble Integration
- **Multi-Model Consensus**: Combine multiple scalping strategies
- **Market Regime Adaptation**: Adjust parameters based on market conditions
- **Risk-Adjusted Position Sizing**: Dynamic position sizing based on volatility

## Troubleshooting

### Still Zero Trades
1. **Check Action Threshold**: Ensure < 0.05
2. **Verify Signal Guidance**: Confirm enabled in config
3. **Reduce Min Trade Size**: Try 0.00001
4. **Increase Reward Bonuses**: Boost trading frequency incentives

### Poor Signal Alignment
1. **Adjust Guidance Level**: Try "conservative" or "aggressive"
2. **Modify Signal Threshold**: Lower to 0.2 for more signals
3. **Feature Engineering**: Add more relevant technical indicators

### Training Instability
1. **Reduce Learning Rate**: Try 1e-4 for stability
2. **Increase Batch Size**: 512 or higher for better gradients
3. **Add Gradient Clipping**: Prevent exploding gradients

## Version History

- **v439.0**: Initial scalping implementation with action signal guidance
- **v439.1**: Enhanced reward function and reduced trading barriers
- **v439.2**: Multi-timeframe exclusion and feature optimization
- **v439.3**: Signal guidance disabled test - confirmed HOLD bias is fundamental

## Known Issues and Limitations

### Critical Failure in v439.2
**Training Results (10000 timesteps):**
- Average trades per episode: 1.0 (Target: >50)
- Win rate: 0% (Target: >50%)
- Total return: -497.79% (Target: Positive)
- Sharpe ratio: -∞

**Root Causes Identified:**
1. **Action Threshold Too High**: Even at 0.01, model defaults to HOLD
2. **Signal Guidance Ineffective**: RSI/MACD signals not triggering trades
3. **Reward Function Mismatch**: HOLD penalty multiplier not encouraging activity
4. **Environment Configuration**: Min position change may be too restrictive

### Signal Guidance Disabled Test (v439.3)
**Configuration Changes:**
- Signal guidance: enabled → disabled
- Reward function: signal_guidance_integration remains true

**Training Results (10000 timesteps):**
- Average trades per episode: 1.0 (No improvement)
- Win rate: 0% (No improvement)
- Total return: -498.29% (Slightly worse)
- Sharpe ratio: -∞

**Findings:**
- Signal guidance removal had no positive effect
- HOLD bias persists even without signal constraints
- Problem is deeper in reward function design
- Current approach fundamentally incompatible with scalping goals

**Lessons Learned:**
- Aggressive scalping requires even lower thresholds (<0.005)
- Signal strength calculation may need different indicators
- HOLD penalties need to be much stronger for high-frequency trading
- 10000 timesteps insufficient for complex reward functions
- Signal guidance is not the root cause of HOLD bias

## Future Directions

Given the fundamental issues with the current implementation, future development should:
1. Focus on simpler, more direct reward functions
2. Implement stronger trading incentives
3. Consider alternative signal guidance approaches
4. Accept limitations and pivot to more achievable goals
