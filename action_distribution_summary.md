# Action Distribution Analysis: Reward Function Comparison

## Overview
This analysis compares the action distributions (HOLD/BUY/SELL ratios) across three reward function variants from short training runs (5k timesteps each).

## Results Summary

### v378_scale_short (Balanced Scaling)
- **HOLD**: 51.0% (most conservative)
- **BUY**: 24.6%
- **SELL**: 24.4%
- **Total Actions**: 512 (final rollout)

### v379_dynamic_short (Dynamic Rewards)
- **HOLD**: 47.1%
- **BUY**: 26.4%
- **SELL**: 26.6%
- **Total Actions**: 512 (final rollout)

### v380_aggressive_short (Aggressive Rewards)
- **HOLD**: 48.2%
- **BUY**: 26.0%
- **SELL**: 25.8%
- **Total Actions**: 512 (final rollout)

## Key Insights

1. **v378 shows most conservative behavior** with highest HOLD ratio (51.0%)
2. **v379 and v380 are more active** with BUY/SELL combined ratios of ~52.9% and ~51.8% respectively
3. **v379 has slight SELL bias** (26.6% vs 26.4% BUY)
4. **v380 is most balanced** among active variants (BUY/SELL nearly equal)

## Recommendation for Risk Analysis

Based on action distributions:
- **v378_scale_short**: Best for conservative strategies (high HOLD ratio)
- **v379_dynamic_short**: Highest trading activity (lowest HOLD)
- **v380_aggressive_short**: Balanced active trading

Proceed to risk metrics analysis (Sharpe ratio, drawdown) using these configurations.

## Data Source
- Extracted from TensorBoard events in short training checkpoints
- Metrics: `pan_action_counts/*` and `pan_action_pct/*` scalars
- Analysis script: `analyze_action_distribution.py`