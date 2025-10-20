# SAC v428 Multiple Optimization Runs Report

**Generated:** 2025-10-20 18:19:21
**Total Runs:** 2
**Trials per Run:** 2

## Best Overall Results
- **Best Score:** -999.0000
- **From Run:** 1

## Score Statistics Across Runs
- **Mean:** -999.0000
- **Standard Deviation:** 0.0000
- **Min:** -999.0000
- **Max:** -999.0000
- **Median:** -999.0000

## Parameter Consistency Analysis

| Parameter | Consistency | CV | Value Range |
|-----------|-------------|----|-------------|
| trading_bonus_multiplier | medium | .3f | .3f |
| gamma | high | .3f | .3f |
| reward_clip_max | medium | .3f | .3f |
| reward_scale | medium | .3f | .3f |
| risk_weight | low | .3f | .3f |
| drawdown_penalty_weight | low | .3f | .3f |
| batch_size | low | .3f | .3f |
| ultra_risk_multiplier | low | .3f | .3f |
| position_penalty_weight | low | .3f | .3f |
| buffer_size | low | .3f | .3f |
| balance_penalty | medium | .3f | .3f |
| profit_weight | medium | .3f | .3f |
| learning_rate | low | .3f | .3f |
| reward_clip_min | low | .3f | .3f |
| ent_coef | low | .3f | .3f |
| ultra_profit_multiplier | high | .3f | .3f |
| use_simple_reward | low | .3f | .3f |
| growth_bonus_weight | low | .3f | .3f |
| stagnation_penalty_weight | low | .3f | .3f |
| consistency_weight | medium | .3f | .3f |
| balance_penalty_tolerance | low | .3f | .3f |
| win_streak_bonus_weight | low | .3f | .3f |
| hold_penalty_rate | low | .3f | .3f |
| tau | medium | .3f | .3f |
| trading_bonus | medium | .3f | .3f |

## Individual Run Results

| Run | Best Score | Key Parameters |
|-----|------------|----------------|
| 1 | -999.0000 | .3f, 512, 50000, ... |
| 2 | -999.0000 | .3f, 128, 500000, ... |

## Recommendations

- **High Consistency:** Results are very consistent across runs. The optimization has likely converged to a stable solution.
- **Stable Parameters:** gamma, ultra_profit_multiplier show high consistency and are likely well-optimized.
- **Variable Parameters:** risk_weight, drawdown_penalty_weight, batch_size, ultra_risk_multiplier, position_penalty_weight, buffer_size, learning_rate, reward_clip_min, ent_coef, use_simple_reward, growth_bonus_weight, stagnation_penalty_weight, balance_penalty_tolerance, win_streak_bonus_weight, hold_penalty_rate show high variation. Consider focusing optimization efforts on these parameters.
