# SAC v428 Multiple Optimization Runs Report

**Generated:** 2025-10-20 18:27:00
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
| ultra_profit_multiplier | low | .3f | .3f |
| reward_scale | medium | .3f | .3f |
| hold_penalty_rate | low | .3f | .3f |
| balance_penalty | medium | .3f | .3f |
| reward_clip_max | medium | .3f | .3f |
| ent_coef | low | .3f | .3f |
| growth_bonus_weight | low | .3f | .3f |
| trading_bonus | low | .3f | .3f |
| position_penalty_weight | high | .3f | .3f |
| buffer_size | low | .3f | .3f |
| use_simple_reward | undefined | .3f | .3f |
| balance_penalty_tolerance | medium | .3f | .3f |
| tau | high | .3f | .3f |
| stagnation_penalty_weight | low | .3f | .3f |
| consistency_weight | low | .3f | .3f |
| drawdown_penalty_weight | low | .3f | .3f |
| trading_bonus_multiplier | high | .3f | .3f |
| win_streak_bonus_weight | low | .3f | .3f |
| gamma | high | .3f | .3f |
| reward_clip_min | medium | .3f | .3f |
| risk_weight | medium | .3f | .3f |
| ultra_risk_multiplier | low | .3f | .3f |
| batch_size | low | .3f | .3f |
| profit_weight | medium | .3f | .3f |
| learning_rate | low | .3f | .3f |

## Individual Run Results

| Run | Best Score | Key Parameters |
|-----|------------|----------------|
| 1 | -999.0000 | .3f, 128, 200000, ... |
| 2 | -999.0000 | .3f, 256, 50000, ... |

## Recommendations

- **High Consistency:** Results are very consistent across runs. The optimization has likely converged to a stable solution.
- **Stable Parameters:** position_penalty_weight, tau, trading_bonus_multiplier, gamma show high consistency and are likely well-optimized.
- **Variable Parameters:** ultra_profit_multiplier, hold_penalty_rate, ent_coef, growth_bonus_weight, trading_bonus, buffer_size, stagnation_penalty_weight, consistency_weight, drawdown_penalty_weight, win_streak_bonus_weight, ultra_risk_multiplier, batch_size, learning_rate show high variation. Consider focusing optimization efforts on these parameters.
