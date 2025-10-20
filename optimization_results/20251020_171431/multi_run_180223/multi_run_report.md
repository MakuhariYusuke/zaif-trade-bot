# SAC v428 Multiple Optimization Runs Report

**Generated:** 2025-10-20 18:11:28
**Total Runs:** 3
**Trials per Run:** 10

## Best Overall Results
- **Best Score:** 1.0000
- **From Run:** 1

## Score Statistics Across Runs
- **Mean:** 1.0000
- **Standard Deviation:** 0.0000
- **Min:** 1.0000
- **Max:** 1.0000
- **Median:** 1.0000

## Parameter Consistency Analysis

| Parameter | Consistency | CV | Value Range |
|-----------|-------------|----|-------------|
| ultra_risk_multiplier | low | .3f | .3f |
| risk_weight | low | .3f | .3f |
| learning_rate | low | .3f | .3f |
| balance_penalty_tolerance | low | .3f | .3f |
| trading_bonus_multiplier | low | .3f | .3f |
| profit_weight | low | .3f | .3f |
| drawdown_penalty_weight | low | .3f | .3f |
| ultra_profit_multiplier | low | .3f | .3f |
| reward_scale | medium | .3f | .3f |
| batch_size | low | .3f | .3f |
| buffer_size | low | .3f | .3f |
| stagnation_penalty_weight | low | .3f | .3f |
| growth_bonus_weight | low | .3f | .3f |
| position_penalty_weight | low | .3f | .3f |
| balance_penalty | medium | .3f | .3f |
| trading_bonus | low | .3f | .3f |
| use_simple_reward | low | .3f | .3f |
| ent_coef | low | .3f | .3f |
| win_streak_bonus_weight | low | .3f | .3f |
| hold_penalty_rate | low | .3f | .3f |
| reward_clip_max | low | .3f | .3f |
| consistency_weight | medium | .3f | .3f |
| gamma | high | .3f | .3f |
| reward_clip_min | low | .3f | .3f |
| tau | low | .3f | .3f |

## Individual Run Results

| Run | Best Score | Key Parameters |
|-----|------------|----------------|
| 1 | 1.0000 | .3f, 256, 50000, ... |
| 2 | 1.0000 | .3f, 512, 200000, ... |
| 3 | 1.0000 | .3f, 64, 100000, ... |

## Recommendations

- **High Consistency:** Results are very consistent across runs. The optimization has likely converged to a stable solution.
- **Stable Parameters:** gamma show high consistency and are likely well-optimized.
- **Variable Parameters:** ultra_risk_multiplier, risk_weight, learning_rate, balance_penalty_tolerance, trading_bonus_multiplier, profit_weight, drawdown_penalty_weight, ultra_profit_multiplier, batch_size, buffer_size, stagnation_penalty_weight, growth_bonus_weight, position_penalty_weight, trading_bonus, use_simple_reward, ent_coef, win_streak_bonus_weight, hold_penalty_rate, reward_clip_max, reward_clip_min, tau show high variation. Consider focusing optimization efforts on these parameters.
