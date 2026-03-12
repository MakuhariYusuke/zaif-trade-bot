# v450 Curriculum Learning: Action Discovery & Stage Progression

## Overview
This document describes the modifications made to the curriculum learning system for v450.

Key changes:
- Added a new initial stage: `action_discovery`.
- Implemented `_calculate_action_discovery_reward` in `RewardCalculator` to encourage early exploration.
- `BalanceCurriculumManager` now includes `action_discovery` as the first stage in the `STAGE_SEQUENCE`.
 - `BalanceCurriculumManager` now includes `action_discovery` as the first stage in the `STAGE_SEQUENCE`, and it is set as the default starting stage for curriculum learning (unless overridden by `curriculum_stage` in `EnvironmentConfig`).
- Heavy environment (`heavy_env`) now passes `continuous_action_value` into `reward_calculator.calculate_reward`.

## Action Discovery Stage
- Purpose: Encourage the agent to take actions (reduce HOLD bias) during early training.
- Behavior: reward is based on `|continuous_action_value|` and `pnl` sign (proxy for directional correctness), while ignoring transaction costs.
- Tunables: `action_discovery.scale` (default: 1.0) to scale discovery reward.

## Integration Notes
- `RewardCalculator.calculate_reward` now accepts `continuous_action_value` kw. If absent, the formula defaults to discrete behavior.
- `BalanceCurriculumManager` stage sequence starts with `action_discovery` followed by `forced_balance`.
- Emergency mechanisms still apply in case of bias/regression.

## Tests
- `tests/unit/trading/components/test_action_discovery_reward.py`: Validates positive & negative PnL behavior for action_discovery stage.

## Next steps
- Validate live performance in backtests to ensure action_discovery doesn't yield noisy or degenerate behavior.
- Tune `action_discovery.scale` and stage progression conditions (BalanceCurriculumManager) via hyperparameter optimization.
