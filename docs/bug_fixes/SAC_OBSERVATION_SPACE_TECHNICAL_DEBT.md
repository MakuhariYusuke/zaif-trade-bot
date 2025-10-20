# SAC Observation Space Mismatch - Technical Debt

## Problem Description

SAC (Soft Actor-Critic) models trained in this project expect a 5-dimensional observation space, but the `TradingEnvironment` outputs 21-dimensional observations when `correlation_reduction=True` (default).

## Root Cause

- SAC models were trained with `correlation_reduction=True` in the environment configuration
- However, the actual observation space after correlation reduction results in 5 features
- The `paper_trade.py` script uses the same environment configuration as training, but the correlation reduction logic may not be properly applied or the feature reduction results in different dimensions

## Impact

- Cannot run paper trading evaluation for SAC models using `paper_trade.py`
- SAC models cannot be properly validated in the unified paper trading framework
- Limits the ability to compare PPO and SAC performance on the same evaluation setup

## Current Status

- `paper_trade.py` has been refactored to support both PPO and SAC algorithms
- SAC-specific environment creation (`_create_env_sac`) attempts to disable correlation reduction with `"enable_correlation_reduction": False`
- However, the environment still outputs 21-dimensional observations, indicating the configuration is not taking effect
- Logs show `correlation_reduction: True` even when explicitly set to `False`

## Investigation Notes

- The `enable_correlation_reduction` setting in environment config is not properly disabling correlation reduction
- Environment initialization logs show correlation reduction as enabled despite config settings
- Need to investigate the environment initialization flow and config processing in `HeavyTradingEnv`

## Required Solution

- Fix the correlation reduction configuration handling in `TradingEnvironment`
- Ensure SAC models can use environments with matching observation spaces
- Verify that `correlation_reduction=False` actually results in 21-dimensional observations for SAC compatibility

## Priority

Medium - Affects SAC model evaluation capabilities but does not break existing PPO functionality.

## Related Files

- `ztb/training/scripts/paper_trade.py` - Main paper trading script
- `ztb/trading/environment/heavy_env/core.py` - Trading environment implementation
- `ztb/trading/environment/utils/config.py` - Environment configuration
- Models: `models/sac_v*.zip` - Affected SAC models

## Next Steps

1. Investigate why `enable_correlation_reduction=False` doesn't disable correlation reduction
2. Fix environment config processing for correlation reduction
3. Test SAC paper trading with corrected observation space
4. Validate both PPO and SAC work correctly in unified framework</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\bug_fixes\SAC_OBSERVATION_SPACE_TECHNICAL_DEBT.md
