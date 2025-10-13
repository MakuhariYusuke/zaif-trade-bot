# SAC Trading Model Action Distribution Analysis

## Problem Statement
The SAC (Soft Actor-Critic) trading model was exhibiting BUY bias in action distribution, with excessive BUY actions and insufficient SELL/HOLD actions. The target distribution was 10% HOLD, 45% BUY, 45% SELL for balanced trading.

## Initial Investigation
- **Symptoms**: Action distribution showed ~10% BUY, 76% SELL, 14% HOLD instead of target 10/45/45
- **Root Cause**: The `ultra_profit` reward stage in `reward_calculator.py` implemented balance_penalty, but it was applied globally to all actions rather than per-action basis
- **Episode Reset Issue**: `_action_counts` are reset at the start of each episode, preventing cumulative action tracking needed for effective balance penalty

## Implemented Fixes

### 1. Balance Penalty Per-Action Application
**File**: `ztb/trading/environment/components/reward_calculator.py`
**Method**: `_calculate_ultra_profit_reward`
**Change**: Modified balance_penalty calculation to apply penalty based on the current action's deviation from target ratio, rather than summing penalties for all actions.

**Before**:
```python
for i, ratio in enumerate(action_ratios):
    deviation = abs(ratio - target_ratios[i])
    if deviation > tolerance:
        excess_deviation = deviation - tolerance
        balance_penalty += penalty * excess_deviation
```

**After**:
```python
deviation = abs(action_ratios[action] - target_ratios[action])
if deviation > tolerance:
    excess_deviation = deviation - tolerance
    balance_penalty = penalty * excess_deviation
```

### 2. Configuration Adjustments
**File**: `config/sac_v414_balanced_trading_config.json`
- `balance_penalty_tolerance`: Adjusted from 0.15 to 0.05 for stricter enforcement
- `balance_penalty`: Increased from 1.0 to 2.0 for stronger penalty application
- `total_timesteps`: Reduced to 5000 for quick validation cycles

## Current Status
- **Latest Results**: BUY 10.9%, SELL 85.4%, HOLD 3.7% (still SELL-biased)
- **Remaining Issues**:
  - Balance penalty magnitude may be insufficient compared to profit/loss rewards
  - Potential data bias in BTC/JPY dataset favoring SELL actions
  - ConfigManager may be merging old defaults from unified_trainer components

## Potential Root Causes Under Investigation

### 1. Unified Trainer Configuration Merging
- **Hypothesis**: `ConfigManager` in unified_trainer may be merging old default configurations containing BUY bonuses
- **Evidence**: Previous AI analysis suggested checking `ConfigManager._merge_reward_settings` and `DEFAULT_PPO_CONFIG`
- **Status**: Not yet verified, as current training script bypasses unified_trainer

### 2. Data Bias
- **Hypothesis**: BTC/JPY 2024 data may have characteristics that make SELL actions more profitable
- **Evidence**: Continuous action mean consistently negative (-0.32), indicating model preference for SELL
- **Mitigation**: Consider random_start variations or data shuffling

### 3. Reward Scale Mismatch
- **Hypothesis**: Profit multipliers (3.0x) create rewards much larger than balance penalties
- **Evidence**: Even with penalty=2.0, profit rewards from successful trades dominate
- **Potential Fix**: Reduce profit_multiplier or increase balance_penalty significantly

## Next Steps
1. Investigate ConfigManager default merging in unified_trainer components
2. Analyze BTC/JPY data for inherent SELL bias
3. Consider alternative balance enforcement mechanisms (e.g., action masking, forced diversity)
4. Implement per-episode action count persistence across training sessions

## Files Modified
- `ztb/trading/environment/components/reward_calculator.py`: Balance penalty calculation
- `config/sac_v414_balanced_trading_config.json`: Penalty parameters and training steps

## Test Results History
| Date | BUY % | SELL % | HOLD % | Notes |
|------|-------|--------|--------|-------|
| 2025-10-14 | 10.9 | 85.4 | 3.7 | Per-action penalty, tolerance=0.05, penalty=2.0 |
| 2025-10-14 | 16.9 | 79.6 | 3.5 | Per-action penalty, tolerance=0.15, penalty=1.0 |
| 2025-10-14 | 0.0 | 100.0 | 0.0 | Zero tolerance test |
| 2025-10-14 | 22.0 | 72.6 | 5.3 | Initial global penalty test |
| 2025-10-14 | 7.2 | 88.7 | 4.1 | Balance penalty=0.0, ultra_profit stage |
| 2025-10-14 | 17.3 | 75.5 | 7.2 | Balance penalty=0.0, profit_optimized stage |
| 2025-10-14 | 64.8 | 33.4 | 1.8 | Balance penalty=0.0, default stage (BUY-biased, profitable model)|
| 2025-10-14 | 20.2 | 72.3 | 7.5 | BUY penalty=-0.05, SELL penalty=-0.1, default stage|
| 2025-10-14 | 30.7 | 62.3 | 7.0 | Balance penalty=0.5, profit_optimized stage (balanced model)|
| 2025-10-14 | 20.4 | 71.3 | 8.3 | v415: trading_bonus=3.0, profit_multipliers=[1.2,1.0,0.8], balance_penalty=1.0|
| 2025-10-14 | 32.8 | 55.9 | 11.2 | v416: trading_bonus=4.0, profit_multipliers=[1.5,0.8,0.5], balance_penalty=2.0 (improved balance)|
| 2025-10-14 | 31.2 | 53.1 | 15.8 | v414 fixed: sell_action_penalty=0.0 (SELL bonus removed, balanced)

## Configuration Parameters
```json
{
  "balance_penalty_tolerance": 0.05,
  "balance_penalty": 2.0,
  "profit_multiplier": 3.0,
  "loss_penalty_multiplier": 3.0,
  "hold_penalty_rate": 0.01,
  "trading_bonus_multiplier": 2.0
}
```</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\bug_fixes\SAC_ACTION_DISTRIBUTION_ANALYSIS.md