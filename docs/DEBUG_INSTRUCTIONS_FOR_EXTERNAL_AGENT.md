# SAC v444 SELL-Lock Problem - Debug Instructions for External AI Agent

## Problem Statement

After 2000 timesteps of training with SAC (Soft Actor-Critic) on cryptocurrency trading environment, the model shows:
- **SELL action**: 66.6% (stuck/locked)
- **BUY action**: 16.4%
- **HOLD action**: 17.0%

Despite implementing balance penalty and action bonuses, no improvement is observed.

**Critical Logs**:
```
2025-11-06 02:45:11,301 - root - INFO - Step   2000 | Elapsed:   53.9s | SPS:  37.1 | HOLD: 17.0% | BUY: 16.4% | SELL: 66.6% | Rewards: 2000 recorded
2025-11-06 02:45:11,300 - root - WARNING - CONTINUOUS ACTION STATS [Step 2000] - Mean: -1.000, Std: 0.000, Min: -1.000, Max: -1.000, Range: 0.000
```

This shows the model outputs action value of -1.000 (SELL signal) with 100% consistency.

## Hypothesis to Investigate

### Hypothesis 1: Balance Penalty Not Applied Correctly
**File**: `ztb/trading/environment/components/reward_calculator.py`

**Current Implementation** (lines 240-268):
```python
buy_target = 0.4
sell_target = 0.25
hold_target = 0.35

deviation_buy = abs(buy_ratio - buy_target)
deviation_sell = abs(sell_ratio - sell_target)
deviation_hold = abs(hold_ratio - hold_target)

total_deviation = deviation_buy + deviation_sell + deviation_hold
balance_penalty = total_deviation * balance_penalty_scale
```

**Debug Questions**:
1. Is `balance_penalty_scale` correctly loaded from config? (Expected: 200.0)
2. Is `curriculum_stage` actually set to "balanced_penalty"?
3. Is the balance penalty actually being **subtracted** from the final reward?
4. Check if reward clipping (`min_reward: -10.0, max_reward: 10.0`) is preventing penalty from taking effect

**Investigation Steps**:
```
a) Add trace logging at line 268 showing:
   - balance_penalty value before application
   - reward value before penalty subtraction
   - reward value after penalty subtraction
   - percentage change to reward
   
b) Check if balance_penalty is being added to reward or subtracted
   (Current code should show: final_reward -= balance_penalty)
   
c) Verify the ratio calculations (buy_ratio, sell_ratio, hold_ratio)
   are being computed correctly over sliding window
```

### Hypothesis 2: Action Bonuses Overriding Balance Penalty
**File**: `ztb/trading/environment/components/reward/action_penalty.py` (lines 40-70)

**Current Implementation**:
```python
if action == ACTION_SELL:
    penalty = base_action_penalty
    penalty = penalty - sell_action_bonus  # sell_action_bonus = 5.0
    return max(0.0, penalty)
```

**Problem**: If `base_action_penalty` (0.015 or 1.0?) < `sell_action_bonus` (5.0), result is clamped to 0, giving SELL a **bonus** instead of penalty.

**Debug Questions**:
1. What is actual `base_action_penalty` value? (Config says 1.0, but may be overridden)
2. Is SELL getting net positive reward from this calculation?
3. Is this bonus added AFTER balance penalty, negating the balance penalty effect?

**Investigation Steps**:
```
a) Log action_penalty calculation at every step showing:
   - base_action_penalty value
   - action_bonus value
   - final penalty/bonus value
   - whether result was clamped to 0.0
   
b) Track cumulative bonus received by each action type
   (Should show: BUY getting +10.0, SELL getting +5.0)
   
c) Compare if action_bonus > balance_penalty effect
```

### Hypothesis 3: Model Network Convergence Issue
**Files**: 
- `ztb/trading/environment/heavy_env/core.py` (HeavyTradingEnv)
- SAC implementation in stable-baselines3

**Problem**: Model may have converged to local minimum (all SELL) early and can't escape.

**Debug Questions**:
1. Is exploration (entropy) happening? Check entropy coefficient
2. Are action values exploring full range [-1, 1]?
3. Is the policy actually learning, or just stuck?

**Investigation Steps**:
```
a) Monitor policy entropy at each step
   - Should decrease slowly, not stay constant
   - Current: All -1.000 suggests zero entropy
   
b) Check policy network output distributions
   - Mean and std of policy output before action sampling
   - Should show exploration, not deterministic -1.000
   
c) Verify reward signal variance
   - If all rewards are the same, model has no gradient signal
   - Log reward distribution: mean, std, min, max
   
d) Check learning rate and gradient norm
   - Is backprop happening?
   - Are weights updating?
```

### Hypothesis 4: Curriculum Stage Not Being Applied
**Files**:
- `config/sac_v444_3_balanced_penalty_scale_200.json`
- `ztb/config/loader.py`
- `ztb/trading/environment/utils/config.py`

**Problem**: `curriculum_stage: "balanced_penalty"` may not be reaching the reward_calculator.

**Debug Questions**:
1. Is curriculum_stage in training.environment or training.curriculum_learning?
2. Does EnvironmentConfig correctly receive curriculum_stage parameter?
3. Is the config being converted/validated correctly?

**Investigation Steps**:
```
a) Add logging at EnvironmentConfig.__init__:
   print(f"EnvironmentConfig received curriculum_stage: {curriculum_stage}")
   
b) Add logging at RewardCalculator.__init__:
   print(f"RewardCalculator curriculum_stage: {self.config.curriculum_stage}")
   
c) Add logging at balance penalty calculation:
   print(f"Applying balance penalty for stage: {curriculum_stage}")
   if curriculum_stage != "balanced_penalty":
       print(f"WARNING: Expected 'balanced_penalty', got '{curriculum_stage}'")
```

### Hypothesis 5: Reward Calculation Order Issue
**File**: `ztb/trading/environment/components/reward_calculator.py` (lines 290-360)

**Problem**: Penalties and bonuses may be applied in wrong order, canceling each other.

**Current Order** (verify this):
1. Calculate profit/loss bonus
2. Calculate action penalty (with bonuses)
3. Calculate balance penalty
4. Apply all other penalties/bonuses
5. Clamp reward to [-10, 10]

**Debug Questions**:
1. What is final reward composition for a SELL action?
   - Profit bonus: ?
   - Action bonus: +5.0
   - Balance penalty: -(0.5 * 200) = -100.0 (rough)
   - Net: ?
   
2. Is reward clipping hiding the balance penalty effect?
   - If -100.0 gets clamped to -10.0, information is lost

**Investigation Steps**:
```
a) Log reward_breakdown dictionary for each action:
   reward_breakdown = {
       "profit_bonus": X,
       "action_bonus": Y,
       "balance_penalty": Z,
       "other_penalties": W,
       "final_before_clip": X+Y+Z+W,
       "final_after_clip": clipped_value
   }
   
b) Accumulate statistics over 100 steps for each action type
c) Compare if SELL is getting systematically higher rewards
```

### Hypothesis 6: Environment Observation Problem
**File**: `ztb/trading/environment/heavy_env/core.py`

**Problem**: Observation may not contain action history needed for balance penalty calculation.

**Debug Questions**:
1. Does observation include recent action history?
2. Is action history properly tracked across episodes?
3. Is the window (last N actions) being calculated correctly?

**Investigation Steps**:
```
a) Log observation shape and content
b) Log recent_actions counter at balance penalty calculation
c) Verify window size (should be ~10-50 last actions)
```

## Specific Files to Trace

### Priority 1 - High Impact
1. **reward_calculator.py** (lines 220-280)
   - Balance penalty calculation
   - Is it actually being applied?

2. **action_penalty.py** (lines 40-70)
   - Are bonuses correctly canceling penalties?

3. **core.py** (heavy_env)
   - Is environment correctly passing curriculum_stage?
   - Is action history being tracked?

### Priority 2 - Configuration
1. **config/sac_v444_3_balanced_penalty_scale_200.json**
   - Verify curriculum_stage location and value
   - Verify all balance_penalty settings

2. **utils/config.py** (EnvironmentConfig)
   - Is curriculum_stage parameter accepted?
   - Is it stored correctly?

### Priority 3 - SAC Algorithm
1. Stable-baselines3 integration
   - Is entropy exploration working?
   - Are gradients flowing?

## Debugging Output Format Requested

When investigating, please provide:

1. **Configuration Trace**:
   ```
   curriculum_stage loaded as: X
   balance_penalty_scale: Y
   action_bonuses: {buy: A, sell: B, hold: C}
   ```

2. **Step-by-step Reward Calculation** (for one SELL action at step 1000):
   ```
   Step 1000 - Action: SELL
   - profit_bonus: X
   - action_bonus: +5.0
   - balance_penalty: -Y
   - reward_before_clip: X + 5.0 - Y = Z
   - reward_after_clip: W
   ```

3. **Action Distribution Stats** (every 100 steps):
   ```
   Step 100: BUY: 20%, SELL: 60%, HOLD: 20%
   Step 200: BUY: 18%, SELL: 65%, HOLD: 17%
   ...
   (Should show convergence toward 40%, 25%, 35%)
   ```

4. **Penalty Effectiveness**:
   ```
   Actions taken: 2000
   - SELL count: 1332 (66.6%)
   - Average balance_penalty received when SELL chosen: X
   - Average balance_penalty received when BUY chosen: Y
   - Difference: X - Y (should be ~-60 if working)
   ```

## Success Criteria

Debugging is successful if we find:
- ✅ Balance penalty is actually being calculated
- ✅ Balance penalty is actually being subtracted from reward
- ✅ Action bonus is not negating balance penalty
- ✅ Model is receiving different reward signals for different actions
- ✅ curriculum_stage is correctly set to "balanced_penalty"

OR find the missing link preventing all of the above.

## Configuration Used

- **Config File**: `config/sac_v444_3_balanced_penalty_scale_200.json`
- **Training Steps**: 2000
- **Initial Balance**: 200,000 JPY
- **Transaction Cost**: 0.001
- **Balance Penalty Scale**: 200.0
- **Action Bonuses**: BUY=10.0, SELL=5.0, HOLD=2.0
- **Targets**: BUY=0.4, SELL=0.25, HOLD=0.35

## Codebase Context

Repository: https://github.com/MakuhariYusuke/zaif-trade-bot
Branch: main
Language: Python 3.11
Framework: stable-baselines3 (SAC)
Environment: Custom HeavyTradingEnv

Key Classes:
- `RewardCalculator`: Calculates rewards with balance penalty
- `ActionPenaltyCalculator`: Calculates action-specific penalties/bonuses
- `HeavyTradingEnv`: The trading environment

