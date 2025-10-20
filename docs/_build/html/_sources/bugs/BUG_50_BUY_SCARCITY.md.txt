# Bug #50: BUY Scarcity Limits SELL Opportunities

**Status**: 🔍 IDENTIFIED → 🔧 FIX IN PROGRESS
**Severity**: 🔴 CRITICAL (Root cause of persistent SELL avoidance)
**Discovered**: 2025-10-08
**Fixed In**: v3.6.5

---

## Summary

v3.6.4 validation revealed **BUY rate of only 8.3%**, which structurally limits SELL opportunities since the agent can only SELL when holding a position (which requires prior BUY). This is the **root cause** of why SELL rate plateaued at 12.2% despite Bug #48 and #49 fixes.

---

## Problem Description

### Observed Behavior (v3.6.4)

From console log analysis (`analyze_v364_console.py`):

```
AGGREGATE STATISTICS:
  Total Actions: 96
  HOLD:  76 ( 79.2%)
  BUY:    8 (  8.3%)
  SELL:  12 ( 12.5%)

SELL=12.5% is 1.5x BUY rate
→ Position turnover is already HIGH
→ To increase SELL further, must increase BUY first
```

**Key Finding**: SELL rate cannot sustainably exceed BUY rate. With BUY at 8.3%, the theoretical maximum SELL rate is ~8-10% (accounting for position holding time).

### Root Cause Chain

1. **HOLD dominance (79.2%)**: Agent is overly conservative
2. **BUY scarcity (8.3%)**: Few positions opened
3. **SELL structural limit (12.5%)**: Cannot sell without position
4. **Lambda ineffective**: λ=30.0 (maximum) cannot overcome reward imbalance

**Configuration Issue**:
```json
"profit_bonus_multipliers": [1.0, 3.0, 1.0]  // [BUY, SELL, HOLD]
```
- BUY=1.0: No incentive to open positions
- SELL=3.0: Strong incentive but no opportunities
- HOLD=1.0: Neutral, becomes default action

---

## Impact

### Symptom Cascade

| Version | BUY%  | SELL% | HOLD% | Issue |
|---------|-------|-------|-------|-------|
| v3.6.1  | ?     | 10.7% | ?     | Bug #49 (wrong array order) |
| v3.6.3  | ?     | 8.8%  | ?     | Bug #48 (reward_settings not passed) |
| v3.6.4  | 8.3%  | 12.2% | 79.2% | **Bug #50 (BUY scarcity)** |

Even with correct reward propagation and array order, the **reward structure itself** doesn't incentivize BUY actions enough.

### Why Previous Fixes Failed

- **Bug #48 fix**: Ensured reward_settings reach environment, but multipliers still imbalanced
- **Bug #49 fix**: Applied SELL 3x bonus correctly, but without BUY opportunities, SELL cannot increase
- **Lambda=30.0**: Maximum constraint strength cannot create BUY opportunities

---

## Fix Implementation

### Strategy: Balanced Multiplier Approach

**Configuration Change** (v3.6.5):
```json
"profit_bonus_multipliers": [2.0, 3.0, 0.5]  // [BUY, SELL, HOLD]
```

**Rationale**:
1. **BUY 1.0 → 2.0**: Create position-opening incentive
2. **SELL 3.0** (unchanged): Maintain strong closing incentive
3. **HOLD 1.0 → 0.5**: Discourage passive holding

**Expected Outcome**:
- BUY: 8.3% → 20-25%
- SELL: 12.2% → 15-20%
- HOLD: 79.2% → 55-60%

### Alternative Strategies Considered

#### Option A: Higher Lambda Max
```json
"lagrange_lambda_max": 50.0  // was 30.0
```
**Rejected**: Risks numerical instability, doesn't address BUY scarcity

#### Option B: Action Penalties
```json
"hold_action_penalty": 0.1,
"buy_action_penalty": -0.05,
"sell_action_penalty": -0.1
```
**Deferred**: Requires reward_calculator.py modifications, not backward compatible

#### Option C: Curriculum Learning
Start with SELL target 10%, gradually increase to 33%
**Deferred**: Adds training complexity, try simpler fix first

---

## Validation Plan

### Success Criteria (v3.6.5)

1. **BUY Rate**: ≥ 20%
2. **SELL Rate**: ≥ 15% (previously 12.2%)
3. **HOLD Rate**: ≤ 65% (previously 79.2%)
4. **Distribution Balance**: Max deviation from 33.3% ≤ 20pp

### Validation Command

```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000 --force
```

### Expected Log Output

With enhanced `_final_validation()` method:
```
FINAL ACTION DISTRIBUTION:
  HOLD:  55.0% (176 / 320)
  BUY:   22.0% ( 70 / 320)
  SELL:  18.0% ( 57 / 320)

OUTPUT PATHS:
  Model:        models/ppo_balanced_mem_optimized.zip
  Checkpoints:  checkpoints/ppo_balanced_mem_optimized
  TensorBoard:  tensorboard
```

---

## Related Changes

### Files Modified

1. **configs/training/ppo_balanced_mem_optimized.json**
   - Line 97: `profit_bonus_multipliers: [1.0, 3.0, 1.0] → [2.0, 3.0, 0.5]`
   - Line 3: Version `3.6.4 → 3.6.5`
   - Added Bug #50 to `_bugs_fixed` list

2. **ztb/training/sell_mitigation_ppo_trainer.py**
   - Enhanced `_final_validation()` to show all action distributions
   - Added output path logging

### Analysis Scripts Created

1. **debug_sell_rate_detail.py**: TensorBoard event analysis (found events not readable)
2. **analyze_v364_console.py**: Console log parsing with detailed diagnosis ✅

---

## Lessons Learned

### System Understanding

1. **Action Dependencies**: SELL requires prior BUY → must balance both
2. **Reward Propagation Chain**: Config → reward_settings → multipliers → actual rewards
3. **Constraint Limitations**: Lagrange can only bias existing distribution, not create new actions

### Debugging Methodology

1. **Console log parsing** more reliable than TensorBoard for quick diagnosis
2. **Action ratio analysis** reveals structural constraints (e.g., SELL/BUY < 1.5x)
3. **Aggregate statistics** over iterations provide clearer picture than single snapshots

### Configuration Design

1. **Multipliers must be holistic**: Cannot optimize one action in isolation
2. **Default action emergence**: With equal multipliers, HOLD becomes default (lowest risk)
3. **Incentive hierarchy**: BUY → SELL chain must be rewarded throughout

---

## Next Steps

### If v3.6.5 Succeeds (BUY ≥ 20%, SELL ≥ 15%)

1. Longer training run (30k steps) for stability validation
2. Document final hyperparameter recommendations
3. Consider fine-tuning multipliers (e.g., [2.5, 3.5, 0.5])

### If v3.6.5 Fails (SELL < 15%)

1. **Increase BUY multiplier to 3.0**: `[3.0, 3.0, 0.5]`
2. **Add action penalties** (Option B above)
3. **Investigate action masking**: Check if SELL is being masked too frequently
4. **Reward scaling audit**: Ensure ATR normalization doesn't dilute multiplier effect

### Long-term Improvements

1. **Implement per-action penalties** in reward_calculator.py
2. **Add position-holding-time bonus** to encourage position opening
3. **Curriculum learning** with gradual SELL target increase
4. **Hybrid Lagrange + multiplier** approach with adaptive scaling

---

## Technical Details

### Configuration Diff (v3.6.4 → v3.6.5)

```diff
- "_comment_header": "=== PPO Balanced Memory-Optimized Configuration v3.6.4 ===",
+ "_comment_header": "=== PPO Balanced Memory-Optimized Configuration v3.6.5 ===",
- "_comment_version": "Last updated: 2025-10-08 - Bug #49 fix: SELL 3x with CORRECT array order [BUY, SELL, HOLD]",
+ "_comment_version": "Last updated: 2025-10-08 - Root cause fix: BUY 2x + SELL 3x + HOLD 0.5x to address BUY scarcity",
+ "_comment_bugs_fixed": "Bug #47 (CLI timesteps), Bug #48 (reward_settings propagation), Bug #49 (array order), Bug #50 (BUY scarcity)",
+ "_config_version": "3.6.5",

- "profit_bonus_multipliers": [1.0, 3.0, 1.0],
+ "profit_bonus_multipliers": [2.0, 3.0, 0.5],

- "_bugs_fixed": [..., "Bug #49: profit_bonus_multipliers order"],
+ "_bugs_fixed": [..., "Bug #49: profit_bonus_multipliers order", "Bug #50: BUY scarcity limits SELL"],
- "_validation_status": "⚠️  EXPERIMENTAL - Bug #49 fix: SELL 3x (correct order)"
+ "_validation_status": "⚠️  EXPERIMENTAL - v3.6.5: BUY 2x + SELL 3x + HOLD 0.5x + HOLD penalty 0.05"
```

### Code Changes

**sell_mitigation_ppo_trainer.py** `_final_validation()`:
- Added action distribution estimation from Lagrange stats
- Enhanced logging with HOLD/BUY/SELL percentages
- Added output path display (model, checkpoints, TensorBoard)

---

## Verification Checklist

- [x] Root cause identified (BUY scarcity)
- [x] Analysis script created (analyze_v364_console.py)
- [x] Configuration updated (v3.6.5)
- [x] Log output enhanced (_final_validation)
- [x] Documentation complete (this file)
- [ ] Validation run executed
- [ ] Results analyzed
- [ ] Success criteria met

---

**Related Bugs**: #47 (CLI), #48 (reward_settings), #49 (array order)
**Supersedes**: v3.6.1-v3.6.4 SELL mitigation attempts
**Config**: `configs/training/ppo_balanced_mem_optimized.json`
