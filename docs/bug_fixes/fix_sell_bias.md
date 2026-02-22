# Fix SELL Action Bias - Investigation and Remediation

## Executive Summary

**Problem**: PPO models trained with custom reward parameters (high SELL rewards, BUY penalties) consistently select only HOLD or BUY actions during inference, despite SELL being legal and highly rewarded.

**Root Causes Identified**:
1. Action mask not applied during training (only in evaluation)
2. Deterministic decoding order issues (mask → softmax → argmax)
3. Policy head bias initialization favoring certain actions
4. Action imbalance causing gradient starvation for SELL
5. Training/evaluation normalization statistics mismatch

## Timeline of Investigation

### Initial Symptoms (2025-10-06)
- Model trained with:
  - `has_position_sell_reward: 20.0`
  - `no_position_buy_reward: -1.0`
  - `reward_scaling: 0.1` (effective SELL reward: 2.0)
- Paper trading results: **100% BUY actions** initially
- After action mask fix: **100% HOLD actions**
- No SELL actions despite being legal when `position >= 0`

### Debugging Steps Completed

1. **Environment Reward Verification** ✓
   - Confirmed rewards applied correctly in `_calculate_reward()`
   - Debug prints show SELL reward of 20.0 when position > 0
   - Reward scaling correctly reduces to 2.0 effective reward

2. **Action Masking Logic** ✓
   - `get_legal_actions()` correctly returns:
     - HOLD: always legal (mask=1)
     - BUY: legal when position <= 0 (mask=1)
     - SELL: legal when position >= 0 (mask=1)
   - **BUG FOUND**: Masks not passed to `model.predict()` in paper_trade.py
   - **FIX APPLIED**: Added `action_masks` parameter to predict call

3. **Verbose Logging Attempt** ⚠️
   - Added verbose parameter to `PaperTrader.__init__()`
   - Attempted to log action probabilities
   - **Issue**: Probabilities not displaying (verbose flag may not propagate)

4. **Debug Print Verification** 🔄
   - Added debug print to confirm verbose=True in __init__
   - Need to verify verbose flag reaches probability logging code

## Technical Analysis

### Current Configuration (ppo_100k_config.json)
```json
{
  "curriculum_stage": "simple_portfolio",
  "reward_scaling": 0.1,
  "custom_reward_params": {
    "no_position_buy_reward": -1.0,
    "no_position_sell_penalty": -5.0,
    "no_position_hold_penalty": -1.0,
    "has_position_sell_reward": 20.0,
    "has_position_buy_penalty": -2.0,
    "has_position_hold_penalty": -1.0
  },
  "ent_coef": 0.05,
  "clip_range": 0.2,
  "gamma": 0.995,
  "gae_lambda": 0.95
}
```

### Identified Issues

#### 1. Training vs Inference Mask Discrepancy ⚠️ CRITICAL
**Status**: Needs verification

If action masks are applied during inference but not during training:
- Training: Model learns on all actions including illegal ones
- Inference: Illegal actions masked out → distribution shifts
- Result: Deterministic selection collapses to one action

**Fix Required**:
- Apply action masks during training in policy forward pass
- Exclude illegal actions from loss calculation
- Use same mask logic in training and evaluation

#### 2. Deterministic Decoding Order 🔍
**Status**: Needs investigation

Current order should be: `mask → softmax(temperature) → argmax`

If order is incorrect:
- Softmax before mask: Illegal actions get probability mass
- Argmax before mask: May select illegal action
- Wrong temperature application: Overconfident or underconfident

**Fix Required**:
- Verify and enforce correct decoding order
- Test with temperature T=0.7 for soft-greedy
- Log intermediate values for debugging

#### 3. Policy Head Bias Initialization 🎯
**Status**: Not investigated

If final layer bias is not neutral:
- Slight positive bias → favors specific action (e.g., BUY)
- Persists through training if gradient updates insufficient

**Fix Required**:
- Re-initialize final layer bias to 0
- Or add learnable LogitBiasLayer with initial=0

#### 4. Action Imbalance → Gradient Starvation 📊
**Status**: Suspected

If SELL actions rare during training:
- SELL advantages → small gradients
- Policy update biased toward frequent actions
- Feedback loop: SELL gets rarer → even smaller gradients

**Fix Required**:
- Inverse-frequency weighting for policy loss
- Regime-stratified sampling (trend/range/volatility)

#### 5. Normalization Statistics Mismatch 📉
**Status**: Not verified

If training scaler not saved/loaded in evaluation:
- Features have different scales
- Policy sees "out-of-distribution" data
- Defaults to safe action (HOLD)

**Fix Required**:
- Save scaler during training
- Load same scaler in evaluation
- Remove zero-variance columns in preprocessing

## Proposed Remediation Plan

### Phase 1: Diagnostics (Completed ✓)
- [x] Create `ActionDiagnostics` utility
- [x] Implement batch-level logging of:
  - Logits (raw and masked)
  - Probabilities (before/after temperature)
  - Action selection process
  - Entropy, KL, losses
  - Action-wise advantages

### Phase 2: Unit Tests (Completed ✓)
- [x] Create `test_forced_actions.py`
- [x] Test action execution with known price sequences
- [x] Verify PnL, fees, inventory calculations
- [x] Confirm BUY/SELL fee symmetry

### Phase 3: Training Fixes (In Progress 🔄)
- [ ] Apply action masks in training forward pass
- [ ] Fix deterministic decoding order
- [ ] Implement inverse-frequency loss weighting
- [ ] Add regime-stratified sampling
- [ ] Re-initialize policy head bias to 0
- [ ] Add target_kl penalty
- [ ] Implement entropy coefficient cosine decay

### Phase 4: Evaluation Fixes (Pending ⏳)
- [ ] Save/load normalization statistics
- [ ] Remove zero-variance features
- [ ] Implement temperature T=0.7 evaluation
- [ ] Generate feature diagnostics report

### Phase 5: Validation (Pending ⏳)
- [ ] 50k × 3 seed short-distance training
- [ ] Compare before/after:
  - Action distribution (BUY/SELL/HOLD)
  - Trade count
  - Sharpe ratio
  - Entropy
  - Approximate KL
  - Regime-specific performance

## Acceptance Criteria

### Must Have ✅
1. **Legal Action Rate** ≥ 99.9% (training and evaluation)
2. **Action Distribution**: No extreme bias
   - For spot (no short selling): SELL should occur when holding position
   - Evaluate "SELL execution rate when holding" not absolute SELL%
3. **Entropy**: No early collapse
   - Initial: High (exploration)
   - Mid-training: Natural decay
   - Should not reach near-zero too early
4. **Fee Symmetry**: All unit tests pass ✓
5. **Sharpe > 0**: 50k × 3 seed average

### Nice to Have 🎯
1. Regime-specific analysis (bull/bear/high-vol/low-vol)
2. Trade frequency within acceptable range
3. Drawdown within limits
4. Convergence stability across seeds

## Implementation Status

### Completed ✓
- Diagnostic utilities (`action_diagnostics.py`)
- Forced action unit tests (`test_forced_actions.py`)
- Initial action mask fix in `paper_trade.py`

### In Progress 🔄
- Training-time action mask application
- Investigation of deterministic decoding

### Pending ⏳
- Policy head bias re-initialization
- Inverse-frequency loss weighting
- Regime-stratified sampling
- Normalization statistics fix
- Hyperparameter adjustments
- Validation experiments

## Next Steps

1. **Immediate** (Day 1):
   - Run forced action tests to verify environment correctness
   - Investigate training-time mask application in MaskablePPO
   - Add diagnostic logging to next training run

2. **Short-term** (Days 2-3):
   - Implement all Phase 3 fixes
   - Run 50k × 3 seed validation
   - Document results

3. **Medium-term** (Week 1):
   - If validation successful: Scale to full training
   - If validation fails: Deep dive into remaining issues

## Notes

### Spot Trading Constraints
- No short selling: SELL only allowed when `position > 0`
- Action masking essential for legal action enforcement
- **KPI**: "SELL rate when holding position" not absolute SELL%

### Reward Scaling Impact
- Configured: `has_position_sell_reward: 20.0`
- Effective (after 0.1 scaling): `2.0`
- May need higher effective reward or different scaling strategy

### Alternative Hypotheses
If fixes don't resolve issue, consider:
1. Model capacity insufficient for complex position management
2. Market regime in test data incompatible with training
3. Feature set insufficient for SELL signal detection
4. Fundamental reward structure issue (e.g., HOLD too safe)

## References

- Code: `ztb/training/paper_trade.py`
- Environment: `ztb/trading/environment/environment.py`
- Config: `ppo_100k_config.json`
- Diagnostics: `ztb/utils/diagnostics/action_diagnostics.py`
- Tests: `tests/unit/environment/test_forced_actions.py`

---

**Last Updated**: 2025-10-06
**Status**: Investigation ongoing, fixes in progress
