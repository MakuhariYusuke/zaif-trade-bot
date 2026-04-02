# Codex Prompt: Ranging regime AS investigation & adaptive guard (695# Task 3)

## Goal
Investigate why the ranging regime AS rate jumped from 36.5% (4/1) to 56.5% (4/2), and implement regime-conditional guard parameters.

## Context
- **694# finding**:
  - Ranging regime: 4/1 → 36.5% AS, PnL +0.600 bps; 4/2 → 56.5% AS, PnL -0.362 bps
  - Trending regime: AS stable (~40%), PnL improved
  - Volatile regime: too few samples for conclusion
- **Hypothesis**: Skip gate bypass (686#) removed ML-based cancellation that disproportionately protected ranging. The skip_gate ML model may have been effective in ranging despite low overall MI.

## Implementation

### Part A: Analysis script

**New file**: `scripts/v460/analysis/sections/section_regime_as_deep_dive.py`

1. Cross-tabulate regime × spread bucket × AS rate
2. Compare 4/1 (skip_gate active) vs 4/2 (bypass) for ranging regime specifically
3. Identify which spread buckets within ranging drove the AS spike
4. Check if veto'd trades (trend_5s) overlapped with ranging regime
5. Output: JSON with regime-specific AS attribution

### Part B: Regime-adaptive guard thresholds

**Modify**: Guard configuration to support per-regime overrides

```yaml
# In fill_test.yaml
regime_guard_overrides:
  enabled: false  # observe mode
  ranging:
    ev_threshold_premium_bps: 0.3  # require higher EV in ranging
    spread_as_guard_penalty_multiplier: 1.5
  trending:
    ev_threshold_premium_bps: 0.0  # no additional premium
    spread_as_guard_penalty_multiplier: 1.0
```

**Implementation pattern**:
- Add `RegimeGuardAdapter` that reads current regime from `regime_detector`
- Applies multipliers to existing guard thresholds
- Must compose with existing guards, not replace them

### Test file: `tests/unit/v460/test_695_regime_as_analysis.py`

1. `test_crosstab_regime_spread_as` — correct cross-tabulation with mock data
2. `test_regime_guard_adapter_ranging` — ranging override applied
3. `test_regime_guard_adapter_trending` — trending override applied
4. `test_regime_guard_adapter_disabled` — passthrough when disabled
5. `test_regime_guard_adapter_unknown_regime` — falls back to default
6. `test_multiplier_composition` — correctly stacks with base guard values

## Constraints
- Analysis must produce reproducible results with command-line invocation
- Guard adapter must NOT duplicate existing guard logic — compose only
- Type-safe regime enum, no string comparisons for regime names
- Run: `python -m pytest tests/unit/v460/test_695_regime_as_analysis.py -x --tb=short -q`
