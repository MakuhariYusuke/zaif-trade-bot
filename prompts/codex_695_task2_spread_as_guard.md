# Codex Prompt: Low-spread AS defense gate (695# Task 2)

## Goal
Add a spread-conditional AS defense mechanism to reduce adverse selection in the 0-1500 bps spread bucket.

## Context
- **694# finding**: 0-1500 spread bucket has 64.3% AS rate on 4/2 (14/22 fills). PnL mean = -1.213 bps.
- **Hypothesis**: Low spread = high liquidity = more informed flow. Current gates don't differentiate by spread.
- **Safety concern**: This is the largest fill bucket. High AS here drags overall PnL.

## Implementation

### Modify: `ztb/trading/market_maker/guards/`

1. **New guard**: `SpreadConditionalASGuard`
   - Trigger: when `current_spread_bps < threshold` (configurable, default 1500)
   - Action: apply additional EV penalty or require higher EV threshold to proceed
   - NOT a hard block — modifies `ev_adjustment` in the guard pipeline
   - Config key: `spread_as_guard.enabled`, `spread_as_guard.spread_threshold_bps`, `spread_as_guard.ev_penalty_bps`

2. **Integration**: Register in guard chain after entry_gate, before order submission
   - Must respect existing guard pipeline pattern (see `ztb/trading/market_maker/guards/`)
   - Emit metrics: `spread_as_guard.triggered`, `spread_as_guard.blocked`

3. **Config addition** (`configs/v460/fill_test.yaml`):
```yaml
spread_as_guard:
  enabled: false  # observe mode first
  spread_threshold_bps: 1500
  ev_penalty_bps: 0.5
```

### Test file: `tests/unit/v460/test_695_spread_as_guard.py`

1. `test_guard_triggers_below_threshold` — spread 1000 → guard fires, ev_adjustment applied
2. `test_guard_skips_above_threshold` — spread 2000 → no adjustment
3. `test_guard_disabled` — enabled=false → passthrough
4. `test_ev_penalty_application` — verify penalty subtracted from EV correctly
5. `test_metrics_emission` — triggered/blocked counters increment
6. `test_config_hot_reload` — threshold changes take effect without restart

## Constraints
- Initial deploy: `enabled: false` (observe mode — log would-have-blocked)
- Follow existing guard pattern exactly (SRP, no god objects)
- Type-safe: no Any types
- Run: `python -m pytest tests/unit/v460/test_695_spread_as_guard.py -x --tb=short -q`
