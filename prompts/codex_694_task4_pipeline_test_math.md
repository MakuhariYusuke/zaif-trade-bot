# Codex Prompt: Offset pipeline test math validation (694# Task 4)

## Goal
Strengthen test coverage for `multiplicative_pipeline.py` stage disable flags and validate pipeline math end-to-end.

## Context
- **Codex review grade**: B- — test math not validated, parameter fragility.
- **Current impl**: 3 stage disable flags (`offset_ev_stage_enabled`, `offset_toxicity_stage_enabled`, `offset_vg_supplement_enabled`) + pipeline stats logging.
- **File**: `scripts/v460/lib/multiplicative_pipeline.py`

## Implementation

### Edit/New: `tests/unit/v460/test_694_pipeline_math_validation.py`

Test cases with explicit math verification:

1. **test_all_stages_enabled_math** — All 3 stages enabled:
   - Input: base_offset=100, ev_mult=1.2, tox_mult=0.8, vg_mult=1.1
   - Expected: 100 × 1.2 × 0.8 × 1.1 = 105.6
   - Assert `pytest.approx(105.6)`

2. **test_ev_stage_disabled** — ev_stage disabled:
   - ev_mult=1.5 should be ignored → 100 × 1.0 × 0.8 × 1.1 = 88.0
   - Assert ev_mult replaced by 1.0

3. **test_toxicity_stage_disabled** — tox_stage disabled:
   - tox_mult=0.5 should be ignored → 100 × 1.2 × 1.0 × 1.1 = 132.0

4. **test_vg_stage_disabled** — vg_stage disabled:
   - vg_mult=2.0 should be ignored → 100 × 1.2 × 0.8 × 1.0 = 96.0

5. **test_all_stages_disabled** — All disabled:
   - All mults ignored → 100 × 1.0 × 1.0 × 1.0 = 100.0

6. **test_pipeline_stats_tracking** — After 100 calls, verify stats contain:
   - `n_calls`: 100
   - `mean_ev_mult`, `mean_tox_mult`, `mean_vg_mult` within `pytest.approx`

7. **test_pipeline_stats_log_interval** — Stats logged every 100 cycles (mock logger)

8. **test_zero_base_offset** — base_offset=0 → output=0 regardless of multipliers

9. **test_negative_mult_clamped** — Negative multiplier handling (if applicable)

10. **test_config_hot_reload_stage_toggle** — Change config mid-run → stage disabled/enabled takes effect

### Also verify: `configs/v460/fill_test.yaml`

Confirm these fields exist and have correct defaults:
```yaml
offset_ev_stage_enabled: true
offset_toxicity_stage_enabled: true
offset_vg_supplement_enabled: true
```

## Constraints
- All expected values must be hand-calculated and documented in test comments
- Use `pytest.approx()` for all float comparisons
- No mocking of pipeline internals — test through public API only
- Run: `python -m pytest tests/unit/v460/test_694_pipeline_math_validation.py -x --tb=short -q`
