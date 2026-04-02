# Codex Prompt: Buy-side cross-venue protection (694# Task 2)

## Goal
Extend `cross_venue_lead_lag.py` and `skip_gate_evaluator.py` to add buy-side protection when the reference venue detects price movement adverse to buy orders. Currently only sell-side veto exists (501#).

## Context
- **501# finding**: cross-venue lead-lag only protects sell-side. Buy-side has no reference venue guard.
- **Existing impl**: `compute_cross_venue_lead_lag_hint()` returns `CrossVenueLeadLagHint` with `adverse_side` field.
- **Current flow**: SkipGateEvaluator checks hint only for sell (if `adverse_side == "sell"` → veto).
- **Expected fix**: Add parallel buy-side check with independent thresholds.

## Implementation

### 1. Edit: `scripts/v460/lib/cross_venue_lead_lag.py`

The existing `compute_cross_venue_lead_lag_hint()` already computes direction and adverse_side for both sides. Verify that `adverse_side = "buy"` is correctly set when reference venue price is dropping rapidly (buy-adverse = falling market = buy at top before drop).

If not already handled, add:
```python
# When reference venue falling fast → buy is adverse
if direction == "down" and abs(spread_bps) > threshold:
    adverse_side = "buy"
```

### 2. Edit: `scripts/v460/lib/fill_config.py`

Add buy-side cross-venue config fields (parallel to existing sell fields):
```python
# 694# Buy-side cross-venue protection.
cross_venue_buy_protect_enabled: bool = False
cross_venue_buy_veto_spread_bps: float = 5.0  # threshold for veto
cross_venue_buy_boost_spread_bps: float = 3.0  # threshold for offset boost
cross_venue_buy_offset_boost_factor: float = 1.3
```

### 3. Edit: `scripts/v460/lib/fill_config_parser.py`

Parse new fields from YAML `cross_venue:` section:
```python
result["cross_venue_buy_protect_enabled"] = cv_dict.get("buy_protect_enabled", False)
result["cross_venue_buy_veto_spread_bps"] = float(cv_dict.get("buy_veto_spread_bps", 5.0))
result["cross_venue_buy_boost_spread_bps"] = float(cv_dict.get("buy_boost_spread_bps", 3.0))
result["cross_venue_buy_offset_boost_factor"] = float(cv_dict.get("buy_offset_boost_factor", 1.3))
```

### 4. Edit: `scripts/v460/lib/skip_gate_evaluator.py`

Find the existing sell-side cross-venue veto block. Add parallel buy-side block:

```python
# 694# Buy-side cross-venue protection
if (
    side == "buy"
    and self._config.cross_venue_buy_protect_enabled
    and cv_hint is not None
    and cv_hint.adverse_side == "buy"
):
    if cv_hint.spread_bps >= self._config.cross_venue_buy_veto_spread_bps:
        # Veto
        logger.info("[dt=%s] [skip_gate] 694# cross-venue buy veto: spread=%.2fbps",
                    decision_trace_id or "n/a", cv_hint.spread_bps)
        early_context = self._build_skip_fill_record_context(...)
        self._set_early_skip_result(
            result,
            context=early_context,
            score=cv_hint.spread_bps,
            reason="rule_cross_venue_buy_veto",
            model_used="rule",
            ...
        )
        return result
    elif cv_hint.spread_bps >= self._config.cross_venue_buy_boost_spread_bps:
        # Offset boost (widen spread)
        result.cross_venue_buy_offset_mult = self._config.cross_venue_buy_offset_boost_factor
```

### 5. Edit: `scripts/v460/lib/cancel_reason_taxonomy.py`

Add to REASON_TABLE:
```python
"cross_venue_buy_veto": _meta(
    "cross_venue_buy_veto", SkipCategory.GATE_BLOCK, True,
    "694# Cross-venue buy-side protection veto",
),
```

### 6. Add result fields to SkipGateResult:

```python
cross_venue_buy_offset_mult: float | None = None
```

### 7. Edit: `configs/v460/fill_test.yaml`

Under `cross_venue:` section:
```yaml
  # 694# Buy-side cross-venue protection (observe mode)
  buy_protect_enabled: false
  buy_veto_spread_bps: 5.0
  buy_boost_spread_bps: 3.0
  buy_offset_boost_factor: 1.3
```

### 8. New test: `tests/unit/v460/test_694_cross_venue_buy_protect.py`

Test cases:
1. `test_buy_veto_when_spread_exceeds_threshold` — spread_bps=6.0 > 5.0 → skipped, reason="rule_cross_venue_buy_veto"
2. `test_buy_boost_when_spread_moderate` — spread_bps=4.0 > 3.0 → not skipped, offset_mult=1.3
3. `test_buy_no_action_below_threshold` — spread_bps=2.0 → no action
4. `test_sell_not_affected_by_buy_config` — side="sell" → buy config ignored
5. `test_disabled_config_no_action` — enabled=False → no action even with high spread
6. `test_no_hint_no_action` — cv_hint=None → no action
7. `test_adverse_side_mismatch` — cv_hint.adverse_side="sell" + side="buy" → no action
8. `test_cancel_reason_in_taxonomy` — "cross_venue_buy_veto" in REASON_TABLE
9. `test_config_yaml_roundtrip` — YAML parse → config fields correct

## Constraints
- Do NOT modify existing sell-side veto logic
- Keep `enabled: false` in YAML — observe mode first
- Independent thresholds from sell-side (no shared config)
- Follow existing cancel_reason_taxonomy exactly
- All floats must have explicit type annotations (no Any)
- Run: `python -m pytest tests/unit/v460/test_694_cross_venue_buy_protect.py -x --tb=short -q`
