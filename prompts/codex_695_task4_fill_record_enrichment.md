# Codex Prompt: Fill record observability enrichment (695# Task 4)

## Goal
Enrich fill_records with guard pipeline decision metadata for post-hoc analysis.

## Context
- **694# finding**: entry_gate blocked 5 orders (ev_negative) and bypassed 27 (skip_gate bypassed), but fill_records lack detail on:
  - Which guard fired and why
  - What the EV estimate was at decision time
  - Whether the order would have been profitable (counterfactual)
- **Current schema**: `skip_gate_bypassed`, `entry_gate_action`, `cancel_reason` exist but are incomplete
- **Problem**: Post-hoc analysis requires manual cross-referencing with logs

## Implementation

### Modify: Fill record emission in `ztb/trading/market_maker/`

Add fields to fill_record JSONL output:

```python
# New fields in fill record
"guard_pipeline_result": {
    "entry_gate_ev_bps": float,        # EV estimate at decision time
    "entry_gate_action": str,           # "allow" | "block" | "bypass"
    "entry_gate_reason": str | None,    # "ev_negative" | "staleness" | ...
    "spread_at_decision_bps": float,    # spread when guard ran
    "regime_at_decision": str,          # "ranging" | "trending" | "volatile"
    "trend_5s_value_bps": float | None, # trend value if sell guard checked
    "trend_5s_action": str | None,      # "pass" | "boost" | "veto"
    "skip_gate_score": float | None,    # ML score even in bypass mode
    "skip_gate_action": str,            # "bypass" | "allow" | "block"
}
```

### Implementation rules:
1. **Backward compatible**: new fields are Optional, old consumers won't break
2. **Performance**: guard metadata is already computed — just needs serialization
3. **Schema versioning**: add `schema_version: 2` field to distinguish enriched records
4. **No new computations**: only serialize data already available in the guard pipeline

### Test file: `tests/unit/v460/test_695_fill_record_enrichment.py`

1. `test_enriched_record_has_guard_pipeline` — new field present in output
2. `test_backward_compatible_deserialization` — v1 records still parseable
3. `test_schema_version_field` — version field present and correct
4. `test_optional_fields_none_safe` — None values serialize correctly
5. `test_guard_pipeline_type_safety` — all fields match expected types
6. `test_existing_analysis_scripts_unaffected` — 694# analysis script still works

## Constraints
- Must not increase per-record serialization time by more than 10%
- Schema v1 records must remain readable
- No Any types in the guard_pipeline_result structure
- Run: `python -m pytest tests/unit/v460/test_695_fill_record_enrichment.py -x --tb=short -q`
