# Codex Prompt: trend_5s veto counterfactual analysis (695# Task 1)

## Goal
Analyze whether the `trend_5s_sell_guard_veto` is providing value by comparing counterfactual outcomes of veto'd trades vs. passed trades in the same time window.

## Context
- **694# finding**: trend_5s_sell_guard_veto = 14% of cancels on 4/2 (was 0.8% on 4/1).
- **Config**: enabled=true, threshold_bps=0.5, hard_veto_threshold_bps=3.0
- **Question**: Are veto'd sell orders avoiding genuine AS risk, or is 0.5 bps too aggressive?

## Implementation

### New file: `scripts/v460/analysis/sections/section_trend_5s_counterfactual.py`

```python
"""695# Trend-5s veto counterfactual analysis."""
```

Analyze from fill_records:
1. **Veto'd group**: records where `cancel_reason == "trend_5s_sell_guard_veto"`
   - Extract `mid_at_order`, `timestamp` → compute mid price 30s/60s/120s later from surrounding fill records
   - Estimate counterfactual PnL: if order had been placed at `order_price`, what would 30s PnL be?
   
2. **Control group**: sell fills in same time window where trend_5s was triggered but below veto threshold
   - These have `trend_5s_guard_action == "boost"` and `filled == True`
   - Actual PnL comparison
   
3. **Output metrics**:
   - Veto'd counterfactual PnL distribution (mean, p10, p90)
   - Control actual PnL distribution
   - AS rate comparison
   - "Value of veto" = control_pnl - counterfactual_pnl
   - Net impact: veto benefit - opportunity cost (lost fills × avg positive PnL)

### Integration with Protocol 688 framework

Add as `section_trend_5s_counterfactual` in protocol registry. Called via:
```bash
python scripts/v460/analysis/run_protocol.py --protocol 695_trend5s --days 1
```

### Test file: `tests/unit/v460/test_695_trend5s_counterfactual.py`

1. `test_counterfactual_pnl_computation` — mock records with known mid_at_order and future mids
2. `test_veto_group_filtering` — correctly identifies veto'd records
3. `test_control_group_filtering` — correctly identifies boost (non-veto) records
4. `test_empty_veto_group` — graceful handling when no veto records exist
5. `test_net_impact_calculation` — benefit vs opportunity cost math

## Constraints
- Use only data from fill_records (no external data sources)
- Mid price interpolation from nearby fill records is acceptable (not exact)
- Output JSON compatible with analysis_results/ pattern
- Run: `python -m pytest tests/unit/v460/test_695_trend5s_counterfactual.py -x --tb=short -q`
