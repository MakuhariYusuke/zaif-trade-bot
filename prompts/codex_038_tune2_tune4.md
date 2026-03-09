# Codex Prompt: 364# TUNE-2 + TUNE-4 Implementation

## Context

- Repo: `zaif-trade-bot` (Python 3.11.9, Windows, `.venv`)
- Branch: `main`, HEAD = `edaab98e0`
- Prior work: 364# TUNE-3 (SDK threshold relaxation) committed. 037-074/075 completed OPS-1/2/4/6.
- **Goal**: K1 (attempted_fill_rate) improvement. Current K1=46.15% (FAIL ≥60%).

## Task C-4: TUNE-2 — `per_side_dd_halt` Threshold Relaxation

### Problem
`per_side_dd_halt` at `-30.0 bps` triggers too aggressively, blocking trade opportunities.
From 360# analysis: 11.3% of records are `per_side_dd_halt` cancels. K1 contribution: +5.6pp.
`per_side_halt_cycles=15` (30 minutes) is also flagged as potentially too long.

### Changes Required

1. **`configs/v460/fill_test.yaml` L686-687**:
   - `per_side_hard_limit_bps: -30.0` → `-50.0`
   - `per_side_halt_cycles: 15` → `10` (30min → 20min)
   - Add comment: `# 364# TUNE-2: -30→-50 (K1寄与+5.6pp), cycles 15→10 (recovery短縮)`

2. **`scripts/v460/lib/fill_config.py` L228-229**:
   - `per_side_dd_hard_limit_bps: float = -30.0` → `-50.0`
   - `per_side_dd_halt_cycles: int = 0` → `10`
   - Keep existing comments, append `364# TUNE-2` reference

3. **`tests/unit/v460/test_168_daily_drawdown_guard.py` L483**:
   - `assert cfg.per_side_dd_hard_limit_bps == -30.0` → `-50.0`

4. **`tests/unit/v460/test_281_deadlock_fix.py` L169** and **`tests/unit/v460/test_215_dd_fix_alert_mode.py` L39**:
   - These use `-30.0` as **constructor arguments** (not YAML assertions). Leave unchanged — they test specific scenarios with explicit values.

5. **`per_side_reanchor_budget_bps`**: Currently `-15.0`. With hard_limit at `-50.0`, consider adjusting to `-25.0` (proportional). Update in YAML (L693) and `fill_config.py` (L235).

### Constraint
- Do NOT change `daily_drawdown.hard_limit_bps` (-50.0) or `soft_limit_bps` (-30.0) — those are global, not per-side.
- Do NOT change `cooldown_release_sec` or `cooldown_release_lot_scale`.

## Task C-5: TUNE-4 — BDK (buy_dynamic_kill) Threshold Consideration

### Problem
From 360# analysis: BDK contributes +9.2pp to K1. However, latest SHA (819ec73b2081) shows **0 BDK cancels**. This means BDK is currently not a bottleneck.

### Decision
- **Do NOT change BDK thresholds** at this time. 0 BDK cancels means the current `-0.8 bps` threshold is adequate.
- Instead, add a comment in `fill_test.yaml` at the BDK section (L621):
  ```yaml
  threshold_bps: -0.8            # 341# revert: ... | 364# TUNE-4 skip: SHA 819ec73b 0件BDK cancel, 変更不要
  ```

## Task C-6: Codex Uncommitted Changes Commit

The working tree has ~35 modified files from Codex 037 sessions (hour_rules.py extraction, test refactoring, etc.) that are NOT yet committed. Please:
1. Run `pytest tests/unit/v460/ -x -q` to verify all tests pass
2. If pass: `git add -A && git commit -m "session037: hour_rules extraction + test refactoring" --no-verify`
3. If fail: fix failures, then commit

## Verification

After all changes:
```bash
pytest tests/unit/v460/test_168_daily_drawdown_guard.py tests/unit/v460/test_169_c1_c3_c4_config.py tests/unit/v460/test_336_fill_config_parser.py tests/unit/v460/test_344_improvements.py -v --tb=short
```
Expect all tests to pass. Then:
```bash
git add -A
git commit -m "feat(364#): TUNE-2 per_side_dd_halt relaxation + TUNE-4 skip note" --no-verify
```

## File Reference Summary

| File | Lines | Change |
|---|---|---|
| `configs/v460/fill_test.yaml` | L621, L686-687, L693 | threshold updates + comments |
| `scripts/v460/lib/fill_config.py` | L228-229, L235 | default value updates |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | L483 | assertion update |

## Priority
- C-6 first (commit existing changes to clean working tree)
- C-4 (TUNE-2) second
- C-5 (TUNE-4 skip note) last
