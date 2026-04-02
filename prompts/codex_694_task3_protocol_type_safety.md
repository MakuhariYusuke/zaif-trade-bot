# Codex Prompt: Protocol 688 type safety and threshold config化 (694# Task 3)

## Goal
Fix type safety issues in `protocol_688.py` and move hard-coded thresholds to configuration.

## Context
- **Codex review grade**: B+ — `.get()` patterns return `Any`, hard-coded thresholds (e.g., AS rate levels) reduce maintainability.
- **run_protocol.py**: Basic error handling insufficient.

## Implementation

### 1. Edit: `scripts/v460/analysis/protocols/protocol_688.py`

**Type safety fixes**:
- All `.get(key, default)` calls must have explicit type casts: `float(data.get("key", 0.0))`, `int(data.get("key", 0))`, `str(data.get("key", ""))`.
- Add type annotations to all section functions: `def section_basic(records: list[dict[str, object]]) -> dict[str, object]:`
- Replace `dict` with `dict[str, object]` where appropriate (no bare `dict`).

**Threshold config化**:
- Move hard-coded thresholds (AS rate levels, spread bucket boundaries, PnL thresholds) to a `Protocol688Config` dataclass at top of file:
```python
@dataclass(frozen=True)
class Protocol688Config:
    as_rate_warn_threshold: float = 0.25
    as_rate_alert_threshold: float = 0.35
    pnl_warn_threshold_bps: float = -0.5
    spread_bucket_edges: tuple[float, ...] = (1500.0, 2500.0, 3500.0)
    min_section_samples: int = 5
```
- Pass config to section functions as parameter (or use module-level default).
- Existing hard-coded values become the defaults (no behavior change).

### 2. Edit: `scripts/v460/analysis/run_protocol.py`

**Error handling improvements**:
- Wrap `protocol.run()` in try/except with specific exception types (not bare `except`).
- Add `--output` flag for output file path (currently prints to stdout).
- Validate `--days` argument (must be > 0).
- Validate `--protocol` argument (must be in PROTOCOL_REGISTRY).

### 3. New/Edit test: `tests/unit/v460/test_694_protocol_688_type_safety.py`

Test cases:
1. `test_protocol_688_config_defaults` — default values match current hard-coded values
2. `test_section_basic_type_annotations` — verify return type is `dict[str, object]`
3. `test_section_with_empty_records` — empty list → graceful handling (no KeyError/ZeroDivisionError)
4. `test_section_with_missing_fields` — records with missing keys → defaults used
5. `test_run_protocol_invalid_protocol_id` — error message for invalid protocol
6. `test_run_protocol_invalid_days` — error for days <= 0

## Constraints
- No behavior change — only type safety and configurability improvements
- Keep backward compatibility with existing protocol calls
- No Any types in new code
- Run: `python -m pytest tests/unit/v460/test_694_protocol_688_type_safety.py -x --tb=short -q`
