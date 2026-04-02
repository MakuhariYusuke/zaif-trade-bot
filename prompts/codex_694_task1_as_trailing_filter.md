# Codex Prompt: AS-aware trailing skip filter (694# Task 1)

## Goal
Add an AS-aware trailing rate filter to `SkipGateEvaluator` that can observe or veto orders based on recent adverse selection rate by regime×spread bucket. This replaces ML model dependence (MI≈0 per 686#) with a deterministic, configurable filter.

## Context
- **686# finding**: SkipGate ML has zero predictive power (MI≈0, |r|<0.16). Q4 scores correlate with worst PnL (-1.24bps). 18.1% cancel rate from skip_gate.
- **Pattern**: Follow `_apply_trend_5s_sell_guard()` at `skip_gate_evaluator.py` L830-839 exactly.
- **Cancel reason pattern**: Follow `cancel_reason_taxonomy.py` REASON_TABLE entries.

## Implementation

### 1. New file: `scripts/v460/lib/as_trailing_tracker.py`

```python
"""694# AS trailing rate tracker by regime×spread bucket."""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field

@dataclass
class ASTrailingConfig:
    """Configuration for AS trailing rate tracker."""
    enabled: bool = False
    window_size: int = 100  # rolling window per bucket
    spread_bucket_edges: tuple[float, ...] = (1500.0, 2500.0, 3500.0)
    soft_threshold: float = 0.30  # AS rate → offset boost
    hard_veto_threshold: float = 0.45  # AS rate → skip
    offset_boost_factor: float = 1.3
    min_samples: int = 10  # minimum fills before acting

@dataclass
class _FillEvent:
    timestamp: float
    is_adverse: bool

class ASTrailingTracker:
    """Track trailing AS rate by regime×spread_bucket."""

    def __init__(self, config: ASTrailingConfig) -> None:
        self._config = config
        # key: (regime, spread_bucket_idx) → deque of _FillEvent
        self._buckets: dict[tuple[str, int], deque[_FillEvent]] = {}

    def _spread_bucket(self, spread: float) -> int:
        for i, edge in enumerate(self._config.spread_bucket_edges):
            if spread < edge:
                return i
        return len(self._config.spread_bucket_edges)

    def record_fill(self, *, regime: str, spread: float, is_adverse: bool) -> None:
        key = (regime, self._spread_bucket(spread))
        bucket = self._buckets.setdefault(key, deque(maxlen=self._config.window_size))
        bucket.append(_FillEvent(timestamp=time.time(), is_adverse=is_adverse))

    def get_as_rate(self, *, regime: str, spread: float) -> tuple[float | None, int]:
        """Return (as_rate_or_None, sample_count) for given regime×spread."""
        key = (regime, self._spread_bucket(spread))
        bucket = self._buckets.get(key)
        if bucket is None or len(bucket) < self._config.min_samples:
            return None, len(bucket) if bucket else 0
        n_adverse = sum(1 for e in bucket if e.is_adverse)
        return n_adverse / len(bucket), len(bucket)

    def evaluate(self, *, regime: str, spread: float, side: str) -> tuple[str, float | None, float | None]:
        """Return (action, offset_mult_or_None, as_rate_or_None).
        
        action: "none" | "boost" | "veto"
        """
        if not self._config.enabled:
            return "none", None, None
        as_rate, n = self.get_as_rate(regime=regime, spread=spread)
        if as_rate is None:
            return "none", None, None
        if as_rate >= self._config.hard_veto_threshold:
            return "veto", None, as_rate
        if as_rate >= self._config.soft_threshold:
            return "boost", self._config.offset_boost_factor, as_rate
        return "none", None, as_rate
```

### 2. Edit: `scripts/v460/lib/fill_config.py`

Add config fields (follow Trend5sSellGuardConfig pattern):

```python
# 694# AS trailing gate configuration.
as_trailing_gate_enabled: bool = False
as_trailing_gate_window_size: int = 100
as_trailing_gate_spread_bucket_edges: str = "1500,2500,3500"
as_trailing_gate_soft_threshold: float = 0.30
as_trailing_gate_hard_veto_threshold: float = 0.45
as_trailing_gate_offset_boost_factor: float = 1.3
as_trailing_gate_min_samples: int = 10
```

Add `@property` for grouped config:
```python
@property
def as_trailing_gate(self) -> ASTrailingConfig:
    edges = tuple(float(x) for x in self.as_trailing_gate_spread_bucket_edges.split(","))
    return ASTrailingConfig(
        enabled=self.as_trailing_gate_enabled,
        window_size=self.as_trailing_gate_window_size,
        spread_bucket_edges=edges,
        soft_threshold=self.as_trailing_gate_soft_threshold,
        hard_veto_threshold=self.as_trailing_gate_hard_veto_threshold,
        offset_boost_factor=self.as_trailing_gate_offset_boost_factor,
        min_samples=self.as_trailing_gate_min_samples,
    )
```

### 3. Edit: `scripts/v460/lib/fill_config_parser.py`

Add `as_trailing_gate_*` parsing in `_parse_skip_gate_section()`:
```python
result["as_trailing_gate_enabled"] = sg_dict.get("as_trailing_gate_enabled", False)
result["as_trailing_gate_window_size"] = int(sg_dict.get("as_trailing_gate_window_size", 100))
result["as_trailing_gate_spread_bucket_edges"] = str(sg_dict.get("as_trailing_gate_spread_bucket_edges", "1500,2500,3500"))
result["as_trailing_gate_soft_threshold"] = float(sg_dict.get("as_trailing_gate_soft_threshold", 0.30))
result["as_trailing_gate_hard_veto_threshold"] = float(sg_dict.get("as_trailing_gate_hard_veto_threshold", 0.45))
result["as_trailing_gate_offset_boost_factor"] = float(sg_dict.get("as_trailing_gate_offset_boost_factor", 1.3))
result["as_trailing_gate_min_samples"] = int(sg_dict.get("as_trailing_gate_min_samples", 10))
```

### 4. Edit: `scripts/v460/lib/skip_gate_evaluator.py`

Add AS trailing gate method (follow `_apply_trend_5s_sell_guard` pattern):

```python
def _apply_as_trailing_gate(
    self,
    *,
    regime: str,
    spread: float,
    side: str,
) -> tuple[str, float | None, float | None]:
    """694# AS trailing rate gate."""
    if self._as_trailing_tracker is None:
        return "none", None, None
    return self._as_trailing_tracker.evaluate(regime=regime, spread=spread, side=side)
```

In `__init__` or initialization, create tracker:
```python
self._as_trailing_tracker: ASTrailingTracker | None = None
if hasattr(self._config, "as_trailing_gate") and self._config.as_trailing_gate.enabled:
    self._as_trailing_tracker = ASTrailingTracker(self._config.as_trailing_gate)
```

In `evaluate()`, add AFTER trend_5s guard block and BEFORE ML model call:
```python
# 694# AS trailing gate
_as_gate_action, _as_gate_mult, _as_gate_rate = self._apply_as_trailing_gate(
    regime=regime, spread=spread_at_order, side=side,
)
result.as_trailing_gate_action = _as_gate_action
result.as_trailing_gate_rate = _as_gate_rate
if _as_gate_action == "veto":
    # ... _set_early_skip_result with reason="rule_as_trailing_gate_veto"
    return result
```

Add `record_fill()` call in post-fill callback (wherever AS determination is made).

### 5. Edit: `scripts/v460/lib/cancel_reason_taxonomy.py`

Add to REASON_TABLE:
```python
"as_trailing_gate_veto": _meta(
    "as_trailing_gate_veto", SkipCategory.GATE_BLOCK, True,
    "694# AS trailing rate gate veto",
),
```

### 6. Add result fields to `SkipGateResult`:

```python
as_trailing_gate_action: str | None = None
as_trailing_gate_rate: float | None = None
as_trailing_gate_offset_mult: float | None = None
```

### 7. Edit: `configs/v460/fill_test.yaml`

Under `skip_gate:` section:
```yaml
  # 694# AS trailing gate (observe mode)
  as_trailing_gate_enabled: false
  as_trailing_gate_window_size: 100
  as_trailing_gate_spread_bucket_edges: "1500,2500,3500"
  as_trailing_gate_soft_threshold: 0.30
  as_trailing_gate_hard_veto_threshold: 0.45
  as_trailing_gate_offset_boost_factor: 1.3
  as_trailing_gate_min_samples: 10
```

### 8. New test: `tests/unit/v460/test_694_as_trailing_tracker.py`

Test cases (follow test_684_trend_5s pattern):
1. `test_no_action_when_disabled` — enabled=False → action="none"
2. `test_no_action_below_min_samples` — 5 fills < min_samples=10 → action="none"
3. `test_boost_at_soft_threshold` — 30% AS → action="boost", mult=1.3
4. `test_veto_at_hard_threshold` — 50% AS → action="veto"
5. `test_no_action_below_threshold` — 20% AS → action="none"
6. `test_spread_bucket_separation` — different spread buckets track independently
7. `test_regime_separation` — different regimes track independently
8. `test_window_eviction` — window_size=10, fill 15 → only last 10 count
9. `test_record_fill_and_get_rate` — verify rate calculation
10. `test_config_yaml_roundtrip` — YAML parse → ASTrailingConfig
11. `test_cancel_reason_in_taxonomy` — "as_trailing_gate_veto" in REASON_TABLE
12. `test_result_fields_populated` — evaluate() populates as_trailing_gate_* fields on SkipGateResult

## Constraints
- Do NOT modify existing ML model path — AS gate is additive (pre-ML rule)
- Keep `enabled: false` in YAML — observe mode first
- Follow existing cancel_reason_taxonomy exactly
- All floats must have explicit type annotations (no Any)
- Use `pytest.approx()` for float comparisons in tests
- Run: `python -m pytest tests/unit/v460/test_694_as_trailing_tracker.py -x --tb=short -q`
