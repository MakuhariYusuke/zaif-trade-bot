# Codex Prompt: spread_as_guard 閾値修正 + 有効化準備 (700# Task 2)

## Goal
`spread_as_guard` の単位バグ (threshold=1500.0 bps) を修正し、正しい bps 閾値で安全に有効化できる状態にする。

## Context
- **697# 指摘**: `threshold: 1500.0` は 1500 bps = 15% で、常時発火する無意味な閾値。
- **場所**: `configs/v460/fill_test.yaml:1289-1300`, `scripts/v460/lib/entry_gate_adjustments.py:37-50`
- **現状**: `enabled: false` のため実害なし。修正して有効化可能にする。
- **実スプレッド分布** (4/2 データ): Q1 < 1.8bps, P50 ≈ 2-3 bps 圏

## 現行コード

### entry_gate_adjustments.py (L37-50)
```python
def apply_spread_as_guard(
    ev_score: float,
    spread_bps: float,
    cfg: SpreadAsGuardConfig,
) -> tuple[float, bool]:
    if not cfg.enabled:
        return ev_score, False
    if spread_bps < cfg.threshold:
        # 低スプレッドでは EV を ev_penalty 分だけ減額
        return ev_score - cfg.ev_penalty, True
    return ev_score, False
```

### fill_test.yaml (L1289-1300)
```yaml
spread_as_guard:
  enabled: false
  threshold: 1500.0   # ← 単位バグ: bps では 15%
  ev_penalty: 0.5
```

## Implementation

### 1. YAML 設定修正

```yaml
spread_as_guard:
  enabled: false  # 有効化は手動で行う
  threshold_bps: 15.0  # 低スプレッド閾値 (bps)。スプレッドがこの値未満なら EV ペナルティ適用
  ev_penalty: 0.5
```

**キー名変更**: `threshold` → `threshold_bps` (単位を明示)

### 2. Config クラス更新

`scripts/v460/lib/fill_config.py` の `SpreadAsGuardConfig` (or 該当 dataclass):
- フィールド名を `threshold` → `threshold_bps` に変更
- デフォルト値を `15.0` に変更
- バリデーション追加: `0.0 < threshold_bps < 100.0` (100 bps = 1% を上限ガード)

### 3. entry_gate_adjustments.py 更新

```python
def apply_spread_as_guard(
    ev_score: float,
    spread_bps: float,
    cfg: SpreadAsGuardConfig,
) -> tuple[float, bool]:
    if not cfg.enabled:
        return ev_score, False
    if spread_bps < cfg.threshold_bps:
        return ev_score - cfg.ev_penalty, True
    return ev_score, False
```

### 4. スプレッド分布分析ユーティリティ (新規)

`scripts/v460/analysis/sections/section_spread_distribution.py`:
- fill_records からスプレッド分布を集計 (p10, p25, p50, p75, p90)
- AS rate を spread bucket ごとに表示 (既存 spread_payload と重複しない形で)
- `threshold_bps` 候補値ごとの影響推定 (何件が guard 適用対象になるか)

### Test file: `tests/unit/v460/test_700_spread_as_guard_fix.py`

1. `test_threshold_bps_field_rename` — config が `threshold_bps` を読めること
2. `test_guard_fires_at_correct_bps` — spread=10 bps, threshold_bps=15 → ペナルティ適用
3. `test_guard_passes_above_threshold` — spread=20 bps, threshold_bps=15 → 通過
4. `test_boundary_exact_threshold` — spread=15 bps, threshold_bps=15 → 通過 (< なので)
5. `test_enabled_false_bypasses` — enabled=false なら threshold 無関係で通過
6. `test_ev_penalty_application` — ev_score=2.0, penalty=0.5 → 1.5 が返る
7. `test_config_validation_rejects_extreme` — threshold_bps=1500 は ValidationError
8. `test_backward_compat_yaml_key` — 旧キー `threshold` でも読めるか、又はエラーメッセージが明確か

## Constraints
- `enabled: false` は維持。有効化は手動。
- 既存の `entry_gate_adjustments.py` のインターフェースを極力維持
- Config のフィールド名変更に伴い、YAML を参照している全箇所を grep で確認すること
- Run: `python -m pytest tests/unit/v460/test_700_spread_as_guard_fix.py -x --tb=short -q`
- 全テスト: `python -m pytest tests/ -x --tb=short -q` で regression なしを確認
