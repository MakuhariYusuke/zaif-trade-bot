# Codex Prompt: インベントリスキューイング強化 (700# Task 3)

## Goal
`maker_price.py` のインベントリスキューイングを強化し、長期在庫ドリフトへの耐性を向上させる。`deque(maxlen=100)` を configurable にし、ドリフト検出 → 段階的 max_factor スケーリングを追加する。

## Context
- **698# 指摘**: window=100 は累積ドリフトに対して短すぎる。BTC 比率が 94.1% まで偏った。
- **699# 留保**: window=1000 + max_factor=0.8 は攻撃的すぎ。window=300 + max_factor=0.6 を推奨。
- **既存実装**: time-decay τ=3600s (`_decayed_imbalance`) が部分的に補償するが、数百 fill の漸進ドリフトには不十分。

## 現行コード

### deque 初期化 (maker_price.py:248-249)
```python
_w = config.inventory_skewing_window if config.inventory_skewing_window > 0 else 100
self._inv_fill_history: collections.deque[str] = collections.deque(maxlen=_w)
```

### _apply_inventory_skew (maker_price.py:792-850, 要点抜粋)
```python
cfg = self._config
_decayed_imb = self._decayed_imbalance(now)

_effective_max_factor = cfg.inventory_skewing_max_factor  # 0.4
if self._regime_detector is not None:
    _r = self._regime_detector.current_regime
    if _r.is_trending:
        if cfg.inv_skew_regime_gate_enabled:
            _inv_skew_regime_blocked = True
        else:
            _effective_max_factor = cfg.inv_skew_max_factor_trending  # 0.15

if (
    not cfg.inventory_skewing_enabled
    or abs(_decayed_imb) <= cfg.inventory_skewing_neutral_band  # 0.05
    or _inv_skew_regime_blocked
):
    self._last_inv_skew_factor = 0.0
    return effective_offset_ratio

_raw_factor = _decayed_imb * _sign * _effective_max_factor
_factor = math.tanh(_raw_factor)
```

### Config (fill_config.py, 概要)
```python
inventory_skewing_enabled: bool = False
inventory_skewing_window: int = 100
inventory_skewing_max_factor: float = 0.4
inventory_skewing_neutral_band: float = 0.05  # YAML上は0.1の可能性あり、要確認
inv_decay_tau_sec: float = 3600.0
inv_skew_max_factor_trending: float = 0.15
```

## Implementation

### 1. Config 拡張 (fill_config.py)

新パラメータ追加:
```python
inventory_skewing_window: int = 300           # 100 → 300 にデフォルト引き上げ
inventory_skewing_max_factor_drift: float = 0.6  # ドリフト検出時の escalated max_factor
drift_detection_threshold: float = 0.15       # |imbalance| がこの値を超えたらドリフト判定
drift_detection_sustain_sec: float = 1800.0   # ドリフト状態が 30 分継続で escalation
```

### 2. ドリフト検出ロジック (maker_price.py)

`_apply_inventory_skew` 内に追加する段階的ロジック:

```python
# ドリフト検出: |imbalance| が drift_detection_threshold を一定時間超過
_abs_imb = abs(_decayed_imb)
_is_drifting = self._check_drift_state(_abs_imb, now)

if _is_drifting:
    _effective_max_factor = max(
        _effective_max_factor,
        cfg.inventory_skewing_max_factor_drift
    )
```

新メソッド `_check_drift_state`:
```python
def _check_drift_state(self, abs_imbalance: float, now: float) -> bool:
    """ドリフト状態を判定。一定時間以上 threshold 超過が続いたら True."""
    if abs_imbalance > self._config.drift_detection_threshold:
        if self._drift_start_time is None:
            self._drift_start_time = now
        elapsed = now - self._drift_start_time
        return elapsed >= self._config.drift_detection_sustain_sec
    else:
        self._drift_start_time = None
        return False
```

新インスタンス変数: `self._drift_start_time: float | None = None`

### 3. Fill history observability

`fill_record_builder.py` に追加:
- `inv_skew_drift_detected: bool` — ドリフト状態かどうか
- `inv_skew_effective_max_factor: float` — 実効 max_factor

### 4. YAML 設定 (configs/v460/fill_test.yaml)

```yaml
inventory_skewing_window: 300
inventory_skewing_max_factor: 0.4
inventory_skewing_max_factor_drift: 0.6
drift_detection_threshold: 0.15
drift_detection_sustain_sec: 1800
```

### Test file: `tests/unit/v460/test_700_inventory_skewing.py`

1. `test_window_expansion_default_300` — デフォルト window が 300 であること
2. `test_window_backward_compat` — config で 100 を指定すれば 100 が使われること
3. `test_drift_detection_threshold` — |imbalance| > 0.15 でドリフト開始
4. `test_drift_sustain_requirement` — 30分未満のドリフトは escalation しない
5. `test_drift_escalation_max_factor` — ドリフト確定後に max_factor が 0.6 になること
6. `test_drift_reset_on_recovery` — |imbalance| が threshold 未満に戻ればリセット
7. `test_regime_trending_interaction` — trending 時の max_factor_trending と drift の max()
8. `test_no_drift_in_neutral_band` — neutral_band 内では skewing も drift もスキップ
9. `test_fill_record_drift_fields` — fill_record に drift 関連フィールドが記録されること
10. `test_existing_behavior_preserved` — window=100, drift 無効時は既存動作と同一

## Constraints
- 後方互換必須: ドリフト検出は新パラメータのデフォルト値で有効化されるが、window=100 の明示設定を尊重
- `_decayed_imbalance` ロジックは変更しない (τ=3600s は維持)
- regime gate (`inv_skew_regime_gate_enabled`) との interaction を壊さない
- tanh 平滑化は維持
- Run: `python -m pytest tests/unit/v460/test_700_inventory_skewing.py -x --tb=short -q`
- Regression: `python -m pytest tests/ -x --tb=short -q`
