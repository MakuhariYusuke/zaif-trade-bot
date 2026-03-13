# 229# コード衛生 + M-5 unknown counter fix + M-2 副作用 getter 改名

> **前提**: 228# (`43ec62ee1`) で inv_skew time-decay と hasattr 排除を実装済み。
> 残存していたコード衛生課題・バグ・命名問題を一括修正。

---

## 概要

| ID | 分類 | 内容 | ファイル |
|----|------|------|----------|
| M-5 | Bug Fix | unknown regime counter Gate 2-3 early return でリセット漏れ | `cycle_gate_aggregator.py` |
| M-2 | API改善 | `get_recovery_lot_scale()` → `consume_recovery_cycle()` | `daily_drawdown_guard.py` |
| H-1/Q5 | hasattr排除 | `_apply_regime_boosts` 6x hasattr → `is not None` | `maker_price.py` |
| H-3 | getattr排除 | `_soft_drawdown_interval_multiplier` 2x getattr → direct | `fill_loop_orchestrator.py` |
| H-4 | import整理 | 2x `import time as _time` → module-level `import time` | `fast_fill_defense.py` |
| H-5 | import整理 | 1x `import time as _time` → 既存 module import 使用 | `fill_loop_orchestrator.py` |
| Q1 | getattr排除 | `getattr(self._config, "inv_decay_tau_sec", 0.0)` → direct | `maker_price.py` |

---

## M-5: unknown regime counter reset 漏れ (Bug Fix)

### 問題

```
Gate 1 (unknown_buy) ブロック → _consecutive_unknown_blocks++
Gate 2 (ranging_low_vol) ブロック → early return → カウンタ未リセット!
Gate 3 (trending_sell) ブロック → early return → カウンタ未リセット!
```

**影響シナリオ**:
1. unknown 5回ブロック → counter=5
2. regime=ranging → Gate 2 ブロック → counter 未リセット (5のまま)
3. unknown 5回ブロック → counter=10 → **偽バイパス発動** (閾値=10)

実際には連続 unknown は 5 回しかないのに、間の ranging を挟んでも
カウンタが蓄積し、バイパス閾値に偽到達するリスクがあった。

### 修正

Gate 1 通過直後に非 unknown リセットを挿入:

```python
# Gate 1 通過後
if _regime != "unknown":
    self._consecutive_unknown_blocks = 0
# → Gate 2-6 のどの early return でもリセット済み
```

旧コード (Gate 7 直後の条件付きリセット) は冗長になったため削除。

---

## M-2: `get_recovery_lot_scale()` → `consume_recovery_cycle()`

### 問題

`get_recovery_lot_scale()` は getter 命名だが、内部で `side_recovery_remaining` を
デクリメントする副作用がある。呼び出し側が「参照だけ」と誤解するリスク。

### 修正

- `daily_drawdown_guard.py`: メソッド名 + docstring 更新
- `fill_loop_orchestrator.py`: 呼び出し元更新
- `test_224_*.py`, `test_225_*.py`: テスト側更新
- `restore_recovery_counter()` docstring: 参照先名更新

---

## コード衛生 (H-1, H-3, H-4, H-5, Q1)

### hasattr 排除 (H-1/Q5)

`maker_price.py._apply_regime_boosts()` から 6 箇所の `hasattr` を削除:
- 5x `hasattr(self._regime_detector, "current_regime")`
- 1x `hasattr(self._regime_detector, "last_volatility_ratio")`

`_regime_detector` は typed as `object | None` で duck typing。
`is not None` チェック + 直接アクセスに統一。

### getattr 排除 (H-3)

`fill_loop_orchestrator.py` の `getattr(self, "_soft_drawdown_interval_multiplier", 1.0)`:
- L275 (state export): → `self._soft_drawdown_interval_multiplier`
- L384 (interval計算): → `self._soft_drawdown_interval_multiplier`

クラスレベル `_soft_drawdown_interval_multiplier: float = 1.0` (L46) が保証。

### inline import 排除 (H-4, H-5)

| ファイル | 変更前 | 変更後 |
|----------|--------|--------|
| `fast_fill_defense.py` L97 | `import time as _time` | module-level `import time` |
| `fast_fill_defense.py` L206 | `import time as _time` | (同上) |
| `fill_loop_orchestrator.py` L191 | `import time as _time` | 既存 `import time` 使用 |

---

## テスト

| セクション | テスト数 | 検証内容 |
|-----------|---------|---------|
| FFD import | 3 | inline import 排除, module-level import 確認, TTL decay 動作 |
| hasattr排除 | 3 | ソースに hasattr なし, regime=None 動作, regime 設定時動作 |
| inv_decay直接 | 4 | attr存在, デフォルト値, getattr不在, decay動作 |
| orchestrator getattr | 2 | getattr排除, inline import排除 |
| M-5 counter fix | 5 | ranging reset, trending reset, unknown維持, 非unknown reset, bypass偽到達防止 |
| M-2 rename | 5 | メソッド存在, 旧名不在, デクリメント動作, docstring更新, orchestrator呼び出し |
| regression | 3 | FFD evaluate_fill, regime=None初期化, gate全通過 |
| **合計** | **25** | |

総テスト: **3109 passed**, 0 failed

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|----------|----------|
| `scripts/v460/lib/cycle_gate_aggregator.py` | M-5 fix |
| `scripts/v460/lib/daily_drawdown_guard.py` | M-2 rename |
| `scripts/v460/lib/maker_price.py` | H-1/Q5, Q1 |
| `scripts/v460/lib/fast_fill_defense.py` | H-4 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | H-3, H-5, M-2 call site |
| `tests/unit/v460/test_229_cleanup_counter_rename.py` | 新規 25 テスト |
| `tests/unit/v460/test_224_halt_recovery_and_kill_reset.py` | M-2 rename 対応 |
| `tests/unit/v460/test_225_warmup_recovery_fire_counts.py` | M-2 rename 対応 |

---

## 残課題 (次回以降)

| ID | 重要度 | 内容 |
|----|--------|------|
| C-1 | Critical | FFD Layer 2 `pnl < 0` → `< -as_deadzone_bps` (spread cost 誤検知) |
| C-2 | Critical | FFD boost instant release (Kyle 1985: 1 normal fill ≠ threat gone) |
| M-3 | Medium | 16 missing config validations in `__post_init__` |
| M-4 | Medium | naming inconsistency across modules |
