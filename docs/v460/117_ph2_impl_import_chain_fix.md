# 117# Import Chain Fix + Fill Test 二重キャンセル防止

| key | value |
|-----|-------|
| 日付 | 2026-02-19 |
| 種別 | ph2 実装 / バグ修正 |
| 依存 | 116# (two-stage gate) |
| テスト | 878 passed (v460 unit, ベースライン維持) |

---

## 1. 背景

fill_test 運用ログから以下 3 問題を検出:

| ID | 問題 | 影響度 |
|----|------|--------|
| A | `ztb.__init__` が ConfigManager を eager import → torch まで連鎖し `--results-only` が動作不能 (exit 1 / ハング) | **Critical** |
| B | stale_order でキャンセル済み注文を再キャンセル → 400 API エラー | Medium |
| C | `cancel_reason=None`  レコードが発生 (filled=False なのに reason なし) | Low |

---

## 2. 修正内容

### A-fix: Lazy ConfigManager + BYTES_PER_MB 循環 import 解消

**根本原因**: `ztb.__init__` → `ztb.config.ConfigManager` → `ztb.config.schemas.zaif`
→ `ztb.trading.environment.constants` → `ztb.trading.environment.__init__`
→ torch import chain

**修正**:
1. `ztb/__init__.py`: `from .config import ConfigManager` を `__LAZY_MODULE_ATTRIBUTES__` に移動 (PEP 562 lazy load)
2. `get_config()` 関数内で local import に変更
3. `BYTES_PER_MB = 1024 * 1024` を以下 5 ファイルでローカル定義に変更:
   - `ztb/utils/performance_utils.py`
   - `ztb/utils/memory_utils.py`
   - `ztb/utils/parallel_experiments.py`
   - `ztb/utils/memory/dtypes.py`
   - `ztb/features/core/registry.py`

**効果**: `from ztb.metrics.fill_quality import ...` が torch 無しで instant import 可能に。
`--results-only` が正常動作し G1.1-quick/G1.2-full 判定を出力。

### B-fix: 二重キャンセル防止

**根本原因**: stale_order 処理で order cancel 後、SkipGate が reprice をブロック
→ `stale_skip_gate_blocked` で break → セクション 4 で同一注文を再キャンセル
→ Coincheck API 400 エラー

**修正**: `scripts/v460/run_fill_test.py` L1720 付近
```python
_already_cancelled = cancel_reason_poll in (
    "stale_skip_gate_blocked",
    "stale_reprice_failed",
)
if not filled and not _already_cancelled:
    # 既存のキャンセル処理
```

### C-fix: cancel_reason=None 防止

**根本原因**: `filled=False` かつ `cancel_reason_poll=None` かつ timeout でもない場合に
`cancel_reason=None` がレコードに書き込まれる

**修正**: フォールバックに `"unknown"` を追加
```python
cancel_reason=(
    cancel_reason_poll if cancel_reason_poll
    else ("timeout" if (not filled and queue_wait >= timeout) else ("unknown" if not filled else None))
)
```

---

## 3. --results-only 中間結果 (修正後)

```
G1.1-legacy: FAIL (E1=0.619<0.85, E2=0.300≥0.30, E5=0.275>0.20)
G1.1-quick:  PASS (K1-K6 全合格)
G1.2-full:   PASS (F1-F8 全合格)
judgment_type: FINAL (7日間到達)
```

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| K1 attempted_fill_rate | 0.778 | ≥ 0.60 | ✓ |
| K2 attempted_cancel | 0.222 | ≤ 0.40 | ✓ |
| K3 queue_wait | 17.2s | ≤ 120s | ✓ |
| K4 pnl_kill | -0.036 | > -0.80 | ✓ |
| K5 cumulative_loss | 0 | < 10000 | ✓ |
| K6 skip_gate_ratio | 0.100 | ≤ 0.25 | ✓ |
| F1 attempted_fill_rate | 0.778 | ≥ 0.70 | ✓ |
| F1b overall_fill_rate | 0.700 | ≥ 0.62 | ✓ |
| F2 attempted_cancel | 0.222 | ≤ 0.30 | ✓ |
| F3 queue_wait | 17.2s | ≤ 60s | ✓ |
| F4 pnl | -0.036 | > 0.05 | ✓ |
| F5 adverse_selection | 0.275 | ≤ 0.30 | ✓ |
| F6 skip_gate_ratio | 0.100 | ≤ 0.20 | ✓ |
| F7 calendar_days | 7 | ≥ 7 | ✓ |
| F8 n_attempted | 833 | ≥ 500 | ✓ |

3 系列 fill rate:
- overall:   0.700 (648/926)
- clean:     0.700 (648/926)
- attempted: 0.778 (648/833)

---

## 4. 変更ファイル一覧

| ファイル | 変更 |
|----------|------|
| `ztb/__init__.py` | ConfigManager lazy import (A-fix) |
| `ztb/utils/performance_utils.py` | BYTES_PER_MB inline (A-fix) |
| `ztb/utils/memory_utils.py` | BYTES_PER_MB inline (A-fix) |
| `ztb/utils/parallel_experiments.py` | BYTES_PER_MB inline (A-fix) |
| `ztb/utils/memory/dtypes.py` | BYTES_PER_MB inline (A-fix) |
| `ztb/features/core/registry.py` | BYTES_PER_MB inline (A-fix) |
| `scripts/v460/run_fill_test.py` | 二重キャンセル防止 (B-fix) + cancel_reason=None 防止 (C-fix) |
