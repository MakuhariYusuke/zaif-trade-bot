# 526# ログ可観測性改善 + dead code 削除

> 524# の preflight_skip_exceeded 停止分析で発覚したログ盲点を修正し、
> 525# Codex レビューで指摘された dead code を除去する。

---

## 1. 背景

524# の分析において、以下のログ不足がインシデント特定・原因究明を遅延させた:

- **order_id の欠如**: キャンセルログに order_id がなく、未キャンセルの注文漏れを追跡できなかった
- **残高コンテキストの欠如**: Insufficient ログに `free` vs `required` の明確な表示がなかった
- **cycle result に order_id がない**: 約定/未約定のサイクル終了時にどの注文だったか不明

---

## 2. 変更一覧

### P0: クリティカルログ改善

| ファイル | 変更内容 |
|---|---|
| `balance_checker.py` | Insufficient BTC/JPY ログに `free=X < min=Y` 形式を導入 |
| `order_monitor.py` L556 | timeout cancel ログに `order.order_id` と `side` を追加 |

### P1: 重要ログ改善

| ファイル | 変更内容 |
|---|---|
| `order_monitor.py` L87 | cancel unexpected error ログに `order_id` 追加 |
| `order_monitor.py` L110 | cancel failed ログに `order_id` 追加 |
| `order_monitor.py` L338 | poll error ログに `order.order_id` と `side` 追加 |
| `order_monitor.py` L345 | F9 consecutive poll error ログに `order.order_id` と `side` 追加 |
| `fill_cycle_executor.py` | `_log_cycle_result()` に `order_id` パラメータ追加、ログ出力に `id=` タグ追記 |
| `fill_cycle_executor.py` | "All order attempts failed" ログに `side` と `qty` 追加 |
| `fill_cycle_executor.py` | rate-limit backoff ログに `attempt` 番号追加 |

### P2: コンテキスト改善

| ファイル | 変更内容 |
|---|---|
| `orchestrator_balance.py` | balance_shrink ログの BTC 精度を `.4f` → `.8f` に向上、`min_lot` 追加 |

### 525# 指摘対応: dead code 削除

| ファイル | 変更内容 |
|---|---|
| `maker_price.py` | `_apply_final_offset_ceiling()` メソッド削除 (523# で offset_pipeline 一本化後、呼び出し元ゼロ) |

---

## 3. ログ出力 Before/After

### balance_checker.py — BTC 不足

**Before:**
```
Insufficient BTC for sell: 0.000000 < 0.0010 (regime_mult=1.00)
```

**After:**
```
Insufficient BTC for sell: free=0.00000000 < min=0.0010 (regime_mult=1.00)
```

### order_monitor.py — timeout cancel

**Before:**
```
Cancelled unfilled order after 30.0s
```

**After:**
```
Cancelled unfilled order 8770082779 after 30.0s (side=sell)
```

### fill_cycle_executor.py — cycle result

**Before:**
```
Cycle 14525 result: filled=True, wait=61.5s, pnl=1.22bps, sidecar=stale
```

**After:**
```
Cycle 14525 result: filled=True, wait=61.5s, pnl=1.22bps, id=8770082779, sidecar=stale
```

---

## 4. テスト結果

- 全 v460 テスト: 3421 passed, 5 failed (全て既存の既知失敗)
- 新規リグレッション: なし
