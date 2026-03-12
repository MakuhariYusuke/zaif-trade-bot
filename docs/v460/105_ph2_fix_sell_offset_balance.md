# 105# SELL offset 引上げ + balance insufficient 削減

- **parent**: `104_ph2_fix_self_review_retrain.md`
- **status**: committed
- **commit**: (後記)

## 背景

104# 運用データ分析より:
- **SELL avg PnL: -0.85 bps** (BUY -0.36 bps の 2.4 倍悪い)
- **SELL AS%: 25.8%** (BUY 21.2%)
- **balance insufficient API 400 エラー: 114 件** (BTC 42, JPY 32, minimum_size 40)

## 変更一覧

### §1 SELL offset 保守化 (YAML)

| パラメータ | Before | After | 根拠 |
|---|---|---|---|
| `side_offset.sell` | 0.12 | **0.14** | SELL PnL -0.85bps → AS% 25.8% 低減 |
| `sell_guard.offset_floor` | 0.08 | **0.10** | sell 下限引上げで最悪ケース抑制 |

**ファイル**: `configs/v460/fill_test.yaml`

### §2 Lot floor guard — 発注直前バリデーション

**問題**: `_check_balance_for_side()` から `place_order()` までの間にタイムラグがあり、
浮動小数点演算の丸め誤差で lot < 0.001 BTC になり API 400 エラーが発生。

**対策**: `place_order()` 呼び出し直前に lot を 0.001 BTC 単位に切り捨て + 最低保証:

```python
# 105#: lot floor guard
self._current_lot = max(
    self._MIN_ORDER_BTC,
    int(self._current_lot / self._MIN_ORDER_BTC) * self._MIN_ORDER_BTC,
)
```

**適用箇所**:
1. 初回発注 (`run_single_cycle` L1422 付近)
2. stale order reprice (`run_single_cycle` L1707 付近)

**ファイル**: `scripts/v460/run_fill_test.py`

### §3 balance_shrink lot 切り捨て

**問題**: `balance_shrink` で `current_lot / divisor` が 0.001 BTC 非アラインの値を生成し、
API minimum_size エラー (40 件) の原因となっていた。

**対策**: shrink 計算結果を 0.001 BTC 単位に切り捨て:

```python
raw_shrunk = self._current_lot / self.config.balance_shrink_divisor
self._current_lot = max(
    min_lot,
    int(raw_shrunk / self._MIN_ORDER_BTC) * self._MIN_ORDER_BTC,
)
```

**ファイル**: `scripts/v460/run_fill_test.py`

### §4 テスト更新

- `test_fill_quality.py::Test052AdaptSellOffsetSync::test_yaml_sell_offset_updated`
  - YAML アサーション 0.12 → 0.14 に更新

## 期待効果

| 指標 | Before | 目標 |
|---|---|---|
| SELL avg PnL | -0.85 bps | > -0.60 bps |
| SELL AS% | 25.8% | < 22% |
| balance insufficient 400 | 114 件/日 | < 30 件/日 |
| minimum_size 400 | 40 件/日 | 0 件/日 |

## テスト結果

- 811 passed, 0 failed (v460 unit tests)
