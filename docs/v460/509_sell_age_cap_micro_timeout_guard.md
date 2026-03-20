# 509# sell_age_cap × micro_timeout ガード + stale reprice 残時間チェック

## 概要

micro_timeout ループが sell_age_cap を無視して最大 55 秒間 sell を保持し続ける
クリティカルバグを発見・修正。併せて stale reprice の残時間不足問題も修正。

---

## 発見された問題

### BUG-A: micro_timeout ループが sell_age_cap を超過 (Critical)

**問題:**
- `sell_age_cap_sec = 25s` 設定にも関わらず、micro_timeout ループは
  最大 `4 rounds × 10s + 3 cooloffs × 5s = 55s` 間 sell 注文を保持
- 原因: sell_age_cap は `monitor()` 内の単一ラウンドの timeout（10s < 25s）に
  対してのみチェックされ、ラウンド跨ぎの累積時間は未検査

**修正 (`fill_cycle_executor.py`):**
- `_first_t_submit` からの経過時間を各ラウンド冒頭でチェック
- `elapsed >= sell_age_cap_sec` なら即座に micro_timeout ループを break
- buy 側は cap 対象外（`_mt_total_cap = None`）

```python
# 509# sell_age_cap を micro_timeout ループ全体にも適用
_mt_total_cap = (
    self.config.sell_age_cap_sec
    if side == "sell" and self.config.sell_age_cap_sec is not None
    and self.config.sell_age_cap_sec > 0
    else None
)

for _mt_attempt in range(_mt_max):
    if _mt_total_cap is not None:
        _mt_elapsed = time.time() - _first_t_submit
        if _mt_elapsed >= _mt_total_cap:
            logger.info("[509#] micro_timeout sell_age_cap exceeded: ...")
            break
```

### BUG-B: stale reprice の残時間不足 (Medium)

**問題:**
- `order_monitor.py` の while ループは `elapsed < _effective_timeout` を先頭で
  チェックするが、ループ内で cancel → re-place を開始する際に残時間を確認しない
- I/O 遅延 3-5s を加味すると、timeout 直前に reprice を開始すると超過するリスク

**修正 (`order_monitor.py`):**
- favorable drift reprice 前に `_remaining = _effective_timeout - elapsed` を計算
- `_remaining < 3.0s` なら reprice をスキップ（info ログ出力）

```python
# 509# 残時間チェック
_remaining = _effective_timeout - elapsed
if _remaining < 3.0:
    logger.info("[509#] Reprice skipped: %.1fs remaining < 3s min", ...)
    continue
```

---

## テスト

`tests/unit/v460/test_506_sell_improvements.py` に追加:

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| `TestMicroTimeoutSellAgeCapGuard` | 5 | config 存在、worst-case 算出、ガード発火/非発火、buy 除外 |
| `TestRepriceRemainingTimeGuard` | 2 | 残時間 3s 未満でスキップ、3s 以上で許可 |
| `TestHintBasisObservability` (508#) | 5 | basis/adjusted_spread 伝搬、fill_fields |

合計: 25 tests pass（既存 13 + 新規 12）

---

## 影響範囲

- `scripts/v460/lib/fill_cycle_executor.py`: micro_timeout ループ部
- `scripts/v460/lib/order_monitor.py`: stale reprice 判定部
- 既存動作への影響なし（sell_age_cap 未設定時は `_mt_total_cap = None` で無関係）

## 調査で排除した False Positive

| 項目 | 結論 |
|------|------|
| confidence boost の de-meaning 不足 | 数学的に正しい（`1.0 + (boost-1.0) * conf` は [1.0, boost] に正しく補間） |
| hot-reload フィールド欠落 | grep で全フィールド存在確認済 |
| basis integration 不完全 | fill_cycle_executor に完全統合済 |
| sidecar_signal_io / skip_gate 問題 | コード精査済、問題なし |
