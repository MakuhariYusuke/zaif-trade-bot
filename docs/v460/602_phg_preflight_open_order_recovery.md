# 602# preflight SAFE_STOP 直前の open order recovery

## 概要

preflight_skip_exceeded による SAFE_STOP を発動する前に、滞留 open order のキャンセルを試みる recovery パスを追加。

## 背景・根本原因

2026-03-24 21:53:45、n=16164 で `preflight_skip_exceeded` により fill_test が停止。

### タイムライン

| 時刻 | イベント |
|------|---------|
| 20:48:50 | sell order #8778575030 配置 (micro_timeout re-quote 1/4) |
| 20:49:07 | status_unknown 3 回リトライ後 → cancel 試行 |
| 20:49:08 | cancel 失敗: "Failed to cancel the order" (約定済みの可能性) |
| 20:49:14 | sell order #8778575857 配置 (re-quote 2/4) |
| 20:49:14 | `sell_age_cap exceeded` (41.3s ≥ 25s) → re-quote 停止 **注文キャンセルなし** |
| 20:50:15 | `btc_reserved: 0.00209677` — BTC が open sell order に拘束 |
| 20:50~ | JPY=230.48 (buy 不可), BTC free=0 (sell 不可) → 両側膠着 |
| 21:07~21:30 | preflight_pause ×3 (各 300s) |
| 21:53:45 | SAFE_STOP: 連続 preflight スキップ 10 回 |
| 21:54:52 | watchdog 自動再起動 |
| 21:55:05 | startup の `_cancel_stale_orders()` で order #8778575857 キャンセル → 復旧 |

### 根本原因

1. `sell_age_cap exceeded` 停止時に open order をキャンセルしない設計欠陥
2. runtime に滞留 open order をキャンセルする recovery パスが存在しなかった
3. 524#/525# で提案された shared `_cancel_stale_orders()` が未実装 (527# `⏳ 未対応`)

## 変更内容

### `scripts/v460/lib/orchestrator_balance.py`

`_handle_preflight_failure()` に open order recovery ロジックを追加:

```
shrink → pause (×3) → [NEW] open order cancel attempt → retry or SAFE_STOP
```

- `preflight_skip_count >= max_preflight_skip` 到達時、SAFE_STOP 前に `get_open_orders()` を呼び出し
- open order が存在すれば `cancel_order()` でキャンセル
- 1 件以上キャンセル成功 → `preflight_skip_count = 0` にリセットし retry
- open order なし or キャンセル失敗 → 従来通り SAFE_STOP

### テスト

`tests/unit/v460/test_regime_detector.py::TestPreflightOpenOrderRecovery` (4 件):

- `get_open_orders` 呼び出しがソースに存在
- `cancel_order` 呼び出しがソースに存在
- cancel 成功後に `_preflight_skip_count = 0` リセット
- recovery 失敗時に `preflight_skip_exceeded` (SAFE_STOP) に到達可能

## 残課題

- `sell_age_cap exceeded` 時のキャンセル漏れ修正 (上流バグ、別チケットで対応)
