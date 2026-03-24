# 603# sell_age_cap exceeded 時の滞留注文キャンセル

## 概要

`sell_age_cap exceeded` で micro_timeout re-quote ループを中断する際、直前に配置した re-quote 注文をキャンセルせずに放置するバグを修正。

## 背景

602# の根本原因調査で発覚。509# で導入された `sell_age_cap` ガードは、sell 注文の総滞留時間が `sell_age_cap_sec` (25s) を超えた場合に re-quote ループを `break` するが、**直前のイテレーションで配置した re-quote 注文のキャンセルを行っていなかった**。

### 障害パターン

```
[iteration N-1]
  _monitor_fill_polling() → timeout → old order cancelled
  re-quote: place_order() → new order #xyz
[iteration N]
  age_cap check → exceeded → break  ← new order #xyz はキャンセルされず放置
```

放置された注文が `btc_reserved` を拘束し、buy/sell 両側膠着 → preflight 連続失敗 → SAFE_STOP に至る (602# で recovery を追加済みだが、根本原因の修正が本チケット)。

## 変更内容

### `scripts/v460/lib/fill_cycle_executor.py`

`_monitor_fill_phase()` 内の age_cap exceeded ブロックに `cancel_order()` を追加:

```python
if mt_elapsed >= mt_total_cap:
    logger.info("[509#] micro_timeout sell_age_cap exceeded: ...")
    # 603# age_cap exceeded: 滞留注文をキャンセル
    try:
        await self.adapter.cancel_order(order.order_id)
    except Exception as e:
        logger.warning("[603#] Cancel failed ... (may be filled/cancelled): %s", e)
    break
```

- 約定済み or キャンセル済みの場合は例外を warning ログで吸収 (非致命的)

### テスト

`tests/unit/v460/test_regime_detector.py::TestAgeCapCancelOrder` (2 件):

- `sell_age_cap exceeded` の後に `cancel_order` 呼び出しが存在
- `cancel_order` が `break` より前に呼ばれる
