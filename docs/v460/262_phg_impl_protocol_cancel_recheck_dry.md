# 262# adaptation_engine Protocol化 + order_monitor cancel-recheck DRY

> Issue: 262#  
> Commit: (本ドキュメント作成時点で未コミット)  
> テスト: 28 新規 / 3648 total (3616 passed, 32 skipped)

---

## 概要

型安全性と DRY 原則の 2 軸で品質改善を実施。

1. **adaptation_engine.py**: `type: ignore` ×4 を Protocol 定義で完全排除
2. **order_monitor.py**: cancel → fill recheck パターン 3 重複を 1 ヘルパーに統合

---

## A) adaptation_engine Protocol化

### 変更前の問題

```python
# fast_fill_defense: object | None — update_base_offsets() 呼び出しに type: ignore 必要
fast_fill_defense.update_base_offsets(...)  # type: ignore[union-attr]  # ×2

# adapter: object — get_current_price / get_balance 呼び出しに type: ignore 必要
await adapter.get_current_price(symbol)  # type: ignore[attr-defined]
await adapter.get_balance()             # type: ignore[attr-defined]
```

### 導入した Protocol

| Protocol | メソッド | 用途 |
|---|---|---|
| `FastFillDefenseLike` | `update_base_offsets(base, buy, sell)` | `try_auto_adapt()` の FFD 更新 |
| `LossCapAdapterProtocol` | `get_current_price(symbol)` / `get_balance()` | `update_dynamic_loss_cap()` の残高取得 |
| `_LossCapBalanceLike` | `.currency` / `.total` (property) | `get_balance()` 返却要素の型 |

### BalanceAdapterProtocol との違い

| 項目 | BalanceAdapterProtocol (261#) | LossCapAdapterProtocol (262#) |
|---|---|---|
| `get_balance()` 引数 | `currency: str` | なし (全通貨一括) |
| 返却要素の属性 | `.free` | `.currency`, `.total` |
| 用途 | BalanceChecker (通貨別残高) | 動的 loss_cap (JY/BTC 合計) |

→ インタフェースが異なるため再利用不可、新規 Protocol として定義。

### 変更サマリ

- `fast_fill_defense: object | None` → `FastFillDefenseLike | None`
- `adapter: object` → `LossCapAdapterProtocol`
- `type: ignore[union-attr]` ×2、`type: ignore[attr-defined]` ×2 → 全削除

---

## B) order_monitor cancel-recheck DRY化

### 変更前の問題

`monitor()` メソッド内に以下の同一パターンが **3 箇所** 重複:

```python
try:
    await adapter.cancel_order(order.order_id)
except Exception as cancel_err:
    if "Failed to cancel" in str(cancel_err) or "not found" in str(cancel_err).lower():
        try:
            recheck = await adapter.get_order_status(order.order_id)
            if recheck and _parse_order_state(recheck.status) == _STATE_FILLED:
                filled = True
                fill_price = recheck.price or order_price
                t_fill = time.time()
                # ...
        except Exception as exc:
            logger.debug(...)
    # ...
```

重複箇所:
1. **Adverse drift cancel** (逆選択撤退)
2. **Favorable drift cancel** (有利方向 reprice 前のキャンセル)
3. **Timeout cancel** (タイムアウト後のキャンセル)

### 導入したヘルパー

```python
class _CancelFillCheck:
    """cancel-recheck 結果."""
    __slots__ = ("was_filled", "fill_price", "t_fill", "cancel_succeeded")

@staticmethod
async def _try_cancel_with_fill_recheck(
    adapter: ExchangeAdapter,
    order_id: str,
    fallback_price: float,
) -> _CancelFillCheck:
```

### 呼び出し側の簡潔化

```python
# Before: 13-15 行 (例外ネスト 3 段)
# After:  8-10 行 (フラットな if/else)
chk = await self._try_cancel_with_fill_recheck(adapter, order.order_id, order_price)
if chk.was_filled:
    filled = True
    fill_price = chk.fill_price
    t_fill = chk.t_fill
    _cancel_failed_likely_filled = True
```

### 行数削減

- `monitor()` 本体: 約 45 行削減 (3×15行 → 3×8行 + ヘルパー 30行)
- `except Exception` 数: `monitor()` 内 11 → 5 (ヘルパー内 2)

---

## テスト

| クラス | テスト数 | 検証内容 |
|---|---|---|
| `TestAdaptationEngineProtocols` | 10 | Protocol 存在、シグネチャ型、type:ignore 排除、Protocol 順守、モック呼び出し |
| `TestCancelFillCheck` | 4 | デフォルト値、filled/failed 値、__slots__ |
| `TestTryCancelWithFillRecheck` | 8 | cancel 成功、Failed to cancel + filled、not found + filled、未 fill、予期しないエラー、fallback price |
| `TestOrderMonitorCancelRecheckDRY` | 5 | monitor() 内重複排除、ヘルパー存在、staticmethod |
| `TestOrderMonitorExceptCount` | 2 | except Exception 数削減 (monitor ≤6, helper =2) |

---

## 影響範囲

| ファイル | 変更種別 |
|---|---|
| `scripts/v460/lib/adaptation_engine.py` | Protocol 追加、シグネチャ変更、type:ignore 削除 |
| `scripts/v460/lib/order_monitor.py` | ヘルパー抽出、cancel-recheck 統合 |
| `tests/unit/v460/test_262_protocol_cancel_recheck.py` | 新規 (28 テスト) |

---

## 残タスク (今後の候補)

- **P2-4: skip_gate_evaluator getattr** — 保留中 (メタプログラミング、3 カテゴリに分類済み)
- **order_monitor `except Exception`**: 残り 5 箇所は意図的 (ポーリングループの堅牢性)
- **fill_loop_orchestrator `run_continuous()`**: 1694 行 → 分割対象 (最大 ROI だが高リスク)
- **P3 市場理論**: AS δ*, Kelly f*/2, Kyle's Lambda, Amihud ILLIQ
