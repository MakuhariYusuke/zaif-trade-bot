# 294# P0: forced_buy_delay デッドロック修正

> **日付**: 2026-03-06
> **障害発生**: 2026-03-06 02:32 JST — buy 注文が一切出ない状態が 3h+ 継続
> **根本原因**: forced_buy_delay リアーム・ループ

---

## 1. 障害概要

2:29:17 の sell 約定後、`balance_forced_switch=True` により buy 方向に切り替わるが、
全サイクルが `cancel_reason=forced_buy_delay` でスキップされ、**83 サイクル連続**で
注文が一切出ない状態が 2:32〜5:08 まで継続。

### 状態
- regime: `trending_down` → velocity が持続的に ≤ -3.0 bps
- `soft_loss_cap_triggered: True`
- PID 58052 はプロセスとして生存（os_lock 保持）、CPU ≈ 8.8 秒しか消費していない
- 毎サイクル空レコード書き込み（`order_price=0`, `filled=false`）

## 2. 根本原因

286# で実装した `forced_buy_delay` のロジックに構造的バグ：

```
# 毎サイクルの判定
if velocity <= threshold:
    delay_remaining = max(delay_remaining, cycles)  # ← 毎回リセット！

if delay_remaining > 0:
    delay_remaining -= 1
    skip → continue
```

velocity が閾値以下を維持する限り、`delay_remaining` が 3 に毎回リセットされ、
デクリメント（-1）されても次サイクルでまた 3 に戻る → **永久ブロック**。

292# で追加した `velocity_threshold_ranging_bps: -3.0` が trending_down でも適用される
ため、緩い閾値がデッドロックをさらに起きやすくした。

## 3. 修正内容

### `forced_buy_delay_max_consecutive` (デフォルト: 10)

連続ブロック回数をカウントし、上限に達したら delay を強制リセットして突破。

```python
# 294# 追加: 連続カウンタ
_forced_buy_delay_consecutive: int = 0

# 判定ロジック
if velocity <= threshold and consecutive < max_consecutive:
    delay_remaining = max(remaining, cycles)
elif consecutive >= max_consecutive:
    delay_remaining = 0  # 強制突破
    logger.warning("[294# GM deadlock break] ...")

# skip 実行時
if delay_remaining > 0:
    consecutive += 1  # カウントアップ
else:
    consecutive = 0   # 通過 → リセット
```

### 安全性
- デフォルト 10 サイクル ≈ 約 20 秒（サイクル間隔 2 秒想定）
- 上限到達後 1 回だけ通過させ、再び velocity 判定が走る
- Hot-Reload 対応（`_HOT_RELOADABLE_FIELDS` に追加済み）

## 4. 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/fill_config.py` | `forced_buy_delay_max_consecutive` フィールド追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 連続カウンタ + 上限判定 + リセットロジック |
| `scripts/v460/lib/config_hot_reload.py` | `forced_buy_delay_max_consecutive` を Hot-Reload 対象に追加 |
| `configs/v460/fill_test.yaml` | `max_consecutive: 10` 追加 |
| `tests/unit/v460/test_292_observability.py` | 新テスト 4 件追加 (計 22 件) |

## 5. テスト

22/22 PASSED (`test_292_observability.py`)
v460 全体: 3914 passed, 32 skipped (リグレッションなし)
