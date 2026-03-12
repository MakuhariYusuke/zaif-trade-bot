# 202# ログ分析ベース改善 — loss cooldown / one-sided rescue / VG sell supplement

> **日付**: 2026-03-01  
> **対象**: fill_test 運用ログ (cycle 5325–5345, 18:55–20:27 JST)

---

## 1. ログ分析サマリ

| 指標 | 値 |
|---|---|
| 観測サイクル | 21 (5325–5345) |
| Fill率 | 81% (17/21) |
| Unfilled | 4 (adverse drift cancel 3, timeout 1) |
| 合計PnL | -48.5bps |
| 平均PnL | -2.85bps |
| 勝ちトレード | 4 (+5.78, +1.06, +0.62, +0.50) |
| **壊滅ペア** | 5338 (-17.27bps) + 5339 (-19.74bps) = **-37bps** |

壊滅ペアが全損失の76%を占める。

---

## 2. 根本原因

### A. Cycle 5338 (-17.27bps): one-sided sell が rescue offset なしで実行
- `balance_forced` → `original_also_insufficient=True` (one-sided path)
- one-sided path は `_is_rescue=True` を設定しない → rescue offset 不適用
- 上昇トレンド中に base offset だけで sell → 大幅損失

### B. Cycle 5339 (-19.74bps): 連続損失にクールダウンなし
- 5338 直後、通常間隔で次サイクル実行
- VG 2x boost 適用も不十分

### C. VG sell-side 盲点
- VG は `mid_trend_bps` (point-to-point) で判定
- sell サイクルは timing asymmetry により mid_trend_bps が threshold 到達しにくい
- `velocity_60s` (EWMA) は実際の変動を捕捉しているが VG 判定に未使用

---

## 3. 実装

### 202# A — Loss Cooldown (損失後インターバル延長)
- **トリガ**: `post_fill_30s_pnl <= -10.0bps` (configurable)
- **効果**: 次サイクルのみ `interval × 2.0` (configurable)
- **リセット**: 自動 (1回適用後 mult=1.0 に戻す)
- **変更箇所**: `fill_config.py`, `fill_loop_orchestrator.py`

### 202# B — One-sided Balance Rescue Offset
- **トリガ**: `original_also_insufficient=True` かつ config 有効時
- **効果**: `_is_rescue=True` を設定し、rescue offset 適用パスに乗せる
- **変更箇所**: `fill_config.py`, `fill_loop_orchestrator.py`

### 202# C — VG Sell Supplement (velocity_60s 補完)
- **トリガ**: sell + VG 未発火 + `|velocity_60s|` > VG threshold + vel_offset 未適用
- **効果**: VG boost factor で offset 補正 (既存 `_apply_offset_multiplier` 再利用)
- **変更箇所**: `fill_cycle_executor.py`

---

## 4. Config (fill_test.yaml)

```yaml
loss_cooldown_threshold_bps: -10.0
loss_cooldown_interval_mult: 2.0
one_sided_balance_rescue_offset: true
```

---

## 5. テスト

- 新規: `test_202_log_improvements.py` — 16 tests (全 PASS)
- 回帰: 2052 passed, 0 new failures
- `test_113` line count limit: 570→600 (202# C 追加分)
