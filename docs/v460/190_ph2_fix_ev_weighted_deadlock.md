# 190# ev_weighted デッドロック修正 + min_spread_jpy 緩和

## 概要

189# デプロイ後のログ分析で発見された **ev_weighted デッドロック問題** を修正。
BTC=0 状態で ev_weighted が 7+ 回連続 skip → 21+ 分間取引不能のデッドロック状態を解消。

## 問題分析

### 致命的問題: ev_weighted デッドロック
- **症状**: BTC=0 → buy しか選べない → ev_weighted が毎回 skip → 取引不能
- **根本原因**: ev_weighted の予測 PnL が全て負 (-0.291 ～ -3.803 bps) で threshold (≈0.1) 未満
- **影響**: 7 連続 skip で 21+ 分間無取引、資本効率ゼロ

### その他の問題
- `min_spread_jpy: 1200` が厳しすぎ → 28 回の "Spread too narrow" 拒否
- B1' ranging + ev_weighted のダブルブロッキング

## 改善施策

### A: ev_weighted 連続 skip 安全弁 (code change)
- `_ev_consecutive_skip_count` カウンタ追加
- `skip_gate_ev_max_consecutive_skip: 5` — 5 回連続 skip で強制 PASS
- reason: `ev_weighted_pass_safety`

### B: 片側 balance 時の threshold 緩和 (code change)
- BTC=0 で buy しか選択肢がない場合、`one_sided_balance=True` が伝搬
- `skip_gate_ev_one_sided_threshold_shift: -1.0` — threshold を 1.0 bps 緩和
- パラメータフロー: orchestrator → run_single_cycle → _evaluate_skip_gate → evaluate → _try_ev_weighted_decision

### C: min_spread_jpy 緩和 (YAML only)
- `min_spread_jpy: 1200 → 1000`
- hot-reload で即時適用 (13:48:28 JST 確認済み)

### D: pnl_threshold 緩和 (YAML)
- `pnl_threshold: 0.0 → -0.5`
- ev_weighted スコアが全て負値 → base threshold 引き下げで適応

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/skip_gate_evaluator.py` | A: 連続 skip カウンタ + 安全弁, B: one_sided_balance threshold 緩和 |
| `scripts/v460/lib/fill_cycle_executor.py` | B: one_sided_balance パラメータ伝搬 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | B: one_sided_balance フラグ設定・伝搬 |
| `scripts/v460/lib/fill_config.py` | A/B: 新設定フィールド + YAML マッピング |
| `scripts/v460/lib/config_hot_reload.py` | A/B/D: hot-reload キー追加 |
| `configs/v460/fill_test.yaml` | C/D: YAML パラメータ更新 |
| `tests/unit/v460/test_188_split_evc_macro.py` | 190# 互換性修正 (MagicMock config) |
| `tests/unit/v460/test_190_ev_weighted_safety.py` | 新規: 28 テスト |

## テスト結果

- 190# テスト: **28/28 PASSED**
- v460 全体: **2467 passed, 0 failed**

## デプロイ結果

- 再起動: 13:03:55 JST (PID 9028)
- Cycle 4868: buy 即時約定 (wait=5.9s) pnl=+0.13bps
- デッドロック解消確認

## パラメータ一覧

```yaml
skip_gate:
  ev_max_consecutive_skip: 5       # 190# A: 安全弁 (0=無効)
  ev_one_sided_threshold_shift: -1.0  # 190# B: 片側 balance 緩和
  pnl_threshold: -0.5             # 190# D: base threshold 緩和
min_spread_jpy: 1000              # 190# C: spread フィルター緩和
```
