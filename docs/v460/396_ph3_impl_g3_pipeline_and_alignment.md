# 396# G3 パイプライン実装 + 課題発見

**作成**: 2026-03-13
**根拠**: 389# P0-3 (reward-PnL alignment), 389# P1-1 (G3 gate 接続), 392# 検証結果
**分類**: ph3 — G3 gate 実装フェーズ

---

## 概要

392# で検証済みの指摘事項を実装に移行。
G3 gate (PnL収益性チェック) を `evaluate_model_oos()` → `run_experiment.py` → `run_g3_judgment()` の
パイプラインとして接続した。

実装中に **2件の CRITICAL バグ** を発見・修正。

---

## 実装内容

### 1. `evaluate_model_oos()` G3 指標拡張 (`sac_common.py`)

**追加指標** (run_g3_judgment() の seed_metrics 形式に合致):
| 指標 | 説明 |
|------|------|
| `pf` | Profit Factor: 正PnLステップ合計 / |負PnLステップ合計| |
| `max_drawdown` | Equity curve の peak-to-trough 最大下落率 |
| `sharpe_annual` | 日次リターン (1440 steps/day) の年率 Sharpe |
| `avg_gross_per_trade` | 取引あたり平均 |trade_pnl| |
| `avg_fee_per_trade` | 取引あたり手数料 (maker 0% 前提で 0.0) |
| `reward_profit_corr` | reward 累積と PnL 累積の相関 (alignment 指標) |

### 2. `_evaluate_g3_from_results()` (`run_experiment.py`)

`run_g3_judgment()` のロジックを dict 入力で再現。
ファイルパス不要で、`evaluate_gate()` 内でインライン判定可能。

### 3. G3 自動判定 (`run_experiment.py` evaluate_gate)

G2 判定後に `seed_metrics` が存在すれば自動で G3 判定を実行し、
`g3_judgment_cache` として結果に保存。

### 4. G2 artifact 再保存 (P0-1)

reward-tuned 実験結果の `g2_judgment_cache` を現行閾値で再判定。
`worst_seed_min_roi` が -0.02 → -0.035 に緩和されたことで:

| 実験 | 旧判定 | 新判定 | 変化理由 |
|------|--------|--------|----------|
| baseline (202003) | FAIL | FAIL | positive_seed_ratio=0.5 < 0.75 |
| reward-tuned (073155) | FAIL | **PASS** | worst_roi=-0.032 > -0.035 |
| warm-start (113612) | FAIL | FAIL | 3項目FAIL (ratio, std, worst) |

---

## 発見した CRITICAL バグ

### BUG-1: `pnl_history` が HeavyTradingEnv に存在しない

`sac_common.py` が `getattr(env, "pnl_history", None)` で取得しようとしていたが、
`HeavyTradingEnv` にはこの属性がない。
**影響**: PF, avg_gross_per_trade, reward_profit_corr が常に 0.0 → G3 ゲートは always FAIL。

### BUG-2: `reward_history` / `portfolio_value_history` が 100 件に切り詰め

`heavy_trading_env.py` L545-547:
```python
if len(self.reward_history) > 100:
    self.reward_history = self.reward_history[-100:]
    self.portfolio_value_history = self.portfolio_value_history[-100:]
```

OOS 評価は ~243K ステップを走査するが、エピソード後に取得できるのは最後 100 件のみ。
**影響**: max_drawdown は最後100ステップのDDしか計測せず過小評価、sharpe_annual は
100 < 1440 (1日分) のため常にフォールバック値 0.0。

### 修正方針

env 内部の履歴に依存せず、`evaluate_model_oos()` の step ループ内で
env.balance, env.unrealized_pnl, env.total_pnl を直接読み取り自前で蓄積。
env の内部状態を変更しないため、訓練時のメモリ効率に影響なし。

---

## テスト

23件のテスト (`test_396_g3_pipeline.py`):
- `_compute_g3_metrics`: 14件 (PF, MaxDD, Sharpe, per-trade, alignment)
- `_evaluate_g3_from_results`: 9件 (PASS/FAIL 各条件, NO_DATA, 構造検証)

既存テスト92件も全件 PASS 確認済み (test_gate_check, test_356_g2_sac_blockers)。

---

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/sac_common.py` | `_compute_g3_metrics()` 追加, `evaluate_model_oos()` にステップ単位データ収集 |
| `scripts/v460/run_experiment.py` | `_evaluate_g3_from_results()` 追加, seed_results G3 enrichment, G3 auto-eval |
| `tests/unit/v460/test_396_g3_pipeline.py` | 新規 23 テストケース |
| `results/v460/v460_g2train_*.json` (3件) | G2 judgment cache を現行閾値で再判定 |

---

## 残課題

| ID | 優先度 | 内容 |
|----|--------|------|
| R1 | P1 | warm-start (113612) は 3項目FAIL — seed安定性の改善が必要 |
| R2 | P1 | baseline (202003) は positive_seed_ratio=0.5 — 2/4 seed しか利益が出ていない |
| R3 | P2 | env の 100件切り詰め自体を修正すべきか (training時のメモリ vs 情報精度のトレードオフ) |
| R4 | P2 | G3 ステップ単位蓄積のメモリ: 243K floats × 3 ≈ 5.8MB — OOS評価時のみなので問題なし |
| R5 | P3 | `avg_fee_per_trade` が maker 0% 前提で hardcode — taker 手数料対応が将来必要 |
