# 388# G3-pnl 準備計画

## 概要

G2-train PASS (387# reward-tuned 実験) を受け、000# §3.5 G3-pnl Gate の通過に向けた準備計画。
本文書では G3 要件・現状ギャップ・実施計画を定義する。

## G3 Gate 要件 (000# §3.5)

| 条件 | 閾値 | 測定方法 |
|------|------|---------|
| Profit Factor (median) | > 1.05 | maker 0% 前提、全 seed の中央値 |
| Profit Factor (worst-seed) | > 0.95 | 最低 seed でも致命的損失を回避 |
| avg gross/trade > avg fee/trade | true | 取引あたり収益 > 取引あたり費用（pooled） |
| Max Drawdown | < 15% | equity curve の最大下落（worst-seed） |
| Sharpe (年率) | > 0.8 | 日次リターン（median） |

## 現状 (G2 結果からの推定)

### G2 reward-tuned 実験結果

| Seed | OOS ROI | G2 判定 |
|------|:-------:|:-------:|
| 42 | +0.01% | PASS |
| 123 | +0.39% | PASS |
| 456 | -3.21% | PASS (E4=-3.5%) |
| 789 | +4.09% | PASS |
| **Mean** | **+0.32%** | **PASS** |

### G3 との距離感

G2 はOOS ROI（粗利）で判定。G3 ではコスト込み実収益性が問われる。

| G3 指標 | 現在推定 | 判定 | 備考 |
|---------|---------|:----:|------|
| Profit Factor (median) | 不明 | ❓ | 取引単位の損益データが G2 では未集計 |
| Profit Factor (worst-seed) | 不明 | ❓ | seed 456 が懸念 (OOS -3.21%) |
| avg gross > avg fee | maker 0% 前提なら YES | ⚠️ | fee=0 なら自明だが、taker 混在時は不明 |
| Max Drawdown | 不明 | ❓ | G2 では equity curve 未出力 |
| Sharpe (年率) | 不明 | ❓ | 日次リターン未集計 |

**結論**: G2 実験は ROI しか出力しておらず、G3 判定に必要な **取引ログ・equity curve・日次リターン** が未出力。G3 評価には評価パイプラインの拡張が必要。

## 実施計画

### Phase A: G3 評価パイプライン構築

#### A-1: 評価メトリクス計算基盤の確認

既存コード:
- `ztb.training.reward_function_evaluator` — `EvaluationMetrics` (PF, Sharpe, MaxDD 計算済み)
- `ztb.trading.comprehensive_backtest` — `BacktestResult` (PF, Sharpe, MaxDD, equity_curve)
- `ztb.analysis.backtest.analyze_backtest` — `BacktestAnalyzer`

**タスク**: 既存の `EvaluationMetrics` を G3 Gate 判定に直接利用できるか検証。
不足している場合は G3 Gate 判定専用ラッパーを作成。

#### A-2: OOS 評価時の trade-level ログ出力

G2 実験で使用している `scripts/v460/lib/tasks/sac_train.py` の `evaluate_model()` で、
以下のデータを出力するよう拡張:

1. **取引ログ** (trade_log): 各取引の timestamp, action, price, pnl, fee
2. **Equity curve**: ステップ毎の累積残高
3. **日次リターン**: 日別の返率集計

#### A-3: G3 Gate 判定スクリプト

```python
# scripts/v460/lib/tasks/g3_gate_check.py (新規)
# 入力: 4 seed の OOS trade_log + equity_curve
# 出力: G3 Gate 判定結果 (JSON)
```

判定ロジック:
```
for each seed:
    PF = sum(winning_pnl) / abs(sum(losing_pnl))
    MaxDD = max_drawdown(equity_curve)
    Sharpe = annualized_sharpe(daily_returns)
    avg_gross = mean(abs(pnl) for each trade)
    avg_fee = mean(fee for each trade)

G3 PASS if:
    median(PF across seeds) > 1.05
    min(PF across seeds) > 0.95
    pooled avg_gross > pooled avg_fee
    max(MaxDD across seeds) < 0.15
    median(Sharpe across seeds) > 0.8
```

### Phase B: G3 評価実験の実行

#### B-1: reward-tuned モデルの再訓練 (必要に応じて)

G2 PASS 時の reward-tuned モデルが保存されていれば再利用。
保存されていなければ再訓練 (100K steps × 4 seeds)。

YAML: `configs/v460/experiments/g2_sac_gamma095_reward_tuned.yaml`

#### B-2: OOS 評価 (trade-level ログ付き)

A-2 で拡張した評価パイプラインで 4 seed を OOS 評価。

```
seed 42:  checkpoints/g2_reward_tuned_seed42/  → oos_trades_42.json
seed 123: checkpoints/g2_reward_tuned_seed123/ → oos_trades_123.json
seed 456: checkpoints/g2_reward_tuned_seed456/ → oos_trades_456.json
seed 789: checkpoints/g2_reward_tuned_seed789/ → oos_trades_789.json
```

#### B-3: G3 Gate 判定実行

A-3 スクリプトで 4 seed の結果を統合判定。

### Phase C: G3 FAIL 時の改善オプション

G3 FAIL の場合、原因に応じた対策:

| FAIL 原因 | 対策 | 優先度 |
|----------|------|:------:|
| PF < 1.05 | reward_scaling 増加 (5.0-10.0)、ペナルティ更なる縮小 | P0 |
| MaxDD > 15% | position_size 縮小、DD ペナルティ追加 | P0 |
| Sharpe < 0.8 | 取引頻度削減 (hold_penalty 完全削除)、ボラフィルタ | P1 |
| avg gross < avg fee | maker 0% で不可能。taker 混在なら taker 排除ロジック | P1 |
| 全 seed FAIL | v461 検討 (000# §3.5 FAIL 時) | - |

## 手数料モデルの前提

000# §4 より:
> 手数料モデル: ExchangeFeeModel（取引所別）maker-only 前提。Coincheck maker 0% がデフォルト

- G3 評価は **maker 0%** で実施
- `avg gross > avg fee` は maker 0% では自明 (fee=0) だが、形式的に検証
- taker 手数料の影響は G4 (paper trading) で検証

## タイムライン

| フェーズ | 作業内容 | 想定工数 |
|---------|---------|---------|
| A-1 | 既存コード調査・流用判断 | 0.5 日 |
| A-2 | evaluate_model() trade-level ログ拡張 | 1 日 |
| A-3 | G3 Gate 判定スクリプト | 0.5 日 |
| B-1 | モデル確認 / 再訓練 | 0-1 日 |
| B-2 | OOS 評価実行 | 0.5 日 |
| B-3 | G3 Gate 判定 | 0.5 日 |
| **合計** |  | **3-4 日** |

## 依存関係

- G2 PASS 確定 (387# ✅)
- reward-tuned YAML (`g2_sac_gamma095_reward_tuned.yaml`) 確定 (387# ✅)
- P0-1〜P0-8 修正済み (385# ✅)
- YAML→env 伝播修正済み (387# P0-5 ✅)

## 次ステップ

1. **即座**: A-1 既存 `EvaluationMetrics` の G3 適合性を検証
2. **優先**: A-2 trade-level ログの evaluate_model() 統合
3. **最後**: B-2/B-3 評価実行・G3 判定
