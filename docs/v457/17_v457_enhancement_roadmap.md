# 17. v457 Enhancement Roadmap (v457.1 & v457.2)

**作成日**: 2026-01-18 (Updated from v458 to v457.x)
**参照**: 
- `docs/v457/15_short_term_profit_strategy_v458_concept.md` (Original Concept)
- `docs/v457/22_v457_1_phase2_frequency_control_review.md` (Review & Feedback)

## 1. 方針の転換と確定

既存モデル (v457) のポテンシャルを最大限に引き出すための調整フェーズを **v457.1** とし、次期改善モデルに向けた厳密な評価と対策を **v457.2** と定義します。

### バージョン定義
- **v457.1 (Frequency Control)**: 既存モデルに対し、外部パラメータ（Cooldown, Threshold, Wrapper）による調整で収益化を試みるフェーズ。 (Completed: Failed to Profit)
- **v457.2 (Robustness & Evaluation)**: 評価指標の厳密化と、Action分布の偏りなど「モデルの学習上の欠陥」を特定・修正するためのフェーズ。単純な再学習 (v459) の前に、何が間違っていたかを技術的に確定させる。

## 2. 評価指標の整備 (Existing Implementation Enhancement)

**課題**: 現在の `backtest_v456.py` は `stats.json` に集計値のみを出力しており、詳細な取引ごとの履歴（Trade History）が含まれていません。そのため、Profit Factor や Expectancy の正確な事後計算が不可能な状態です。

**対策**: 以下の改修を Step 0 として実施します。
1.  **Export Enhancement**: `backtest_v456.py` を修正し、`trades.json`（全取引のリスト）を出力するように変更。
2.  **Analysis Upgrade**: `scripts/analysis/analyze_backtest_v456.py` を修正し、`trades.json` を読み込んで以下の指標を計算・表示。

*   **Profit Factor**: (Total Profit / Total Loss)
*   **Average Win / Loss**: (Total Profit / Win Count), (Total Loss / Loss Count)
*   **Expectancy (期待値)**: (Avg Win * Win Rate) - (Avg Loss * Loss Rate)
*   **Trade Cost Analysis**: 手数料とスリッページの推定累積値

## 3. 実装・実験プロセス (Phase 1: Tuning / v457.1)

**Status: Completed**
v457モデルに対し、`min_delta` や `cooldown` による制御を試みたが、Net PnLで赤字解消できず。モデルのAlpha不足が示唆された。

## 4. 動的制御の導入計画 (Phase 2: Evaluation / v457.2)

Doc 22 の指摘を受けて、単にモデルを捨てて再学習するのではなく、**「なぜ学習に失敗したか」** を特定するための詳細分析を行います。

### 重点施策
1.  **Metric Rigor**: `script/analysis` を強化し、Estimated Fee ではなく、Backtest 実行時の真の Fee/Slippage を使った正確な PnL を出す。
2.  **Action Distribution Analysis**: モデルの出力分布 (`action_mean`, `action_std`) を記録し、"Bang-Bang Control" (常に±1張り付き) になっていないか確認。
3.  **Entropy Control Plan**: もし張り付いているなら、次期学習の `ent_coef` 戦略に直結させる。

## 5. タスクリスト

1.  [x] **Backtest Script Fix (Step 0)**: `backtest_v456.py` を改修し、`--config` 対応と `trades.json` 出力を追加。
2.  [x] **Analysis Script Enhancement (Step 0)**: Profit Factor 表示追加。
3.  [x] **Grid Search (v457.1)**: Case A-F 実施完了。
4.  [ ] **Evaluation Rigor (v457.2)**: 
    - `backtest_v456.py` で `fee` と `slippage` を各トレードごとに正確に記録する。
    - `analyze_backtest_v456.py` で正確な Profit Factor と Expectancy を出す。
    - Action Distribution の可視化。

---
本ロードマップに従い、**タスク4（Evaluation Rigor）** へ移行します。

