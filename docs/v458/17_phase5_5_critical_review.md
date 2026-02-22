# Phase 5.5 Critical Review (Doc 17)

**Date**: 2026-01-22  
**Scope**: Doc14 実装レビュー + Doc16 以外の発見事項  
**Tone**: 厳しめ／現場止め基準で評価

---

## Executive Summary
**総合スコア: 3/10（No-Go）**  
「動いているように見える」だけで、**評価の正確性が担保されていない**。  
trade 統計・entry gate・baseline 比較・seed 評価の設計が崩れており、  
**現時点での結論や優劣判定は無効**。

---

## Strengths（少数だが評価できる点）
- `reset(options={"start_step": ...})` により評価開始位置を固定できるようにした点は良い方向。  
- `close_prices` を保持して `df` を解放する方向性はメモリ面で正しい。  
- entry gate の配線自体は導入されており、将来的な拡張余地はある。

---

## Critical Issues（ブロッカー）

1) **trade_pnl が常時0で統計が破綻**  
   - `trade_pnl = 0.0` が固定化されており、評価側が `trade_pnl` を採用するため  
     **PF/Expectancy が虚偽の値になる**。  
   - `ztb/trading/environment/fast_intraday_env_v456.py:490`  
   - `ztb/evaluation/walk_forward/evaluator.py:470`

2) **entry gate が常時ブロック状態になる**  
   - gateに渡す `MarketState` が `high/low/close/atr/volume` を満たさず、  
     `volume=0` により **コスト推定が常に inf** になる。  
   - `ztb/trading/environment/fast_intraday_env_v456.py:472`  
   - `ztb/trading/signal/calibration_map.py:285`  
   - `ztb/trading/signal/calibration_map.py:297`

3) **entry gate が例外で落ちる可能性**  
   - `self.regime_data` が未定義。gate有効時に `AttributeError` で崩壊する。  
   - `ztb/trading/environment/fast_intraday_env_v456.py:476`

4) **baseline 比較が実際には動いていない**  
   - pipeline 側で `price_data` が未定義、`evaluation_result` も未セット。  
   - `scripts/v458/run_walk_forward_v458.py:253`  
   - `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:337`

5) **「ロバスト判定」が再び過学習比率のみになっている**  
   - ROI / Sharpe / PF の要件が消え、負け続けでも ROBUST 扱いになる。  
   - `ztb/evaluation/walk_forward/types.py:321`

---

## Detailed Analysis by Perspective（要点のみ）

### 1) Technical Accuracy
- entry/exit price が **次ステップの close** を参照しており、ズレが出る。  
  `ztb/evaluation/walk_forward/evaluator.py:475`  
- `pnl -= fee + slippage` の後に `BacktestReporter` で再度控除され、  
  **コスト二重計上**になる。  
  `ztb/evaluation/walk_forward/evaluator.py:472`  
  `ztb/evaluation/walk_forward/reporter.py:215`

### 2) Architectural Design
- `BacktestReporter` が `scripts/v457` と `ztb/evaluation` の二重実装になっており、  
  **責務と依存が発散**している。  
  `ztb/evaluation/walk_forward/evaluator.py:51`  
  `ztb/evaluation/walk_forward/evaluator.py:425`

### 3) Performance & Scalability
- multi-seed 評価は **window×seed の直列実行**のみで、  
  結果の集計も seed 単位で整理されていない。  
  `scripts/v458/run_walk_forward_v458.py:83`

### 4) Robustness & Security
- baseline比較は例外を握りつぶして warning になるだけで、  
  **失敗しても成功と見えるログ設計**。  
  `scripts/v458/run_walk_forward_v458.py:247`

### 5) Code Quality
- `compare_with_baselines` は baseline 名が一致しない（buy_and_hold vs buy_hold）。  
  `ztb/analysis/evaluation/walk_forward_integration_pipeline.py:317`  
  `ztb/analysis/baseline_comparison.py:158`

---

## Improvement Opportunities（優先度順）

### P0: 評価の正確性
1) trade_pnl の再設計（PositionManager を正しく統合するか、実取引で確定計算）  
2) entry gate 用 MarketState に **high/low/close/atr/volume** を供給  
3) `is_robust_model` を ROI/Sharpe/PF 条件に戻す

### P1: baseline / seed / AB の正当化
1) `walk_forward_integration_pipeline.py` の `price_data` 未定義を修正  
2) baseline 名の整合を取る（buy_hold に統一）  
3) seed 評価を window×seed で **集計レイヤを分離**

### P2: 重複削減
1) BacktestReporter を **1箇所に統一**  
2) Doc14 で新規実装を作らず、`tools/ab_test_runner.py` を本流にする

---

## Remaining Gaps（Doc11以前から未達）
- A/B テストは既存の `tools/ab_test_runner.py` を使わず再発明している  
- baseline 比較は pipeline 側が壊れており未実施  
- entry gate は配線されたが **動作不全**

---

## Conclusion
現状は **No-Go**。  
trade 統計と entry gate が破綻している以上、  
評価結果の信頼性は無く、次フェーズ判断は危険。  
まず **P0修正（trade/entry gate/robust判定）** を完了させ、  
その後に baseline/seed/AB を再検証すべき。
