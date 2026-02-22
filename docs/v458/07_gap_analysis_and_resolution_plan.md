# v458 Phase 5.2/5.3: Gaps, Code Review, and Remediation Plan (Doc 07)

対象:  
`docs/v458/04_challenges_and_next_steps.md`  
`docs/v458/05_refactor_and_reuse_plan.md`  
`docs/v458/06_walk_forward_analysis_results.md`

## 1. Doc04/Doc05 で未達の項目（抜粋）

### 評価・再現性
- Walk-Forwardの**3ウィンドウ以上**（Doc05）: 2ウィンドウ止まり。
- **複数seed検証**（Doc04/Doc05）: 2-4 seedsの統計検証が未実施。
- **baseline比較の標準化**（Doc05）: buy/hold, flat, shortの統一比較が未完。
- **評価指標の統一（PF/Expectancy/Trades/Day）**（Doc05）: Walk-Forward評価に未反映。

### 最適化・探索
- **ABテスト / ハイパラ探索**（Doc05）: `tools/ab_test_runner.py` 未活用。
- **2M steps での本学習**（Doc04）: 短期学習の延長段階が未完。

### 運用系
- **Paper trading統合**（Doc05/Doc04）: 実運用評価の導線が未整備。
- **リスク管理の統一**（Doc05）: backtest/paper/live で基準統一が未完。
- **Calibration Gate / Integrated Entry**（Doc05）: v455資産が未統合。

## 2. Code Review: 重大な課題（修正必須）

### A. Walk-Forward評価の「堅牢」判定が誤解を生む
- `ztb/evaluation/walk_forward/reporter.py` の堅牢性は **overfitting_ratioのみ**で判定。  
  ROI/Sharpeが負でも「ROBUST」になる。  
  Doc06の `Status: ROBUST` は**性能の良さを意味しない**。

### B. Walk-Forward分割の実装が二重化
- `scripts/v456/phase4/modules/splitter.py` は embargo を使わず、リークチェックも無し。  
- 一方 `ztb/evaluation/walk_forward/splitter.py` は embargo とリーク検証を持つ。  
  **古い splitter を使うと評価の信頼性が落ちる**。

### C. Trade計測が壊れている可能性
- `ztb/evaluation/walk_forward/evaluator.py` は `info["trade_executed"]` を参照。  
  しかし `FastIntradayEnvV456` は `trade_executed` を返さないため、  
  **trades/win_rate が常に 0 になる**。

### D. 評価が「全期間」ではなく「ランダム区間」になっている
- `FastIntradayEnvV456.reset()` は **ランダム開始**。  
  Walk-Forward評価で **val/testの全期間を評価していない可能性**がある。  
  seed固定や `max_steps` の指定が必要。

### E. v458 config が評価に反映されている保証が弱い
- `ztb/evaluation/walk_forward/evaluator.py` のデフォルトは v456環境。  
  v458の `action_space_type` や `guidance_decay_steps` が**無視されるリスク**。

### F. 旧Phase4 evaluatorの不整合
- `scripts/v456/phase4/modules/evaluator.py` は  
  `BacktestStatsRecorder` を import するが **実体が存在しない**。  
  実行不能または trade統計が破綻する。

## 3. Doc06結果の解釈で注意すべき点
- ROI/Sharpeが負でも「ROBUST」になるのは **指標定義の問題**。  
  **“堅牢” = “負けが安定”** になっている可能性。
- Win Rate 0% と Sell偏重 (99.7%) は、  
  **trade検出ロジックの未配線**か**報酬/閾値のバグ**を疑うべき。  
  「Lost Alpha失敗」の結論は、まずログと計測整備が必要。

## 4. 解決方針（優先度順）

### Phase 5.2a: 評価パイプライン修正（最優先）
1) Walk-Forwardは `ztb/evaluation/walk_forward/*` に統一  
2) v458 env_factory を明示注入  
3) `reset(seed=...)` と `max_steps=len(segment)` を指定し、全期間評価  
4) trades/win_rate は **positionの差分**で計測  
5) 堅牢性判定に ROI/PF/Sharpe の閾値を追加

### Phase 5.2b: 再現性と探索
1) 4 seeds (42/123/777/999) を **同条件で再評価**  
2) `tools/ab_test_runner.py` で  
   `guidance_decay_steps`, `reward_clip`, `cooldown_steps` を絞り込み

### Phase 5.2c: Sell偏重の根因調査
1) Ichimokuシグナル分布をログ出力  
2) `trend_alignment` と reward の相関を検証  
3) reward_scale/clip の飽和を可視化

### Phase 5.3: 既存資産の統合
1) `ztb/trading/signal/entry_system.py`（Calibration Gate）を導入  
2) `ztb/trading/live/simulation/paper_trader.py` で実運用評価  
3) `ztb/trading/production/risk_based_allocator.py` + circuit_breaker で  
   **risk管理を共通基準化**

## 5. Go / No-Go 条件（再定義）
- Walk-Forward 全ウィンドウで **Average Test ROI > 0**  
- Profit Factor > 1.05  
- trades/day が目標帯（50–300）に入る  
- 複数seedで中央値が正の結果

---
参照:
- `ztb/evaluation/walk_forward/splitter.py`
- `ztb/evaluation/walk_forward/evaluator.py`
- `scripts/v456/phase4/modules/evaluator.py`
- `ztb/trading/environment/fast_intraday_env_v456.py`
- `docs/v457/20_v458_grid_search_review.md`
