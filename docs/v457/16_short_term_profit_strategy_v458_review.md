# 16. v458 Short-Term Profit Strategy Review

対象: `docs/v457/15_short_term_profit_strategy_v458_concept.md`

## 1. 良い方向性（現時点で妥当な点）
- Under-Tradingの認識と「回転数の回復」に焦点を当てた点は正しい。
- v450のDynamic Threshold・v437の頻度制御を参照する方針は整合的。
- `min_delta`のグリッドサーチを先に行う段取りは実装負担が低く、有効。

## 2. 見落とし・前提不足
1) **実行市場の制約（1分足・成行）を強く受ける**  
   - 1分足では短期利益の多くが「バー内変動」で消える。  
   - Micro-TPを強化すると**スリッページと手数料に埋没**しやすい。

2) **頻度目標とコストの整合が未定義**  
   - 10〜20回/日（=1440分基準で0.7〜1.4%）が目標なら、  
     **期待値/トレード（after fee）を明示**しないと方針がぶれる。

3) **動的閾値の実装先が曖昧**  
   - v450のZ-score動的閾値は heavy_env で実装済みだが、  
     fast_intraday_env_v456 系には未接続。  
   - v458で実装する場合、**ThresholdManagerを流用するか新設するか**を明記すべき。

4) **`min_delta`調整が実効値に直結しない可能性**  
   - `max_position_size` と `max_delta_per_step` によって実際の変化幅が抑制される。  
   - `min_delta`だけを動かしても「実効エントリー頻度」が変わらない場合がある。

5) **報酬と頻度のトレードオフに評価指標が不足**  
   - Win率だけでなく **avg win/loss, profit factor, expectancy** を追うべき。  
   - `edge_shortfall / trade_cost / vol_ratio` をログに出さないと、  
     「頻度が増えたが損失が増えた」原因を切り分けられない。

## 3. 具体的な改善提案

### A. 取引頻度の制御を「閾値」以外でも分散
- `min_delta`を下げると過熱しやすいので、  
  **cooldown_steps** か **max_trades_per_episode** を併用した方が安定する。  
- v437の頻度制御と併用し、**閾値依存を緩和**する。

### B. 「動的閾値」の最小実装
- v450のZ-scoreロジックをそのまま移植するより、  
  **ATRベースの簡易動的閾値**で検証 → Z-score実装の順が安全。  
- 例: `min_delta = base_min_delta * clamp(atr / atr_avg, 0.5, 2.0)`

### C. 報酬強化は「利確即時」より「コスト可視化」から
- 利確強化は過度な回転を招く可能性があるため、  
  まず **trade_cost / edge_shortfall の可視化**を優先。

### D. 評価基準（Go/No-Go）を先に設定
- 例:  
  - Trade/Day: 10〜20  
  - Profit factor: > 1.05  
  - Average trade PnL: 手数料の2倍以上  
  - MaxDD: 5%以内  
  - 3 seedsで中央値判定

## 4. 参考になるvXXX（再掲+補強）
- v450: Z-score動的閾値の仕様と実装  
  `docs/v450/01_dynamic_thresholding.md`
- v437: 取引頻度制御の具体例  
  `docs/README_v437.md`
- v455: `min_edge_mult / vol_floor`の感度分析とログ設計  
  `docs/v455/11_sensitivity_and_training_plan.md`  
  `docs/v455/12_sensitivity_results_and_training_conclusion.md`
- v452: 閾値の安全化（相対閾値）  
  `docs/v452/changes_v452.md`
- v420: HOLD封じの副作用（高頻度化で手数料負け）  
  `docs/bug_fixes/SAC_V420_HOLD_RELAXED_LEARNING_PROCESS.md`

## 5. 次の最小アクション（現実的な順序）
1) `min_delta` グリッド + 3 seeds  
2) `trade_cost / edge_shortfall` ログ出力  
3) 低リスクのATR動的閾値を追加  
4) 目標頻度に近づいたら報酬（利確強化）を微調整
