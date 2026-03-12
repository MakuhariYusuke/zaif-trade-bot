# 18. v458 Strategy Roadmap Review (Feedback)

対象: `docs/v457/17_v458_strategy_roadmap.md`

## 総評
方向性は良く、既存資産を使う前提も妥当。ただし「分析指標の追加」「Config差し替え」「動的閾値統合」の各段で、現行スクリプトの入出力仕様と噛み合っていない箇所がある。そこを先に詰めないと、Phase 1の検証が空振りになるリスクが高い。

## 重大な齟齬・見落とし

1) **`stats.json`に取引履歴が入っていない**
   - `backtest_v456.py` は `stats.json` と `portfolio_history.csv` しか出力しない。  
   - Profit Factor / Avg Win/Loss / Expectancy は **trade_historyが無いと計算できない**。
   - `scripts/analysis/analyze_backtest_v456.py` も trade_history を読まない設計。

2) **Config変更だけでは `backtest_v456.py` に反映されない**
   - `backtest_v456.py` は `config/v458/` を読まない。  
   - Roadmapの「Config Grid Search」だけでは **min_delta/hold_ramp/max_ttl が適用されない**。

3) **取引判定が “環境” ではなく “手書きルール”**
   - `backtest_v456.py` は action > 0.3 / < -0.3 を使った手書きのエントリー/エグジット判定。  
   - `min_delta` や `cooldown_steps` の影響が **評価に反映されにくい**。

4) **成果指標の解釈が一致していない**
   - `analyze_backtest_v456.py` は `portfolio_history.csv` からシャープ等を計算するが、  
     ポートフォリオ価値は **簡易計算**で、手数料/スリッページ/実現損益の整合が弱い。

## 軽微だが影響する点
- Case A/B/Cの「max_ttl」表記と、実装パラメータ名の一致確認が必要。  
  (`FastIntradayEnvV456` は `max_ttl_steps`)
- v457モデルでv458設定を試す場合、**“学習済み方策と環境差分”の影響**を注意書きに入れるべき。
- 評価対象の期間・データファイルが未定義。再現性に影響。

## 改善案（Roadmapに組み込み推奨）

### A. Phase 1の前提整備（必須）
1) `backtest_v456.py` に **configパス指定**を追加  
   - `--config config/v458/case_a.yaml` などで `env_config` を反映。
2) `backtest_v456.py` に **trade_historyの保存**を追加  
   - `trades.json` 出力（Profit Factor 等の計算に必要）。
3) `scripts/analysis/analyze_backtest_v456.py` を **trades.json対応**に拡張  
   - Profit Factor / Avg Win/Loss / Expectancy を算出。

### B. 評価の整合性向上（推奨）
- 手書きのエントリー/エグジット判定ではなく、  
  `env` 側の**position/balance/fee**に基づく評価に寄せる。  
- 難しければ、最低限 `backtest_v456.py` の閾値 (`0.3`) を設定可能にする。

### C. Dynamic Thresholdの段階導入（Phase 2）
- v450のZ-score統合は後回しで良いが、  
  まず **ATRベースの簡易動的min_delta** を実装して効果検証。  
- 適用順: `min_delta` (固定) → `min_delta` (ATR可変) → Z-score統合

## v458 Roadmapへの具体的な追記案

1) **Step 0 (Prep)**  
   - `backtest_v456.py` に `--config` と `trades.json` 出力を追加  
   - `analyze_backtest_v456.py` に Profit Factor / Expectancy を追加
2) **Step 1 (Grid Search)**  
   - `min_delta` だけでなく `cooldown_steps` も1軸追加  
   - 3 seeds を最低ラインに
3) **Step 2 (Adaptive)**  
   - ATR動的min_deltaを最小実装  
   - Z-score統合は Phase 2.5 以降

## 結論
ロードマップは良いが、「Configを変えれば実験できる」という前提が現行実装と噛み合っていない。  
まず **Phase 1の前提整備（config入力 + trade_history出力 + analyzer拡張）** を入れてから実験に入るのが安全。
