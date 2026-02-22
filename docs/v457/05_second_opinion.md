# Second Opinion Report: Recommended Assets for v457

## 1. 復活推奨のロジック・特徴量 (Lost Alpha)
- **Dynamic Thresholds (Z-Score / Volatility)**:
  - **Source**: `config/v450/base/config.yaml`, `ztb/trading/environment/components/threshold_manager.py`
  - **Reason**: 行動閾値を市場ボラや出力分布で調整でき、HOLDロックの緩和と過剰取引の両面で効きやすい。
- **v453 Hybrid Filters (Volatility/Regime Gate)**:
  - **Source**: `config/v453/hybrid_config_v3.json`, `scripts/v453/run_backtest_v453.py`
  - **Reason**: `backtest_results/v453_hybrid_v3/backtest_results.json` で高い総リターン/Sharpeを記録。
- **ZScore + ReturnStdDev + Kalman_Residual_Norm**:
  - **Source**: `config/v445/sac_v445.2_aggressive_performance_optimized.json`, `config/feature_sets.yaml`, `ztb/features/generators/technical/volatility/kalman_ext.py`, `ztb/features/volatility/zscore.py`
  - **Reason**: 旧最適化セットで継続採用されていたシグナル群。v457では未使用。
- **ExecutionModel (Realistic/PseudoHFT)**:
  - **Source**: `config/v451/sac_v451_optimized.json`, `ztb/trading/execution/realistic.py`, `ztb/trading/execution/pseudo_hft.py`
  - **Reason**: スリッページ・レイテンシを取り込み、過学習的なPnLを抑える。

## 2. 参照すべき成功コンフィグ
- **Version/Context**: v453 Hybrid v3
  - **Key Variables**: `config/v453/hybrid_config_v3.json`, `config/v452/threshold_optimized.json`（regime multiplier）、`adaptive_threshold_mode=True`（`scripts/v453/run_backtest_v453.py`）
  - **Performance**: total_return=0.1243, sharpe_ratio=18.02, max_drawdown=-0.024, total_pnl=24,853 JPY (`backtest_results/v453_hybrid_v3/backtest_results.json`)
- **Version/Context**: v453 Hybrid Final Solution
  - **Key Variables**: `config/v453/hybrid_config_v3.json`, `config/v452/threshold_optimized.json`
  - **Performance**: total_return=0.0870, sharpe_ratio=13.14, max_drawdown=-0.030 (`backtest_results/v453_hybrid_final_solution/backtest_results.json`)
- **Version/Context**: v451 Regime-aware baseline
  - **Key Variables**: gamma=0.80, loss_multiplier=1.2, execution_model base_slippage=0.0001/atr_slippage_factor=0.1 (`config/v451/sac_v451_optimized.json`)
  - **Performance**: return_pct=1.99%, action distributionの偏りは小さめ (`backtest_results/v451/backtest_results.json`)

## 3. 有用ツール・スクリプト
- `scripts/v456/verify_fixes_v2.py`: config読み込み/報酬設定/因果特徴のサニティチェック。
- `scripts/v456/diagnose_env.py`: 環境の次元・特徴量欠損を早期検出。
- `scripts/v456/monitor_training.py`: 学習の健全性監視（崩壊の早期検知）。
- `scripts/v456/convergence_analyzer.py`: 収束傾向の定量レポート化。
- `scripts/v456/feature_calculator_v456.py`: v456互換のMTF特徴量計算。

## 4. 警告 (Avoid these)
- **[High]** レジーム閾値の過度な緩和は取引過多と手数料負けで破産リスク（`analysis/backtest_v451_optimization_report.md`）。
- **[High]** v456 FastIntradayEnvは88次元前提。生OHLCVやダミーデータでは環境初期化で落ちる（`ztb/trading/environment/fast_intraday_env_v456.py`, `scripts/v457/train.py`）。
- **[Medium]** v457のconfig読み込み箇所が `training.environment.config` 参照になっており、実際の設定が無視される可能性（`scripts/v457/train.py`）。
- **[Medium]** ExecutionModel無しのバックテストは過大評価になりやすい（`config/v451/sac_v451_optimized.json`, `ztb/trading/execution/realistic.py`）。
