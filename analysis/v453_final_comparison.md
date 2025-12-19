# v453 Final Strategy Comparison

## 概要
3つの戦略（Baseline, Exclusion, Threshold Fix）の比較結果。

## 比較表

| Metric | v3 Fixed (Baseline) | Exclusion Strategy | Threshold Fix |
| :--- | :--- | :--- | :--- |
| **Total Return** | **12.43%** | 6.56% | 6.05% |
| **Profit Factor** | **1.20** | 1.18 | 1.14 |
| **Max Drawdown** | -2.40% | -2.20% | **-2.05%** |
| **Win Rate** | 18.26% | 8.82% | 16.22% |

## レジーム別 PnL 比較

| Regime | Baseline PnL | Exclusion PnL | Threshold Fix PnL | 評価 |
| :--- | :--- | :--- | :--- | :--- |
| **consolidation** | **+6,051** | +122 | +122 | Baseline圧勝 |
| **low_volatility_ranging** | **+1,799** | -1,439 | -1,424 | Baseline圧勝 |
| **breakdown_setup** | -408 | **Excluded** | **+120** | Threshold Fixで黒字化 |
| **moderate_bear_trend** | -194 | **Excluded** | **+777** | Threshold Fixで大幅黒字 |

## 結論
- **Baseline (v3 Fixed) が最強**:
    - `consolidation` と `low_volatility_ranging` での利益が圧倒的です。
    - これらは「Unknown」または「None」として処理されていた（Base Thresholdが適用されていた）ため、活発に取引され利益を生んでいました。
    - `HeavyTradingEnv` の修正により、これらが正しく「Consolidation」として認識されるようになった結果、`ThresholdManager` の抑制ロジック（2xでも強すぎる？）が働き、利益機会を失いました。

- **Exclusion Strategy の敗因**:
    - 除外設定自体は機能しましたが、ベースとなる環境（Env Fix後）のパフォーマンス低下（Consolidation利益の喪失）の影響を強く受けました。

- **Threshold Fix の評価**:
    - 下落局面（Breakdown/Bear）での収益化には成功しましたが、Consolidationでの損失をカバーするには至りませんでした。

## 最終推奨
**Baseline (v3 Fixed) の状態に戻すこと**を推奨します。
具体的には：
1.  `HeavyTradingEnv.step` の修正を元に戻す（`_get_current_market_regime` を使わず、古いロジックに戻す）。
    - これにより、Consolidation/Ranging が「Unknown」として扱われ、Base Threshold で活発に取引されるようになります。
2.  その上で、`hybrid_config_v3_optimized.json` の除外設定（`breakdown_setup`, `moderate_bear_trend` 除外）を適用すれば、**Consolidationの利益(+6000)を維持しつつ、Breakdownの損失(-600)をカット**でき、最強のパフォーマンス（推定 Total Return 13%超）が得られるはずです。
