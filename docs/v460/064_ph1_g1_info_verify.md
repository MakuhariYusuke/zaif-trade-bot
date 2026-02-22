# 064# ph1 G1-info 再検証結果

**Phase**: ph1 (検証フェーズ)
**Date**: 2026-02-15
**Data**: 1755 rows, 10 features, 3 days (2/13-2/15)

## 1. Raw Feature IC (Spearman rank correlation)

| Feature | h1 IC | h5 IC | h15 IC |
|---|---|---|---|
| bid_ask_spread | +0.0273 | +0.0219 | +0.0222 |
| depth_imbalance | +0.0163 | -0.0082 | +0.0154 |
| trade_flow_imbalance | -0.0013 | +0.0243 | -0.0080 |
| vwap_deviation | -0.0901*** | -0.0510** | -0.0274 |
| trade_intensity | +0.0001 | -0.0021 | +0.0194 |
| order_flow_toxicity | -0.0077 | -0.0037 | -0.0672*** |
| price_impact | -0.0103 | -0.0174 | -0.0162 |
| micro_return_vol | +0.0129 | +0.0147 | +0.0438 |
| bid_depth_slope | -0.0222 | -0.0452 | -0.0243 |
| ask_depth_slope | -0.0523** | -0.0387 | -0.0242 |

## 2. Walk-Forward Results

| Target | Model | Acc/MAE | IC_mean | IC_sig | Folds |
|---|---|---|---|---|---|
| target_direction_h1 | XGBoost | acc=0.4900 | -0.0536 | 0/3 | 3 |
| target_direction_h1 | Baseline | acc=0.5043 | +0.0113 | 0/3 | 3 |
| target_direction_h15 | XGBoost | acc=0.4713 | +0.1459 | 1/3 | 3 |
| target_direction_h15 | Baseline | acc=0.3994 | +0.0356 | 0/3 | 3 |
| target_direction_h5 | XGBoost | acc=0.5014 | +0.1088 | 1/3 | 3 |
| target_direction_h5 | Baseline | acc=0.4444 | -0.0340 | 1/3 | 3 |
| target_magnitude_h1 | XGBoost | mae=0.000424 | +0.0721 | 0/3 | 3 |
| target_magnitude_h1 | Baseline | mae=0.000283 | +0.0289 | 1/3 | 3 |
| target_magnitude_h15 | XGBoost | mae=0.001111 | -0.0048 | 1/3 | 3 |
| target_magnitude_h15 | Baseline | mae=0.001181 | -0.1119 | 0/3 | 3 |
| target_magnitude_h5 | XGBoost | mae=0.000607 | +0.1409 | 0/3 | 3 |
| target_magnitude_h5 | Baseline | mae=0.000679 | +0.1493 | 1/3 | 3 |
| target_volatility_h1 | XGBoost | mae=nan | +0.0000 | 0/0 | 0 |
| target_volatility_h1 | Baseline | mae=nan | +0.0000 | 0/0 | 0 |
| target_volatility_h15 | XGBoost | mae=0.000182 | +0.2825 | 1/3 | 3 |
| target_volatility_h15 | Baseline | mae=0.000196 | +0.0688 | 3/3 | 3 |
| target_volatility_h5 | XGBoost | mae=0.000213 | +0.0811 | 2/3 | 3 |
| target_volatility_h5 | Baseline | mae=0.000223 | +0.1568 | 2/3 | 3 |

## 3. G1-info 判定

- **Direction IC 平均**: +0.067070
- **G1-info 基準 (|IC| > 0.02)**: **PASS**

> Microstructure features are informative. Proceed to SkipGate live integration.

## 4. 分析所見

### 有力特徴量

1. **vwap_deviation** (IC=-0.090, p<0.001 at h1): 最強シグナル。VWAP乖離は短期逆張りシグナル。
2. **order_flow_toxicity** (IC=-0.067, p<0.005 at h15): 中期で有効。VPIN近似が流動性リスクを捕捉。
3. **ask_depth_slope** (IC=-0.052, p<0.03 at h1): 売り板勾配が短期方向性を示唆。

### XGBoost vs Baseline

- h1方向: XGBoostがbaseline負け (IC -0.054 vs +0.011) → 過学習の兆候
- h5方向: XGBoost勝ち (IC +0.109 vs -0.034) → 5分horizonでモデルが有効
- h15方向: XGBoost勝ち (IC +0.146 vs +0.036) → 15分horizonでも有効
- volatility h15: XGBoost IC=+0.283 (strong) — ボラ予測力が高い

### 注意点

- 1,755行 (3日) は検証としては最小限 — 更なるデータ蓄積で再検証必要
- volatility h1 は rolling(1).std() = NaN で評価不能 (h≧5で使用推奨)
- h1方向のXGBoost過学習はfold数不足が要因の可能性

### SkipGate 推奨設定

上記を踏まえ、SkipGate AS mode の推奨:
- **特徴量**: vwap_deviation, order_flow_toxicity, ask_depth_slope を重視
- **horizon**: h5 (5分先) を主ターゲット — IC/安定性のバランス最良
- **閾値**: 低め (conservative) から開始 — false skip を避ける