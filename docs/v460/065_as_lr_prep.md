# 065# AS-LR SkipGate 学習・検証レポート

**Date**: 2026-02-17 22:33
**Fill Records**: 254 samples
**Features**: 16 features
**AS Threshold**: 0.52
**Model Path**: `models\v460\skip_gate_as.pkl`

## 1. Walk-Forward 検証結果

- **Folds**: 10
- **ROC-AUC (mean)**: 0.45005946830946836
- **ROC-AUC (std)**: 0.130253723181752
- **PR-AUC (mean)**: 0.6016477554666981
- **Brier (mean)**: 0.2546977697871896

## 2. Skip Simulation (OOF)

- **Baseline PnL**: -0.839 bps
- **Skip 20% 改善**: +0.269 bps
- **Skip 10% 改善**: -0.085 bps
- **Valid samples**: 200

## 3. Feature Stability

- **Jaccard stability**: 0.267
- **Always selected (4)**: ['buy_ratio', 'side_aligned_velocity', 'trade_count_60s', 'trade_flow_imbalance_60s']
- **Ever selected (15)**: ['avg_trade_size', 'buy_ratio', 'hour_cos', 'hour_sin', 'offset_ratio', 'price_velocity_60s', 'regime_ranging', 'regime_trending', 'side_aligned_tfi', 'side_aligned_velocity', 'side_buy', 'spread_jpy', 'trade_count_60s', 'trade_flow_imbalance_60s', 'vpin_60s']

## 4. Per-Fold Results

| Fold | Train | Test | ROC-AUC | PR-AUC | Brier | AS rate (test) | Selected Features |
|---|---|---|---|---|---|---|---|
| 0 | 50 | 20 | 0.4615 | 0.6500 | 0.2471 | 0.650 | side_buy, hour_cos, spread_jpy, trade_count_60s... |
| 1 | 70 | 20 | 0.5000 | 0.6000 | 0.2482 | 0.600 | side_buy, offset_ratio, trade_count_60s, buy_ratio... |
| 2 | 90 | 20 | 0.3438 | 0.5392 | 0.2540 | 0.600 | side_buy, hour_cos, regime_trending, regime_ranging... |
| 3 | 110 | 20 | 0.4600 | 0.5763 | 0.2521 | 0.500 | side_buy, hour_cos, spread_jpy, regime_trending... |
| 4 | 130 | 20 | 0.2418 | 0.5599 | 0.2610 | 0.650 | side_buy, hour_cos, regime_trending, regime_ranging... |
| 5 | 150 | 20 | 0.6000 | 0.6705 | 0.2599 | 0.500 | side_buy, hour_cos, offset_ratio, trade_count_60s... |
| 6 | 170 | 20 | 0.6465 | 0.6444 | 0.2433 | 0.550 | side_buy, hour_cos, spread_jpy, offset_ratio... |
| 7 | 190 | 20 | 0.3200 | 0.4174 | 0.2610 | 0.500 | side_buy, hour_cos, spread_jpy, offset_ratio... |
| 8 | 210 | 20 | 0.3333 | 0.6491 | 0.2759 | 0.700 | hour_sin, hour_cos, spread_jpy, offset_ratio... |
| 9 | 230 | 20 | 0.5938 | 0.7097 | 0.2445 | 0.600 | hour_cos, spread_jpy, trade_count_60s, buy_ratio... |

## 5. 学習済みモデル情報

- **Total samples**: 254
- **AS rate**: 0.571
- **Selected features**: ['side_buy', 'hour_cos', 'spread_jpy', 'trade_count_60s', 'buy_ratio', 'trade_flow_imbalance_60s', 'avg_trade_size', 'vpin_60s', 'side_aligned_tfi', 'side_aligned_velocity']

### Feature Importances (LR coefficient abs)

| Feature | Importance |
|---|---|
| buy_ratio | 0.0630 |
| trade_flow_imbalance_60s | 0.0630 |
| hour_cos | 0.0451 |
| spread_jpy | 0.0368 |
| side_aligned_tfi | 0.0346 |
| side_aligned_velocity | 0.0284 |
| avg_trade_size | 0.0278 |
| side_buy | 0.0217 |
| trade_count_60s | 0.0126 |
| vpin_60s | 0.0123 |

## 6. ph2 投入設定

```yaml
# configs/v460/fill_test.yaml skip_gate section
skip_gate:
  enabled: true
  mode: as
  model_path: models\v460\skip_gate_as.pkl
  as_threshold: 0.52
  max_skip_rate: 0.3
```

### 判定基準 (200 cycle 評価)

| 指標 | 継続条件 | 中止条件 |
|---|---|---|
| post_fill_30s_pnl mean | baseline 比改善 | baseline 以下 |
| AS ratio | baseline 比低下 | 増加 |
| fill rate | 劣化軽微 | 大幅悪化 |
| skip rate | 設定範囲内 | 上限張り付き |