# 065# AS-LR SkipGate 学習・検証レポート

**Date**: 2026-02-17 16:25
**Fill Records**: 215 samples
**Features**: 16 features
**AS Threshold**: 0.65
**Model Path**: `models\v460\skip_gate_as.pkl`

## 1. Walk-Forward 検証結果

- **Folds**: 8
- **ROC-AUC (mean)**: 0.44163841366966367
- **ROC-AUC (std)**: 0.1199908393517083
- **PR-AUC (mean)**: 0.5780678552604024
- **Brier (mean)**: 0.2533233232781213

## 2. Skip Simulation (OOF)

- **Baseline PnL**: -0.781 bps
- **Skip 20% 改善**: +0.405 bps
- **Skip 10% 改善**: -0.027 bps
- **Valid samples**: 160

## 3. Feature Stability

- **Jaccard stability**: 0.357
- **Always selected (5)**: ['buy_ratio', 'side_aligned_velocity', 'side_buy', 'trade_count_60s', 'trade_flow_imbalance_60s']
- **Ever selected (14)**: ['avg_trade_size', 'buy_ratio', 'hour_cos', 'offset_ratio', 'price_velocity_60s', 'regime_ranging', 'regime_trending', 'side_aligned_tfi', 'side_aligned_velocity', 'side_buy', 'spread_jpy', 'trade_count_60s', 'trade_flow_imbalance_60s', 'vpin_60s']

## 4. Per-Fold Results

| Fold | Train | Test | ROC-AUC | PR-AUC | Brier | AS rate (test) | Selected Features |
|---|---|---|---|---|---|---|---|
| 0 | 50 | 20 | 0.4615 | 0.6500 | 0.2464 | 0.650 | side_buy, hour_cos, spread_jpy, trade_count_60s... |
| 1 | 70 | 20 | 0.5000 | 0.6000 | 0.2482 | 0.600 | side_buy, offset_ratio, trade_count_60s, buy_ratio... |
| 2 | 90 | 20 | 0.3438 | 0.5392 | 0.2538 | 0.600 | side_buy, hour_cos, regime_ranging, trade_count_60s... |
| 3 | 110 | 20 | 0.4600 | 0.5763 | 0.2521 | 0.500 | side_buy, hour_cos, spread_jpy, regime_ranging... |
| 4 | 130 | 20 | 0.2418 | 0.5599 | 0.2609 | 0.650 | side_buy, hour_cos, regime_trending, regime_ranging... |
| 5 | 150 | 20 | 0.5900 | 0.6679 | 0.2602 | 0.500 | side_buy, hour_cos, offset_ratio, trade_count_60s... |
| 6 | 170 | 20 | 0.6061 | 0.6130 | 0.2441 | 0.550 | side_buy, hour_cos, spread_jpy, offset_ratio... |
| 7 | 190 | 20 | 0.3300 | 0.4184 | 0.2610 | 0.500 | side_buy, hour_cos, spread_jpy, offset_ratio... |

## 5. 学習済みモデル情報

- **Total samples**: 215
- **AS rate**: 0.558
- **Selected features**: ['hour_sin', 'hour_cos', 'spread_jpy', 'offset_ratio', 'trade_count_60s', 'buy_ratio', 'trade_flow_imbalance_60s', 'avg_trade_size', 'price_velocity_60s', 'side_aligned_velocity']

### Feature Importances (LR coefficient abs)

| Feature | Importance |
|---|---|
| spread_jpy | 0.0494 |
| avg_trade_size | 0.0484 |
| offset_ratio | 0.0455 |
| trade_count_60s | 0.0440 |
| hour_cos | 0.0406 |
| side_aligned_velocity | 0.0354 |
| buy_ratio | 0.0327 |
| trade_flow_imbalance_60s | 0.0327 |
| price_velocity_60s | 0.0298 |
| hour_sin | 0.0053 |

## 6. ph2 投入設定

```yaml
# configs/v460/fill_test.yaml skip_gate section
skip_gate:
  enabled: true
  mode: as
  model_path: models\v460\skip_gate_as.pkl
  as_threshold: 0.65
  max_skip_rate: 0.3
```

### 判定基準 (200 cycle 評価)

| 指標 | 継続条件 | 中止条件 |
|---|---|---|
| post_fill_30s_pnl mean | baseline 比改善 | baseline 以下 |
| AS ratio | baseline 比低下 | 増加 |
| fill rate | 劣化軽微 | 大幅悪化 |
| skip rate | 設定範囲内 | 上限張り付き |