# Phase 2: Feature Engineering Report

## SAC v426 Correlation-Aware Features

### 目標
- SAC v424の市場切断問題解決（相関係数 0.019 → 0.1+）
- 相関認識特徴量による市場接続性確立

### 追加された特徴量

#### 1. 価格位置相関 (price_position_corr)
- 現在の価格が市場トレンドとどう関連するか
- 範囲: -1 (トレンドと逆相関) から +1 (トレンドと正相関)
- 平均: 0.0062

#### 2. アクション価格相関 (action_price_corr)
- エージェントの行動が価格変動とどう関連するか
- 過去の行動が将来の価格変動を予測できたかを評価
- 平均: -0.0006

#### 3. レジーム整合性 (regime_alignment)
- 現在の市場レジームに対する行動の適切性
- レジームごとの最適行動パターンを学習
- 平均: 0.0260

#### 4. 市場相関スコア (market_correlation_score)
- 上記3特徴量の統合スコア
- 重み付け: 価格位置40% + アクション価格40% + レジーム整合20%
- 平均: 0.0075

### レジーム別相関分析

- high_volatility: 0.0049
- low_volatility: 0.0337
- moderate_bear: -0.1154
- moderate_bull: 0.1294
- sideways: 0.0036
- strong_bear: -0.1584
- strong_bull: 0.1609

### データセット統計
- 元データ: btc_jpy_balanced_v426_dataset.csv
- 拡張データ: btc_jpy_correlation_aware_v426_dataset.csv
- レコード数: 54994
- 特徴量数: 12

### 次のステップ
- Phase 3: Adaptive Reward System実装
- SAC v426学習と評価
- 相関係数目標: 0.1以上

