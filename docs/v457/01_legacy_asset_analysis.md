# Asset Discovery & Refactoring Plan (Legacy Analysis)

## 1. 目的
v457の開発にあたり、過去のバージョン（v451, v444等）から「有効だった要素（Assets）」を発掘し、逆に「ボトルネックとなっていた複雑性」を特定する。
これを元にv457の設計方針（Config, Reward, Features）を決定する。

## 2. 発掘された資産 (Legacy Assets)

### 2.1 The "Golden Era" Config (v451)
`config/v451/sac_v451_optimized.json` が最も有望なベースラインであることが判明。

| Setting | Value | Analysis |
|:---|:---|:---|
| **Algorithm** | SAC | PPOではなくSACが採用されていた時期の収益性が高かった可能性。 |
| **Gamma (Discount)** | **0.80** | 一般的な0.99よりかなり低い。HFT/Scalping指向で「直近の報酬」を極端に重視している。これが効いていた可能性大。 |
| **Entropy Coef** | **0.05** | 比較的高めの探索設定。 |
| **Hold Penalty** | **0.0** | v455/v456で導入された「Holdペナルティ」が存在しない。待つことへの罰がないため、機会を待つことができていた。 |
| **Profit/Loss Ratio** | Profit:1.0 / Loss:1.2 | 損失を1.2倍重く評価する非対称報酬。 |

### 2.2 Action Distribution (v454)
`action_analysis_sac_v454_inverse_confidence.txt` によれば、v454時点ではまだ行動分布は健全だった。
- BUY: ~35%
- SELL: ~35%
- HOLD: ~30%
このバランスが崩れた（HOLD 99%になった）のは、v455以降の「報酬関数の複雑化（Penaltyの雨あられ）」が主因と推測される。

### 2.3 Feature Sets
`config/feature_sets.yaml` には `minimal`, `balanced` 等の定義があるが、v451で指定されていた `"feature_set": "v451"` の実体はYAML内には見当たらない。コード内でハードコードされているか、動的に生成されていた可能性が高い。
**方針**: v457では「明示的なリスト」としてFeatureを定義しなおす。

---

## 3. リファクタリング計画 (Refactoring Plan)

### 3.1 現状の問題点
1.  **Configの散乱**: `config/` 直下に大量の `ab_search_temp_*.json` や、`v1`〜`v456` までの過去遺産が混在し、どれが「正」かわからない。
2.  **複雑なRegimeロジック (v444)**: `sac_v444_6_optimized_config.json` を見ると、Regimeごとに詳細なパラメータ（学習率まで！）を変える過剰設計が見られる。これはメンテナンス不能。
3.  **不透明なFeature定義**: 文字列指定（"v451"）で中身がブラックボックス化している。

### 3.2 v457での改善案
1.  **Config構造の一本化**: `config/v457/config.yaml` を唯一の正解とする。
2.  **パラメータの単純化**:
    - `Gamma = 0.80` (v451採用)
    - `Hold Penalty = 0` (v451採用)
    - Regimeによる学習率変更などの「動的ハイパーパラメータ」は廃止。
3.  **Featureのホワイトボックス化**:
    - v457のConfig内に、使用する特徴量リストを直接配列として記述する。
    - "minimal" などのセット名を使う場合も、その定義ファイルを `config/v457/features.yaml` にコピーして固定する。

## 4. 次のアクション
1.  **`config/v457/base/config.yaml` の作成**: v451のパラメータをベースに、不要な設定を削ぎ落としたクリーンな設定ファイルを作成する。
2.  **Reward Functionの実装**: `(PnL - Fee - Slippage) * Scale` という極めて単純な形式に戻し、`Hold Penalty=0` を厳守する。

### 3.3 追加調査結果 (Code Archeology)
コードベース（`HeavyTradingEnv`, `RewardCalculator`）を調査した結果、現在の報酬計算クラスが "God Object" と化しており、v451のシンプルさを設定変更だけで再現することは困難であることが判明。

- **Liability**: `RewardCalculator.py` (2000行超)。不要なコンポーネント（Behavioral Penalty, Smart Incentive, Regime Detector）が強結合している。
- **Solution**: `V457RewardCalculator` を新規実装し、複雑な旧実装を完全にバイパスする "Circuit Breaker" アプローチを採用する。
- **Action**: 環境初期化時に `config.reward_settings.type == "pnl_centered"` の場合、レガシー実装ではなく新クラスをロードするように `HeavyTradingEnv` を改修する（またはラッパーを作成する）。



