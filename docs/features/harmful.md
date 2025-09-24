# Harmful Features Documentation

## 概要

このドキュメントは、Wave3評価で `harmful` と判定された特徴量について、その理由と再評価条件を明文化したものです。

`harmful` フラグは「永久追放」ではなく「一時隔離」を意味します。条件が整えば再評価・再有効化が可能です。

## Harmful特徴量一覧

### Ichimoku (一目均衡表)

**判定時期**: Wave3評価

**理由**:

- VIF (Variance Inflation Factor) > 10: 多重共線性の問題
- MI (Mutual Information) ≈ 0: ターゲット変数との関連性が低い
- アブレーション分析で Sharpe ratio 大幅低下

**現状**: `harmful: true` で registry から除外

### RegimeClustering (レジームクラスタリング)

**判定時期**: Wave3評価

**理由**:

- アブレーション分析で Sharpe ratio 大幅低下
- 計算コストが高く、実用的でない
- クラスタリングの安定性が低い

**現状**: `harmful: true` で registry から除外

### Donchian

**判定時期**: Wave3評価

**理由**:

- VIF > 10: 多重共線性の問題
- 他のトレンド指標との重複が高い

**現状**: `harmful: true` で registry から除外

### KalmanFilter

**判定時期**: Wave3評価

**理由**:

- 計算コストが高く、実用的でない
- パラメータチューニングが難しい

**現状**: `harmful: true` で registry から除外

## 再評価条件

harmful と判定された特徴量も、以下の条件を満たした場合には再評価を行い、
必要に応じて stable セットへ復帰させる可能性があります。

- **Wave間の補完効果**  
  単独では無効でも、Wave1/2 の特徴量と組み合わせた場合に相関低下や補完効果が確認された場合。

- **特徴変換の導入**  
  差分化・正規化・次元削減（PCA/Autoencoder など）によって多重共線性や冗長性が解消された場合。

- **Ablationの定量基準**  
  再評価アブレーションで **delta_sharpe > +0.05** の改善が確認された場合は harmful 解除候補とする。  
  改善幅が +0.01 未満の場合は harmful 維持。

- **局面依存性の確認**  
  上昇局面・下落局面・レンジ市場など特定の Regime で有効性が示された場合。

- **軽量化後の見直し**  
  実装改善により計算負荷やメモリ効率が大幅に改善し、トレードオフが解消された場合。

## 再評価手順

1. **条件確認**: 上記再評価条件のいずれかを満たす
2. **実験的特徴として追加**: `src/trading/features/experimental.py` に実装
3. **評価実行**: `experimental_evaluator.py` で性能評価
4. **アブレーション分析**: `benchmarks/ablation_runner.py` で影響度測定
5. **成熟度更新**: `feature_maturity.yaml` で `stable` に移行
6. **自動移行**: `migrate_experimental.py` で本番コードに統合
7. **harmfulフラグ解除**: `features.yaml` から `harmful: true` を削除

## 注意事項

- harmful特徴量は `feature_sets.yaml` の `extended` セットにコメント付きで残されています
- CI/CDでは自動的に除外されます
- 再評価時は必ずアブレーション分析を実施してください
