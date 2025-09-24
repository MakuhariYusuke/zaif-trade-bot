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

**再評価条件**:

- 特徴変換（例: 差分化、正規化、距離表現）を実装した場合
- Wave4+ で新しい特徴量と組み合わせた場合
- ランキングで `delta_sharpe > 0.05` が出た場合

### RegimeClustering (レジームクラスタリング)

**判定時期**: Wave3評価

**理由**:

- アブレーション分析で Sharpe ratio 大幅低下
- 計算コストが高く、実用的でない
- クラスタリングの安定性が低い

**現状**: `harmful: true` で registry から除外

**再評価条件**:

- 特徴変換（例: 差分化、正規化、距離表現）を実装した場合
- Wave4+ で新しい特徴量と組み合わせた場合
- ランキングで `delta_sharpe > 0.05` が出た場合
- 計算効率が改善された場合

### Donchian

**判定時期**: Wave3評価

**理由**:

- VIF > 10: 多重共線性の問題
- 他のトレンド指標との重複が高い

**現状**: `harmful: true` で registry から除外

**再評価条件**:

- 特徴変換を実装した場合
- 他の特徴量との組み合わせで有用性が確認された場合

### KalmanFilter

**判定時期**: Wave3評価

**理由**:

- 計算コストが高く、実用的でない
- パラメータチューニングが難しい

**現状**: `harmful: true` で registry から除外

**再評価条件**:

- 計算効率が改善された場合
- パラメータ自動最適化を実装した場合

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
