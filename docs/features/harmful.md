# Harmful Features Documentation

## 概要

このドキュメントは、Wave3評価で `harmful` と判定された特徴量について、その理由と再評価条件を明文化したものです。

`harmful` フラグは「永久追放」ではなく「一時隔離」を意味します。条件が整えば再評価・再有効化が可能です。

## Harmful特徴量一覧

### Ichimoku (一目均衡表)

**判定時期**: Wave3評価
**Status**: isolated
**Reason**: high_multicollinearity, misinterpretation
**Last Eval**: 2025-09-25
**Note**: 基準線・転換線のみで評価。雲の厚み・抵抗帯距離・遅行スパン確認が未考慮

**理由**:

- VIF (Variance Inflation Factor) > 10: 多重共線性の問題
- MI (Mutual Information) ≈ 0: ターゲット変数との関連性が低い
- アブレーション分析で Sharpe ratio 大幅低下

**現状**: `harmful: true` で registry から除外

### RegimeClustering (レジームクラスタリング)

**判定時期**: Wave3評価
**Status**: isolated
**Reason**: weak_effect_size, unstable_output
**Last Eval**: 2025-09-25
**Note**: クラスタリングの安定性とパフォーマンス問題。代替手法検討要

**理由**:

- アブレーション分析で Sharpe ratio 大幅低下
- 計算コストが高く、実用的でない
- クラスタリングの安定性が低い

**現状**: `harmful: true` で registry から除外

### Donchian

**判定時期**: Wave3評価
**Status**: isolated
**Reason**: high_multicollinearity
**Last Eval**: 2025-09-25
**Note**: 他トレンド指標との重複。独自性再検証要

**理由**:

- VIF > 10: 多重共線性の問題
- 他のトレンド指標との重複が高い

**現状**: `harmful: true` で registry から除外

### ADX (Average Directional Index)

**判定時期**: Wave3評価
**Status**: isolated
**Reason**: high_multicollinearity, weak_signal
**Last Eval**: 2025-09-25
**Note**: 方向性指数だが、他のトレンド指標との重複が高く、シグナルが弱い

**JSON Metadata**:

```json
{
  "feature": "ADX",
  "category": "trend",
  "harmful_reasons": ["high_multicollinearity", "weak_signal"],
  "vif_score": 12.5,
  "sharpe_impact": -0.08,
  "correlation_with": ["DOW", "RegimeClustering"],
  "recommended_action": "consider_directional_components_only"
}
```

**理由**:

- VIF > 10: 多重共線性の問題
- アブレーション分析で Sharpe ratio 低下
- 他のトレンド指標との重複が高い

**現状**: `harmful: true` で registry から除外

### KAMA (Kaufman's Adaptive Moving Average)

**判定時期**: Wave3評価
**Status**: isolated
**Reason**: high_computation_cost, unstable_output
**Last Eval**: 2025-09-25
**Note**: 適応型移動平均だが、計算コストが高く出力が不安定

**JSON Metadata**:

```json
{
  "feature": "KAMA",
  "category": "trend",
  "harmful_reasons": ["high_computation_cost", "unstable_output"],
  "avg_computation_time_ms": 1850.0,
  "memory_usage_mb": 45.2,
  "sharpe_impact": -0.05,
  "stability_score": 0.65,
  "recommended_action": "optimize_implementation_or_remove"
}
```

**理由**:

- 計算コストが非常に高い (平均 1.8秒)
- パラメータ依存性が高く出力が不安定
- メモリ使用量が多い

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
