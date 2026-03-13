# V434: 多彩な学習方法の組み合わせ戦略

## 概要

v434シリーズでは、SACアルゴリズムを基盤とし、v433で確立された適応型学習フレームワークを拡張して、多彩な学習方法を統合したハイブリッド学習アプローチを採用します。本ドキュメントでは、これらの学習方法の組み合わせ方と実装戦略について説明します。

## 学習方法の概要

### 1. Curriculum Learning (カリキュラム学習)
段階的に学習難易度を上げることで、安定した学習を実現します。

**特徴:**
- 基本取引 → 中級取引 → 上級取引の段階的進行
- 各段階で最適化された報酬関数と特徴量
- 進捗基準による自動遷移

### 2. Adaptive Feature Selection (適応型特徴量選択)
市場条件に応じて最適な特徴量を動的に選択します。

**特徴:**
- Mutual Informationベースの特徴量重要度評価
- リアルタイムでの特徴量更新
- 計算コストの最適化

### 3. Market Regime Detection (市場レジーム検知)
現在の市場状況をリアルタイムで分類し、対応する戦略を適用します。

**特徴:**
- 強気/弱気/横ばい/高ボラティリティの4レジーム分類
- クラスタリングベースの検知アルゴリズム
- レジーム別パラメータ適応

### 4. Dynamic Reward Shaping (動的報酬形成)
市場状況に応じて報酬関数を動的に調整します。

**特徴:**
- ボラティリティ調整報酬
- トレンド強度ボーナス
- リスク調整報酬スケーリング

### 5. Ensemble Learning (アンサンブル学習)
複数のモデルを組み合わせることで、堅牢性を向上させます。

**特徴:**
- パラメータノイズによる多様性確保
- パフォーマンスベースの重み付け
- コンフリクト解決メカニズム

### 6. Online Learning (オンライン学習)
実運用中に継続的に学習を更新します。

**特徴:**
- 概念ドリフト検知
- インクリメンタルパラメータ更新
- 適応レート制御

## 組み合わせ戦略

### Phase 1: Foundation Building (基盤構築フェーズ)
**組み合わせ:** Curriculum Learning + Adaptive Feature Selection

**目的:** 安定した学習基盤の確立

**実装:**
```json
{
  "curriculum_learning": {
    "enabled": true,
    "stages": [
      {"name": "basic_trading", "timesteps": 2000, "difficulty": 0.3},
      {"name": "intermediate_trading", "timesteps": 3000, "difficulty": 0.6},
      {"name": "advanced_trading", "timesteps": 5000, "difficulty": 1.0}
    ]
  },
  "adaptive_feature_selection": {
    "enabled": true,
    "selection_method": "mutual_info",
    "top_k_features": 20,
    "update_frequency": 1000
  }
}
```

**効果:**
- 段階的な難易度上昇により学習の安定化
- 各段階で最適化された特徴量セットの使用
- 過学習の防止

### Phase 2: Adaptive Optimization (適応最適化フェーズ)
**組み合わせ:** Market Regime Detection + Dynamic Reward Shaping

**目的:** 市場状況への適応性向上

**実装:**
```json
{
  "market_regime_detection": {
    "enabled": true,
    "regimes": ["bull", "bear", "sideways", "volatile"],
    "adaptation_frequency": 500
  },
  "dynamic_reward_shaping": {
    "enabled": true,
    "market_regime_adaptation": true,
    "volatility_adjustment": true,
    "trend_strength_bonus": 0.005
  }
}
```

**効果:**
- 市場レジームに応じた戦略自動切換
- ボラティリティに応じたリスク調整
- トレンド相場でのパフォーマンス向上

### Phase 3: Ensemble & Continuous Learning (統合・継続学習フェーズ)
**組み合わせ:** Ensemble Learning + Online Learning

**目的:** 堅牢性と継続的改善の実現

**実装:**
```json
{
  "ensemble_learning": {
    "enabled": true,
    "num_models": 3,
    "diversity_method": "parameter_noise",
    "weighting_method": "performance_based"
  },
  "online_learning": {
    "enabled": true,
    "update_frequency": 100,
    "concept_drift_detection": true,
    "adaptation_rate": 0.01
  }
}
```

**効果:**
- 複数のモデルの予測を統合し、安定性向上
- 市場変化へのリアルタイム適応
- 概念ドリフトへの自動対応

## 実装アーキテクチャ

### 統合学習マネージャー

```python
class CombinedLearningManager:
    def __init__(self, config):
        self.curriculum_manager = CurriculumManager(config)
        self.feature_selector = AdaptiveFeatureSelector(config)
        self.regime_detector = MarketRegimeDetector(config)
        self.reward_shaper = DynamicRewardShaper(config)
        self.ensemble_manager = EnsembleManager(config)
        self.online_learner = OnlineLearningEngine(config)

    def train_step(self, observation, action, reward, next_observation, done):
        # 1. 市場レジーム検知
        regime = self.regime_detector.detect(observation)

        # 2. カリキュラム段階確認
        stage = self.curriculum_manager.get_current_stage()

        # 3. 適応型特徴量選択
        features = self.feature_selector.select(observation, regime)

        # 4. 動的報酬形成
        shaped_reward = self.reward_shaper.shape(reward, regime, stage)

        # 5. アンサンブル学習
        ensemble_action = self.ensemble_manager.predict(features)

        # 6. オンライン学習更新
        self.online_learner.update(features, shaped_reward, next_observation)

        return shaped_reward, ensemble_action
```

### 学習パイプライン

```
Raw Data → Feature Selection → Regime Detection → Reward Shaping → Ensemble Prediction → Online Update
     ↓            ↓              ↓              ↓              ↓              ↓
Curriculum   Adaptive       Market         Dynamic       Multiple        Continuous
Learning    Selection      Adaptation     Rewards       Models         Learning
```

## 評価と検証

### パフォーマンス指標

1. **学習安定性指標**
   - 報酬の分散度
   - 学習曲線の滑らかさ
   - 過学習検知スコア

2. **適応性指標**
   - レジーム別パフォーマンス
   - 特徴量選択の有効性
   - アンサンブルの多様性スコア

3. **堅牢性指標**
   - ストレステスト結果
   - アウトオブサンプル性能
   - 概念ドリフト耐性

### 検証方法

1. **クロスバリデーション**
   - 時系列クロスバリデーション
   - レジーム別検証
   - アンサンブル成分分析

2. **ウォークフォワード分析**
   - 段階的検証
   - パフォーマンス劣化検知
   - 再訓練トリガー評価

## 設定例: v434.1

v434.1では、上記の組み合わせ戦略を実装した設定ファイルを提供します。

**主要パラメータ:**
- `total_timesteps`: 10000 (カリキュラム学習対応)
- `curriculum_stages`: 3段階の学習難易度
- `market_regimes`: 4つの市場状況分類
- `ensemble_models`: 3つの多様性モデル
- `online_update_frequency`: 100ステップ毎の更新

## 利点と課題

### 利点

1. **学習効率の向上**: カリキュラム学習による段階的難易度上昇
2. **適応性の強化**: 市場レジーム検知による状況別最適化
3. **堅牢性の確保**: アンサンブル学習による予測安定化
4. **継続的改善**: オンライン学習による運用中適応

### 課題と対策

1. **計算コスト増加**
   - **対策**: 適応型特徴量選択による次元削減

2. **複雑性管理**
   - **対策**: モジュール化されたアーキテクチャ

3. **ハイパーパラメータ調整**
   - **対策**: 自動最適化アルゴリズムの導入

## 次のステップ

1. **v434.2**: 高度な転移学習統合
2. **v434.3**: マルチモーダル学習拡張
3. **v434.4**: 自己教師あり学習導入

---

*本ドキュメントはv434シリーズの学習戦略を定義します。実装の詳細はコードベースを参照してください。*
