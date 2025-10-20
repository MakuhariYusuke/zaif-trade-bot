# Phase 4: Live & Paper Trade Schema Integration

**完了日**: 2025年10月10日  
**バージョン**: v1.0  
**ステータス**: ✅ 完了

## 📋 概要

Phase 3で実装したFeatureSchemaManagerシステムを、実取引システム（`live_trade.py`）とペーパートレードシステム（`paper_trade.py`）に統合しました。これにより、トレーニング・バックテスト・本番運用の全フェーズで一貫した特徴量管理が実現されました。

## 🎯 目的

元々の目的は「市場取引と過去BTC価格で確認」することでしたが、その過程で特徴量の不整合問題が発覚し、根本的な改革が必要になりました。Phase 4では、この改革の最終段階として実取引環境での検証を可能にします。

## 🔧 実装内容

### 1. live_trade.py の強化

#### 変更箇所

**A. スキーマ読み込み強化** (lines 443-488)
```python
# 古い実装: テスト環境を作成してスキーマを検証
# 問題点: 不要な環境作成、限定的なログ

# 新しい実装: FeatureSchemaManagerで直接読み込み
from ztb.training.core.feature_schema_manager import FeatureSchemaManager

model_name = self.model_path.stem
schema_manager = FeatureSchemaManager(model_name)
metadata = schema_manager.load_schema()

self.expected_features = metadata.num_features
self.feature_names = metadata.feature_names
self.schema_hash = metadata.schema_hash
self.schema_available = True

self.logger.info("✅ Schema loaded for model: %s", model_name)
self.logger.info("   Expected features: %d", self.expected_features)
self.logger.info("   Schema hash: %s", self.schema_hash[:16])
self.logger.info("   Created at: %s", metadata.created_at)
```

**B. 特徴量検証の改善** (lines 966-1003)
```python
# 3段階フォールバック: スキーマ > モデル > デフォルト
if self.schema_available and self.expected_features is not None:
    expected_features = self.expected_features  # スキーマ優先
elif self.model and hasattr(self.model, "observation_space"):
    expected_features = self.model.observation_space.shape[0]
else:
    expected_features = 68  # デフォルト

# TODO: 将来的には特徴量の並び順もスキーマに基づいて検証・修正
```

**C. 起動通知の強化** (lines 310-318)
```python
# スキーマ検証ステータスを通知メッセージに追加
if self.schema_available and self.expected_features:
    feature_info = f"{self.expected_features} features (schema-validated ✅)"
else:
    feature_info = f"{expected_features} features (no schema ⚠️)"
```

### 2. paper_trade.py の統合

#### 変更箇所

**A. レガシースキーマコードの削除と置き換え** (lines 279-370)
```python
# 削除されたコード (~90行):
# - ztb.utils.feature_schema からのインポート
# - models/features_schema.json の読み込み
# - 手動特徴量検証ロジック
# - 正規化統計の手動チェック

# 新しい実装 (~50行):
from ztb.training.core.feature_schema_manager import FeatureSchemaManager

model_name = self.model_path.stem
schema_manager = FeatureSchemaManager(model_name)
metadata = schema_manager.load_schema()

self.expected_features = metadata.num_features
self.feature_names = metadata.feature_names
self.schema_hash = metadata.schema_hash
self.schema_available = True

# 詳細ログ
self.logger.info("✅ Schema loaded for model: %s", model_name)
self.logger.info("   Expected features: %d", self.expected_features)
self.logger.info("   Schema hash: %s", self.schema_hash[:16])
self.logger.info("   Created at: %s", metadata.created_at)
```

**B. データ検証の簡素化** (lines 371-410)
```python
# シンプルな特徴量数検証
if self.schema_available and self.expected_features is not None:
    feature_columns = [
        col for col in self.test_df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(self.test_df[col])
    ]
    
    if len(feature_columns) != self.expected_features:
        self.logger.error(
            "❌ Feature count mismatch! Dataset has %d features, "
            "but schema expects %d",
            len(feature_columns),
            self.expected_features
        )
        raise ValueError(...)
    else:
        self.logger.info(
            "✅ Feature count validated: %d features match schema",
            len(feature_columns)
        )
```

**C. 起動通知の強化** (lines 757-775)
```python
# スキーマステータスをDiscord通知に追加
schema_status = (
    f"{trader.expected_features} features (schema-validated ✅)"
    if trader.schema_available and trader.expected_features
    else "schema not available ⚠️"
)

notifier.send_notification(
    title="📈 Paper Trading Started",
    fields={
        "Model": Path(args.model_path).name,
        "Features": schema_status,  # 新規追加
        # ... その他のフィールド
    },
)
```

## ✅ 検証結果

### live_trade.py テスト結果

```bash
$ python live_trade.py --model-path models/ppo_reward_v385_curated.zip \
                       --duration-hours 0.01 --dry-run
```

**出力**:
```
2025-10-10 18:09:03 [INFO] FeatureSchemaManager initialized for model: ppo_reward_v385_curated
2025-10-10 18:09:03 [INFO] 📖 Loaded schema for ppo_reward_v385_curated
2025-10-10 18:09:03 [INFO]    Features: 68
2025-10-10 18:09:03 [INFO]    Hash: c7a296f3d7c6ece4
2025-10-10 18:09:03,085 - INFO - ✅ Schema loaded for model: ppo_reward_v385_curated
2025-10-10 18:09:03,091 - INFO -    Expected features: 68
2025-10-10 18:09:03,091 - INFO -    Schema hash: c7a296f3d7c6ece4
2025-10-10 18:09:03,092 - INFO -    Created at: 2025-10-10T17:12:56.427356
2025-10-10 18:09:03,092 - INFO - 📋 Model feature requirements:
2025-10-10 18:09:03,093 - INFO -    Total: 68 features
2025-10-10 18:09:03,093 - INFO -    First 5: ['rsi', 'sma_short', 'sma_long', 'price', 'qty']
2025-10-10 18:09:03,094 - INFO -    Last 5: ['high', 'low', 'open', 'rolling_mean_20', 'volume']
2025-10-10 18:09:03,861 - INFO - Feature count validated: 68 features
2025-10-10 18:09:04 [INFO] Discord notification sent: ✅ Model Loaded Successfully
```

**結果**: ✅ スキーマ読み込み成功、特徴量検証OK

### paper_trade.py テスト結果

```python
from ztb.training.scripts.paper_trade import PaperTrader
t = PaperTrader('models/ppo_reward_v385_curated.zip', 'ml-dataset-enhanced.csv')
```

**出力**:
```
Successfully loaded model using Stable Baselines3 load method

=== Schema Status ===
Schema Available: True
Expected Features: 68
Schema Hash: c7a296f3d7c6ece4
```

**結果**: ✅ スキーマ読み込み成功、68特徴量確認

## 📊 Phase 4 の成果

### コード品質向上

| 項目 | 改善前 | 改善後 | 効果 |
|------|--------|--------|------|
| **live_trade.py** スキーマ読み込み | テスト環境作成 | 直接読み込み | シンプル化、高速化 |
| **live_trade.py** ログ情報 | 最小限 | 詳細（hash, 日時、リスト） | デバッグ容易性 |
| **paper_trade.py** レガシーコード | ~90行 | ~50行 | 40行削減 (44%減) |
| **paper_trade.py** スキーマシステム | 独自実装 | 統一システム | 一貫性向上 |
| 特徴量検証 | 環境依存 | スキーマ優先 | 信頼性向上 |

### 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    Feature Schema System                     │
│                  (Single Source of Truth)                    │
└─────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Training    │    │    Backtest      │    │ Live/Paper   │
│ (Phase 1-2)   │    │   (Phase 3)      │    │  (Phase 4)   │
├───────────────┤    ├──────────────────┤    ├──────────────┤
│ - UnifiedTrainer│  │ - schema_env_    │    │ - live_trade │
│ - Auto schema  │    │   factory        │    │ - paper_trade│
│   save         │    │ - backtest_with_ │    │              │
│                │    │   schema         │    │              │
└───────────────┘    └──────────────────┘    └──────────────┘
```

## 🔄 後方互換性

両システムはスキーマが存在しない場合も正常動作します：

```python
# スキーマがない場合
if not schema_available:
    logger.warning("⚠️  No schema found. Feature validation disabled.")
    # デフォルト値を使用して継続
```

これにより、古いモデルやテスト環境でも問題なく動作します。

## 📝 使用方法

### Live Trade (実取引・Dry Run)

```bash
# Dry Runモード（推奨: 初回検証）
python live_trade.py \
  --model-path models/ppo_reward_v385_curated.zip \
  --duration-hours 0.05 \
  --dry-run

# 実取引モード
python live_trade.py \
  --model-path models/ppo_reward_v385_curated.zip \
  --duration-hours 24
```

### Paper Trade (バックテスト評価)

```bash
# 1エピソード
python -m ztb.training.scripts.paper_trade \
  --model-path models/ppo_reward_v385_curated.zip \
  --test-data ml-dataset-enhanced.csv \
  --episodes 1

# 複数エピソード
python -m ztb.training.scripts.paper_trade \
  --model-path models/ppo_reward_v385_curated.zip \
  --test-data ml-dataset-enhanced.csv \
  --episodes 10
```

## 🎓 学んだ教訓

### 1. 統一システムの重要性
- 各スクリプトが独自実装を持つと保守が困難
- 単一の真実の情報源（FeatureSchemaManager）が必須

### 2. 段階的な改革の効果
- Phase 1: 基盤構築
- Phase 2: 自動化
- Phase 3: 環境統合
- Phase 4: 実運用統合
- 各段階で検証しながら進めることで安全に改革完了

### 3. ログの重要性
- 詳細なログ（hash, timestamp, feature list）により:
  - デバッグが容易
  - スキーマ検証が視覚的に確認可能
  - 問題発生時の追跡が簡単

## 🚀 今後の展開

### Phase 5 候補（将来）

1. **特徴量順序検証**
   ```python
   # TODO: paper_trade.py L1003
   # 特徴量の並び順もスキーマに基づいて検証・修正
   ```

2. **自動リバランス**
   - データの特徴量順序が異なる場合、自動で並び替え

3. **マルチモデル対応**
   - 複数モデルの同時検証
   - アンサンブル推論のスキーマ管理

4. **リアルタイム監視**
   - 実取引中のスキーマdrift検出
   - 特徴量分布の継続的モニタリング

## 📚 関連ドキュメント

- [Phase 1-2: FeatureSchemaManager](./PHASE1_2_FEATURE_SCHEMA_MANAGER.md)
- [Phase 3 Implementation](./PHASE3_IMPLEMENTATION_INSTRUCTIONS.md)
- [Phase 3 Review](./PHASE3_FINAL_REVIEW.md)
- [CHANGELOG](../CHANGELOG.md)

## ✅ Phase 4 完了チェックリスト

- [x] live_trade.py スキーマ統合
- [x] live_trade.py 特徴量検証強化
- [x] live_trade.py 起動通知強化
- [x] live_trade.py テスト実行（v385モデル）
- [x] paper_trade.py レガシーコード削除
- [x] paper_trade.py FeatureSchemaManager統合
- [x] paper_trade.py データ検証簡素化
- [x] paper_trade.py 起動通知強化
- [x] paper_trade.py テスト実行（v385モデル）
- [x] ドキュメント作成
- [x] CHANGELOG更新

**Phase 4 ステータス**: 🎉 **完了** (2025年10月10日)
