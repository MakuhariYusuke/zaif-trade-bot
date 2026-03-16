# Feature Schema Management Reform - 実装完了サマリー

## 実装日時
2025年10月10日

## 実装概要

特徴量の増減時の不便さを解消するため、`unified_trainer.py`を中心に**モデルごとの特徴量スキーマ管理システム**を実装しました。

## 主な変更

### 1. FeatureSchemaManager (NEW)

**ファイル**: `ztb/training/core/feature_schema_manager.py`

**機能**:
- ✅ モデルごとのスキーマ保存・読み込み
- ✅ 互換性チェック
- ✅ スケーラー情報管理
- ✅ メタデータ管理
- ✅ レガシースキーマ移行サポート

**新しいディレクトリ構造**:
```
models/
├── ppo_reward_v384_curated_60.zip
└── schemas/
    └── ppo_reward_v384_curated_60/
        ├── metadata.json           # モデルメタデータ
        ├── features_schema.json    # 特徴量リスト
        └── scaler.npz              # 正規化パラメータ
```

### 2. UnifiedTrainer統合

**ファイル**: `ztb/training/core/unified_trainer.py`

**変更内容**:
1. ✅ FeatureSchemaManagerのインポート
2. ✅ `_save_model_schema()`メソッド追加
3. ✅ 訓練完了時の自動スキーマ保存

**追加メソッド**:
```python
def _save_model_schema(self, session_id: str, model_dir: Path, df: Optional[Any] = None):
    """モデルの特徴量スキーマを保存"""
    # 1. DataFrameから特徴量を自動検出
    # 2. スケーラーデータを計算
    # 3. FeatureSchemaManagerで保存
```

**修正箇所**:
- Line 85: FeatureSchemaManagerインポート追加
- Line 502-561: `_save_model_schema()`メソッド実装
- Line 659: モデル保存後にスキーマ自動保存

## 使用方法

### 訓練（自動スキーマ保存）

```bash
# 従来通り訓練するだけ
python run_training.py --config configs/training/ppo_reward_v385_curated.json

# スキーマが自動保存される:
# models/ppo_reward_v385_curated.zip
# models/schemas/ppo_reward_v385_curated/
#   ├── metadata.json
#   ├── features_schema.json
#   └── scaler.npz
```

### スキーマ確認

```python
from ztb.training.core.feature_schema_manager import FeatureSchemaManager

# 全スキーマのサマリー表示
FeatureSchemaManager.print_schema_summary()

# 特定モデルのスキーマ読み込み
manager = FeatureSchemaManager("ppo_reward_v384_curated_60")
metadata = manager.load_schema()
print(f"Features: {metadata.num_features}")
print(f"Hash: {metadata.schema_hash}")
print(f"Created: {metadata.created_at}")

# 互換性チェック
compatible = manager.verify_compatibility("ppo_reward_v381_revised_profit_focused")
```

### レガシースキーマの移行

```python
from ztb.training.core.feature_schema_manager import migrate_legacy_schema

# v384の既存スキーマを移行
migrate_legacy_schema(
    model_name="ppo_reward_v384_curated_60",
    legacy_schema_path=Path("models/features_schema.json"),
    legacy_scaler_path=Path("models/scaler.npz"),
    config={
        "curated_features_list": "curated_features.py::CURATED_FEATURES",
        "enable_feature_filtering": True,
        "total_timesteps": 50000
    }
)
```

## 解決された問題

### Before (問題あり)
```
models/
├── ppo_reward_v381.zip
├── ppo_reward_v384.zip
├── features_schema.json      ← v384で上書き（v381の情報消失）
└── scaler.npz                 ← v384で上書き

# バックテスト実行
python backtest.py --model models/ppo_reward_v381.zip
# ❌ エラー: 次元不一致 (期待: 110, 実際: 68)
```

### After (解決済み)
```
models/
├── ppo_reward_v381.zip
├── ppo_reward_v384.zip
└── schemas/
    ├── ppo_reward_v381/
    │   ├── metadata.json       ← v381固有（110特徴量）
    │   ├── features_schema.json
    │   └── scaler.npz
    └── ppo_reward_v384/
        ├── metadata.json       ← v384固有（68特徴量）
        ├── features_schema.json
        └── scaler.npz

# バックテスト実行（将来実装）
python backtest.py --model models/ppo_reward_v381.zip
# ✅ 成功: スキーマから110特徴量を自動検出
```

## メリット

### 1. 自動化
- ✅ 訓練時にスキーマ自動保存
- ⏳ 推論時にスキーマ自動読み込み（次フェーズ）
- ✅ 手動設定不要

### 2. 安全性
- ✅ モデルごとの特徴量情報を永続化
- ⏳ 次元不一致エラーの自動防止（環境統合後）
- ✅ バージョン管理とハッシュ検証

### 3. 再現性
- ✅ 過去のモデルの正確な情報保持
- ✅ 特徴量構成の完全な記録
- ✅ デバッグの容易化

### 4. 柔軟性
- ✅ 複数の特徴量セットを並行管理
- ✅ 実験の迅速化
- ✅ A/Bテストの簡易化

## 未実装（次のステップ）

### Phase 3: Environment統合 ⏳
```python
# 目標: スキーマから環境を自動構築
from ztb.training.core.feature_schema_manager import FeatureSchemaManager

manager = FeatureSchemaManager("ppo_reward_v384_curated_60")
metadata = manager.load_schema()

# 環境を適切な特徴量数で作成
env = create_env_from_schema(
    model_name="ppo_reward_v384_curated_60",
    df=df
)
# → 68特徴量で環境構築
```

### Phase 4: Backtest/Inference統合 ⏳
```python
# 目標: スキーマベースのバックテスト
def run_backtest_with_schema(model_path: str, data_path: str):
    """スキーマを考慮したバックテスト"""
    model_name = Path(model_path).stem

    # スキーマ読み込み
    manager = FeatureSchemaManager(model_name)
    metadata = manager.load_schema()

    # 環境作成（スキーマに基づく）
    env = create_env_from_schema(model_name, df)

    # バックテスト実行
    # （次元不一致エラーなし！）
```

### Phase 5: Migration Tool ⏳
```bash
# 目標: 既存モデルの一括移行
python scripts/migrate_all_schemas.py

# 実行内容:
# 1. models/ディレクトリのすべてのモデルをスキャン
# 2. 各モデルの訓練configを検索
# 3. スキーマを再構築して保存
# 4. レガシーファイルをbackup/に移動
```

## テスト

### 単体テスト
```bash
# FeatureSchemaManagerのテスト
pytest tests/test_feature_schema_manager.py

# テスト項目:
# - スキーマ保存・読み込み
# - 互換性チェック
# - ハッシュ計算
# - エラーハンドリング
```

### 統合テスト
```bash
# 訓練→スキーマ保存の統合テスト
python run_training.py --config configs/training/ppo_test_schema.json

# 確認:
# 1. models/schemas/ppo_test_schema/が作成される
# 2. metadata.jsonに正しい情報が含まれる
# 3. features_schema.jsonが正しい
# 4. scaler.npzが正しい
```

## ドキュメント

- ✅ `docs/FEATURE_SCHEMA_MANAGEMENT_REFORM.md`: 改修計画書
- ✅ `docs/V381_V384_VERIFICATION_STATUS.md`: 検証状況
- ✅ このドキュメント: 実装完了サマリー

## 既知の制限

1. **Environmentは未統合**: バックテストでの次元不一致はまだ発生
2. **レガシーファイル残存**: `models/features_schema.json`は手動削除が必要
3. **Paper trading未対応**: live_trade.pyの改修が必要

## 次回訓練での動作確認

### テスト手順
```bash
# 1. 新しいモデルを訓練
python run_training.py --config configs/training/ppo_reward_v385_test.json

# 2. スキーマディレクトリを確認
ls -R models/schemas/ppo_reward_v385_test/

# 3. スキーマ内容を確認
python -c "
from ztb.training.core.feature_schema_manager import FeatureSchemaManager
manager = FeatureSchemaManager('ppo_reward_v385_test')
metadata = manager.load_schema()
print(f'Features: {metadata.num_features}')
print(f'Config: {metadata.training_config}')
"

# 4. すべてのスキーマをリスト
python -c "
from ztb.training.core.feature_schema_manager import FeatureSchemaManager
FeatureSchemaManager.print_schema_summary()
"
```

## まとめ

✅ **実装完了**: FeatureSchemaManager + UnifiedTrainer統合
⏳ **次フェーズ**: Environment統合（バックテスト修正）
📊 **効果**: モデルごとの特徴量情報を永続化、管理の自動化

これにより、**特徴量を増やしたり減らしたりする際の手間が大幅に削減**され、モデルの管理と再現性が向上しました。

---

**実装日**: 2025-10-10
**ステータス**: Phase 2完了、Phase 3へ移行準備中
**影響**: 今後の訓練で自動的にスキーマ保存される
