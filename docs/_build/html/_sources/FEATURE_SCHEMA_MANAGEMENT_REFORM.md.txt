# Feature Schema Management Reform - 改修計画書

## 問題の本質

### 現在の問題
1. **グローバルschemaの上書き問題**
   - `models/features_schema.json`と`models/scaler.npz`が最後の訓練で上書き
   - v381(110特徴量)の訓練後、v384(68特徴量)の訓練でv381の情報が消失
   - 推論時にモデルと環境の次元不一致エラー

2. **特徴量変更時の手間**
   - 特徴量リストの変更時に複数ファイルを手動更新
   - 環境、トレーナー、configの不整合リスク
   - バックテスト時の互換性確認が困難

3. **デバッグの困難さ**
   - どのモデルがどの特徴量で訓練されたか不明
   - 過去のモデルの再現が不可能

## 解決策: モデル固有スキーマ管理

### 新しいディレクトリ構造

```
models/
├── ppo_reward_v381_revised_profit_focused.zip  # モデルファイル
├── ppo_reward_v384_curated_60.zip
├── schemas/                                     # 新規: スキーマディレクトリ
│   ├── ppo_reward_v381_revised_profit_focused/
│   │   ├── metadata.json                        # モデルメタデータ
│   │   ├── features_schema.json                 # 特徴量リスト
│   │   └── scaler.npz                           # 正規化パラメータ
│   └── ppo_reward_v384_curated_60/
│       ├── metadata.json
│       ├── features_schema.json
│       └── scaler.npz
└── (レガシー: 削除予定)
    ├── features_schema.json                     # 削除予定
    └── scaler.npz                                # 削除予定
```

### metadata.jsonの構造

```json
{
  "model_name": "ppo_reward_v384_curated_60",
  "num_features": 68,
  "feature_names": [...],
  "schema_hash": "f7be18533fa61876",
  "created_at": "2025-10-10T16:26:20",
  "training_config": {
    "curated_features_list": "curated_features.py::CURATED_FEATURES",
    "enable_feature_filtering": true,
    "feature_filter_mode": "whitelist",
    "total_timesteps": 50000,
    "learning_rate": 0.003
  },
  "curated_features_spec": "curated_features.py::CURATED_FEATURES",
  "feature_filtering_enabled": true,
  "feature_filter_mode": "whitelist"
}
```

## 実装コンポーネント

### 1. FeatureSchemaManager (✅ 実装済み)

**ファイル**: `ztb/training/core/feature_schema_manager.py`

**主要機能**:
- モデルごとのスキーマ保存・読み込み
- 互換性チェック
- レガシースキーマの移行
- スキーマサマリー表示

**使用例**:
```python
# 訓練時
manager = FeatureSchemaManager(model_name="v384_curated_60")
manager.save_schema(
    features=feature_list,
    config=training_config,
    scaler_data={"mean": mean, "std": std}
)

# 推論時
manager = FeatureSchemaManager(model_name="v384_curated_60")
metadata = manager.load_schema()
features = manager.get_feature_list()
scaler = manager.load_scaler()
```

### 2. UnifiedTrainer統合 (🔄 進行中)

**修正箇所**: `ztb/training/core/unified_trainer.py`

**変更内容**:
1. FeatureSchemaManagerのインポート ✅
2. 訓練完了時のスキーマ自動保存 ⏳
3. 設定からの特徴量情報抽出 ⏳

**追加メソッド**:
```python
def _save_model_schema(self, model, session_id: str):
    """モデルと一緒にスキーマを保存"""
    manager = FeatureSchemaManager(model_name=session_id)
    
    # 環境から特徴量リスト取得
    features = self.env.get_feature_names()
    
    # スケーラー情報取得
    scaler_data = {
        "mean": self.env.scaler.mean,
        "std": self.env.scaler.std
    }
    
    # 保存
    manager.save_schema(
        features=features,
        config=self.config,
        scaler_data=scaler_data
    )
```

### 3. PPOTrainer統合 (⏳ 未実装)

**修正箇所**: `ztb/training/core/ppo_trainer.py`

**変更内容**:
1. 訓練完了時にスキーマ保存
2. 環境作成時にスキーマ読み込みオプション

**追加メソッド**:
```python
def save_with_schema(self, model_path: str, session_id: str):
    """モデルとスキーマを一緒に保存"""
    # モデル保存
    self.model.save(model_path)
    
    # スキーマ保存
    manager = FeatureSchemaManager(model_name=session_id)
    manager.save_schema(
        features=self.env.feature_names,
        config=self.config,
        scaler_data=self._get_scaler_data()
    )
```

### 4. Environment統合 (⏳ 未実装)

**修正箇所**: 
- `ztb/trading/environment/environment.py`
- `ztb/trading/environment/heavy_env/mixins/initialization.py`

**変更内容**:
1. スキーマからの環境初期化
2. 特徴量リストの動的読み込み

**新しいインターフェース**:
```python
def create_env_from_schema(model_name: str, df: pd.DataFrame):
    """スキーマからEnvironmentを作成"""
    manager = FeatureSchemaManager(model_name=model_name)
    metadata = manager.load_schema()
    scaler = manager.load_scaler()
    
    config = {
        "feature_names": metadata.feature_names,
        "scaler_mean": scaler["mean"],
        "scaler_std": scaler["std"],
        # ... その他の設定
    }
    
    return HeavyTradingEnv(df=df, config=config)
```

### 5. Backtest/Inference統合 (⏳ 未実装)

**新しいバックテストインターフェース**:
```python
def run_backtest_with_schema(model_path: str, data_path: str):
    """スキーマを考慮したバックテスト"""
    # モデル名を抽出
    model_name = Path(model_path).stem
    
    # スキーマ読み込み
    manager = FeatureSchemaManager(model_name=model_name)
    metadata = manager.load_schema()
    
    # 環境作成（スキーマに基づく）
    df = load_csv_data_optimized(data_path)
    env = create_env_from_schema(model_name, df)
    
    # モデル読み込み
    model = MaskablePPO.load(model_path)
    
    # バックテスト実行
    # （次元不一致エラーなし！）
```

## 実装スケジュール

### Phase 1: 基礎インフラ ✅
- [x] FeatureSchemaManager実装
- [x] ディレクトリ構造設計

### Phase 2: Trainer統合 (現在)
- [ ] UnifiedTrainerへの統合
- [ ] PPOTrainerへの統合
- [ ] 訓練時の自動スキーマ保存

### Phase 3: Environment統合
- [ ] スキーマからの環境作成
- [ ] 動的特徴量読み込み
- [ ] スケーラー自動設定

### Phase 4: Inference統合
- [ ] バックテスト改修
- [ ] live_trade改修
- [ ] paper_trade改修

### Phase 5: Migration
- [ ] 既存モデルのスキーマ移行ツール
- [ ] レガシーファイル削除
- [ ] ドキュメント更新

## 使用例

### 訓練（自動スキーマ保存）
```bash
# 従来通り訓練するだけでスキーマも自動保存
python run_training.py --config configs/training/ppo_reward_v385.json

# 結果:
# models/ppo_reward_v385.zip
# models/schemas/ppo_reward_v385/
#   ├── metadata.json
#   ├── features_schema.json
#   └── scaler.npz
```

### バックテスト（自動スキーマ読み込み）
```bash
# モデル名からスキーマを自動検出
python backtest_model.py --model models/ppo_reward_v385.zip --data btc_jpy_real_dataset.csv

# 内部処理:
# 1. v385のスキーマ読み込み
# 2. 68特徴量を認識
# 3. 環境を68特徴量用に設定
# 4. バックテスト実行（エラーなし！）
```

### スキーマ確認
```bash
# 全スキーマのサマリー表示
python -c "from ztb.training.core.feature_schema_manager import FeatureSchemaManager; FeatureSchemaManager.print_schema_summary()"

# 出力:
# ================================================================================
# Available Feature Schemas
# ================================================================================
# 
# 📦 ppo_reward_v381_revised_profit_focused
#    Features: 110
#    Hash: a1b2c3d4e5f6g7h8
#    Created: 2025-10-09T15:30:00
# 
# 📦 ppo_reward_v384_curated_60
#    Features: 68
#    Hash: f7be18533fa61876
#    Created: 2025-10-10T16:26:20
#    Curated: curated_features.py::CURATED_FEATURES
# ================================================================================
```

### 互換性チェック
```python
manager_v384 = FeatureSchemaManager("ppo_reward_v384_curated_60")
compatible = manager_v384.verify_compatibility("ppo_reward_v381_revised_profit_focused")

# 出力:
# ⚠️  ppo_reward_v384_curated_60 and ppo_reward_v381_revised_profit_focused are NOT compatible
#    ppo_reward_v384_curated_60: 68 features
#    ppo_reward_v381_revised_profit_focused: 110 features
```

## メリット

### 1. 自動化
- 訓練時にスキーマ自動保存
- 推論時にスキーマ自動読み込み
- 手動設定不要

### 2. 安全性
- 次元不一致エラーの自動防止
- 特徴量の整合性保証
- バージョン管理

### 3. 再現性
- 過去のモデルの正確な再現
- 特徴量構成の完全な記録
- デバッグの容易化

### 4. 柔軟性
- 複数の特徴量セットを並行管理
- A/Bテストの簡易化
- 実験の迅速化

## 次のステップ

1. **UnifiedTrainer統合** ⭐ 優先度: 高
   - `_train_ppo()`メソッドにスキーマ保存処理追加
   - 訓練完了時の自動保存

2. **Environment改修** ⭐ 優先度: 高
   - `create_env_from_schema()`関数実装
   - HeavyTradingEnvの拡張

3. **Backtest改修**
   - スキーマベースのバックテスト
   - v381/v384両方でテスト可能に

4. **Migration Tool**
   - v381/v384のスキーマ移行
   - レガシーファイルのバックアップ

5. **Documentation**
   - API reference
   - Migration guide
   - Best practices

---

**作成日**: 2025-10-10  
**ステータス**: Phase 2 (Trainer統合) 進行中
