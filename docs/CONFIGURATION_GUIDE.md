# 統一設定管理ガイド (Unified Configuration Management Guide)

## Bug #52修正による変更点

Bug #52の修正に伴い、全てのトレーナーで一貫した設定管理を実現するように設計を改善しました。

## 設定の優先順位 (Configuration Priority)

設定パラメータは以下の優先順位で適用されます:

1. **コマンドライン引数** (最優先)
2. **設定ファイル (トップレベル)** 
3. **設定ファイル (セクション別)**
4. **デフォルト値** (最低優先)

## 統一設定構造 (Unified Configuration Structure)

### 推奨される設定ファイル構造

```json
{
  "algorithm": "ppo",
  "session_id": "my_training_session",
  "data_path": "ml-dataset-enhanced-balanced.csv",
  "checkpoint_dir": "checkpoints/my_session",
  
  "comment_memory": "===== MEMORY OPTIMIZATION =====",
  "comment_memory_1": "These parameters were added as part of Bug #52 fix",
  "memory_optimization": {
    "data_rows_limit": 500,
    "max_features": 40
  },
  
  "comment_ppo": "===== PPO HYPERPARAMETERS =====",
  "ppo": {
    "total_timesteps": 30000,
    "learning_rate": 0.0003,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 6,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "normalize_advantage": true,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": false,
    "sde_sample_freq": -1,
    "verbose": 1
  },
  
  "comment_env": "===== ENVIRONMENT PARAMETERS =====",
  "environment": {
    "max_position_size": 1.0,
    "initial_balance": 1000000,
    "transaction_cost": 0.001,
    "reward_scaling": 1.0
  },
  
  "comment_backward": "===== BACKWARD COMPATIBILITY =====",
  "comment_backward_1": "These top-level settings are maintained for backward compatibility",
  "data_rows_limit": 500,
  "max_features": 40,
  "total_timesteps": 30000,
  "learning_rate": 0.0003,
  "n_steps": 512,
  "batch_size": 128
}
```

### 簡略化された設定 (後方互換性あり)

トップレベルに直接設定を記述する従来の方法も引き続きサポートされます:

```json
{
  "algorithm": "ppo",
  "session_id": "my_training_session",
  "data_path": "ml-dataset-enhanced-balanced.csv",
  
  "data_rows_limit": 500,
  "max_features": 40,
  "total_timesteps": 30000,
  "learning_rate": 0.0003,
  "n_steps": 512,
  "batch_size": 128
}
```

## メモリ最適化パラメータ (Memory Optimization Parameters)

### data_rows_limit

**目的**: データセットの行数を制限し、メモリ使用量を削減

**適用場所**:
- `ppo_trainer.py`: データ読み込み時
- `sell_mitigation_ppo_trainer.py`: データ読み込み時

**設定方法** (優先順位):
1. `memory_optimization.data_rows_limit`
2. トップレベル `data_rows_limit`

**推奨値**:
- テスト/開発: 300-500行
- 本番トレーニング: 500-1000行
- フルデータ: 指定しない (nullまたは省略)

**例**:
```json
{
  "data_rows_limit": 500  // 1000行 → 500行に削減
}
```

### max_features

**目的**: 特徴量数を制限し、メモリ使用量を削減

**適用場所**:
- `environment.py`: 特徴量選択時 (分散ベース)

**設定方法** (優先順位):
1. コンストラクタ引数 `max_features`
2. `memory_optimization.max_features`
3. トップレベル `max_features`
4. `ppo.max_features`
5. `config.max_features` 属性

**推奨値**:
- 最小限: 30個
- バランス: 40-50個
- フル: 指定しない (110個全て使用)

**例**:
```json
{
  "max_features": 40  // 110特徴量 → 40特徴量に削減 (高分散30個を選択)
}
```

## 設定の伝播フロー (Configuration Propagation Flow)

```
unified_trainer.py (エントリーポイント)
  ↓
build_unified_config() メソッド
  ├─ get_memory_optimization_config()  → memory_optimization section
  ├─ get_ppo_core_config()            → ppo section
  ├─ get_environment_config()         → environment section
  └─ 全トップレベル設定をコピー         → top-level (backward compatibility)
  ↓
unified_config = {
  "ppo": {...},
  "memory_optimization": {...},
  "environment": {...},
  ...all top-level settings...
}
  ↓
TrainerParams / SELLMitigationParams
  ↓
ppo_trainer.py / sell_mitigation_ppo_trainer.py
  ├─ data_rows_limit取得 (優先順位順)
  └─ max_features取得 (優先順位順)
  ↓
environment.py
  └─ max_features取得 (多段階フォールバック)
```

## トラブルシューティング

### 設定が適用されない場合

1. **設定ファイルの読み込みを確認**
   ```
   Loaded configuration from configs/training/your_config.json
   ```

2. **メモリ最適化ログを確認**
   ```
   ⚠️  MEMORY OPTIMIZATION: Limiting data from 1000 to 500 rows
   ⚠️  MEMORY OPTIMIZATION: Reducing features from 110 to 40
   ```

3. **トップレベルと セクション両方に設定**
   後方互換性のため、両方に記述することを推奨:
   ```json
   {
     "data_rows_limit": 500,  // トップレベル
     "memory_optimization": {
       "data_rows_limit": 500  // セクション内
     }
   }
   ```

### 型エラーが発生する場合

設定ファイルで型を明示:
```json
{
  "data_rows_limit": 500,      // 数値
  "max_features": 40,          // 数値
  "learning_rate": 0.0003,     // 浮動小数点
  "normalize_advantage": true  // ブール値
}
```

## 変更履歴

### v3.7.0 (2025-10-08) - Bug #52修正

**追加された機能**:
- 統一設定管理システム
- `build_unified_config()` メソッド
- メモリ最適化パラメータ (`data_rows_limit`, `max_features`)

**改善されたコンポーネント**:
- `unified_trainer.py`: 統一設定ヘルパーメソッド追加
- `ppo_trainer.py`: 優先順位ベースの設定取得
- `sell_mitigation_ppo_trainer.py`: 優先順位ベースの設定取得
- `environment.py`: 多段階フォールバック設定取得

**後方互換性**:
- 既存の設定ファイルは変更なしで動作
- トップレベル設定は引き続きサポート
