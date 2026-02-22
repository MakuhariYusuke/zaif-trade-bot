# Configuration Guide

## Core Configuration Classes

### ZTBConfig

Central configuration management for all system components.

```python
from ztb.utils.config import ZTBConfig

config = ZTBConfig()
mem_profile = config.get('ZTB_MEM_PROFILE', False)
cuda_warn = config.get('ZTB_CUDA_WARN_GB', 0.0)
```

### Environment Variable Categories

1. **Observability**: `ZTB_MEM_PROFILE`, `ZTB_CUDA_WARN_GB`, `ZTB_LOG_LEVEL`
2. **Training**: `ZTB_CHECKPOINT_INTERVAL`, `ZTB_MAX_MEMORY_GB`
3. **Testing**: `ZTB_TEST_ISOLATION`, `ZTB_FLOAT_TOLERANCE`

## Configuration File Examples

### YAML Configuration File

```yaml
# config/trade-config.yaml
observability:
  mem_profile: true
  cuda_warn_gb: 8.0
  log_level: "INFO"

training:
  checkpoint_interval: 1000
  max_memory_gb: 16.0

testing:
  isolation: true
  float_tolerance: 0.01
```

### JSON Configuration File

```json
{
  "observability": {
    "mem_profile": true,
    "cuda_warn_gb": 8.0,
    "log_level": "INFO"
  },
  "training": {
    "checkpoint_interval": 1000,
    "max_memory_gb": 16.0
  },
  "testing": {
    "isolation": true,
    "float_tolerance": 0.01
  }
}
```

## Environment Variables vs Configuration Files

| Aspect | Environment Variables | Configuration Files |
|--------|----------------------|-------------------|
| **Priority** | High (overrides files) | Low (fallback) |
| **Use Case** | Secrets, runtime overrides | Default settings, complex configs |
| **Format** | String only | YAML/JSON (structured) |
| **Validation** | Type conversion with fallbacks | Schema validation |
| **Examples** | `ZTB_MEM_PROFILE=1` | `mem_profile: true` |

### Usage Patterns

**Development (Environment Variables):**

```bash
export ZTB_MEM_PROFILE=1
export ZTB_CUDA_WARN_GB=4.0
export ZTB_LOG_LEVEL=DEBUG
```

**Production (Configuration File):**

```yaml
# Load via ZTB_CONFIG_FILE=/path/to/config.yaml
observability:
  mem_profile: true
  cuda_warn_gb: 8.0
  log_level: "WARN"
```

**Hybrid Approach:**

```bash
# Base config from file
export ZTB_CONFIG_FILE=config/prod.yaml
# Runtime overrides
export ZTB_LOG_LEVEL=DEBUG
```

## Training Configuration Files

### unified_training_config.json

Unified training systemのメイン設定ファイル。PPOトレーニングと反復トレーニングの両方をサポート。

| パラメータ | 型 | 説明 | デフォルト値 | 例 |
|----------|----|------|------------|----|
| `algorithm` | string | トレーニングアルゴリズム: "ppo" または "iterative" | "ppo" | "ppo" |
| `data_path` | string | トレーニングデータファイルのパス | - | "ml-dataset-enhanced.csv" |
| `session_id` | string | セッション識別子（ログ/チェックポイント用） | - | "scalping_15s_ultra_aggressive_1M" |
| `total_timesteps` | int | 総トレーニングステップ数 | 1000000 | 1000000 |
| `checkpoint_dir` | string | チェックポイント保存ディレクトリ | "checkpoints" | "checkpoints" |
| `log_dir` | string | ログ保存ディレクトリ | "logs" | "logs" |
| `model_dir` | string | モデル保存ディレクトリ | "models" | "models" |
| `verbose` | int | ログ詳細度 (0-2) | 1 | 1 |
| `validation_level` | string | 検証レベル: "basic" または "full" | "basic" | "basic" |
| `correlation_id` | string | 監視/レポート用識別子 | - | "scalping_15s_ultra_aggressive_1M" |
| `trading_mode` | string | トレードモード: "scalping", "swing", "position" | "scalping" | "scalping" |
| `feature_set` | string | 特徴量セット: "full" または "reduced" | "full" | "full" |
| `timeframe` | string | タイムフレーム | "15s" | "15s" |
| `transaction_cost` | float | 取引コスト（手数料率） | 0.0005 | 0.0005 |
| `max_position_size` | float | 最大ポジションサイズ（比率） | 0.5 | 0.5 |
| `reward_*` | float | 報酬関数パラメータ | 各種 | 0.01 |
| `offline_mode` | bool | オフラインモード（データファイル使用） | true | true |

### scalping-config.json

スキャルピング戦略専用の設定ファイル。リスク管理とトレーニングパラメータを含む。

#### データ設定

| パラメータ | 説明 |
|----------|------|
| `data.train_data` | トレーニングデータファイル |
| `data.test_data` | テストデータファイル |
| `data.validation_data` | 検証データファイル |

#### トレーニング設定

| パラメータ | 説明 | デフォルト |
|----------|------|----------|
| `training.total_timesteps` | 総ステップ数 | 250000 |
| `training.eval_freq` | 評価頻度 | 10000 |
| `training.batch_size` | バッチサイズ | 64 |
| `training.n_steps` | ステップ数 | 2048 |
| `training.gamma` | 割引率 | 0.99 |
| `training.learning_rate` | 学習率 | 0.0003 |
| `training.ent_coef` | エントロピー係数 | 0.0 |
| `training.clip_range` | PPOクリップ範囲 | 0.2 |

#### 環境設定（リスク管理）

| パラメータ | 説明 | デフォルト |
|----------|------|----------|
| `environment.reward_scaling` | 報酬スケーリング | 1.0 |
| `environment.transaction_cost` | 取引コスト | 0.001 |
| `environment.max_position_size` | 最大ポジションサイズ | 0.05 |
| `environment.reward_clip_value` | 報酬クリップ値 | 5.0 |
| `environment.reward_position_penalty_scale` | ポジションペナルティ | 20.0 |
| `environment.reward_inventory_penalty_scale` | 在庫ペナルティ | 3.0 |
| `environment.reward_trade_frequency_penalty` | 取引頻度ペナルティ | 3.0 |

#### パス設定

| パラメータ | 説明 |
|----------|------|
| `paths.log_dir` | ログディレクトリ |
| `paths.model_dir` | モデルディレクトリ |
| `paths.results_dir` | 結果ディレクトリ |
| `paths.opt_dir` | 最適化ディレクトリ |
| `paths.tensorboard_log` | TensorBoardログディレクトリ |

### リスク管理パラメータの詳細

#### ポジション管理

- **max_position_size**: ポートフォリオに対する最大ポジション比率
- **reward_position_penalty_scale**: ポジション保持に対するペナルティ（取引を促進）
- **reward_inventory_penalty_scale**: 在庫（未決済ポジション）に対するペナルティ

#### 取引頻度制御

- **reward_trade_frequency_penalty**: 取引ごとのペナルティ（過度な取引を抑制）
- **reward_clip_value**: 報酬の最大/最小値を制限

#### コスト考慮

- **transaction_cost**: 実際の手数料を反映したコストモデル
- **reward_scaling**: 全体的な報酬のスケーリング調整

## Pydantic-Based Configuration Schema

The system now uses Pydantic models for type-safe configuration management.

### Schema Components

- **TrainingConfig**: Training parameters (timesteps, environments, learning rate)
- **CheckpointConfig**: Checkpoint management (compression, retention, async saving)
- **StreamingConfig**: Data streaming settings (batch size, buffer policy)
- **EvalConfig**: Evaluation parameters (DSR trials, bootstrap resampling)
- **GlobalConfig**: Container for all configuration sections

### Loading Configuration

```python
from ztb.config.loader import load_config

# Load from YAML file
config = load_config(config_path="config/example.yaml")

# Access typed configuration
print(f"Training timesteps: {config.training.total_timesteps}")
print(f"Checkpoint compression: {config.checkpoint.compress}")
```

### Effective Configuration Dump

To see the merged configuration from all sources:

```bash
python scripts/print_effective_config.py --config config/example.yaml
```

This shows the final configuration after applying priority: CLI > ENV > YAML > defaults.

### JSON Schema Export

The configuration schema can be exported to JSON:

```python
from ztb.config.schema import GlobalConfig
import json

schema = GlobalConfig.model_json_schema()
with open('schema/config_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
```

Or use the convenience script:

```bash
python scripts/dump_config_schema.py
```

## Default Configuration Values

### Trading Defaults

| Component | Parameter | Default Value | Description |
|-----------|-----------|---------------|-------------|
| **Symbol** | Trading pair | `BTC_JPY` | Primary trading instrument |
| **Venue** | Exchange | `coincheck` | Default trading venue |
| **Risk Profile** | Risk management | `aggressive` | Default risk profile (conservative/balanced/aggressive) |

### Risk Profile Settings (Aggressive)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Position Size | 1.0 | Maximum position size multiplier |
| Stop Loss % | 5.0% | Stop loss threshold |
| Take Profit % | 10.0% | Take profit threshold |
| Max Daily Loss % | 20.0% | Maximum daily loss limit |
| Circuit Breaker % | 15.0% | Circuit breaker threshold |

### Training Defaults

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Total Timesteps | 1,000,000 | Total training steps |
| Environments | 4 | Number of parallel environments |
| Eval Frequency | 10,000 | Evaluation interval |
| Learning Rate | 3e-4 | PPO learning rate |
| Batch Size | 64 | Training batch size |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | GAE lambda parameter |
| Clip Range | 0.2 | PPO clip range |
| Seed | 42 | Random seed |
