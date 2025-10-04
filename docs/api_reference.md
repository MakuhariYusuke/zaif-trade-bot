# API Reference

このドキュメントは、Zaif Trade Botの主要クラスの使用方法とAPIリファレンスを提供します。

## 目次

1. [PPOTrainer](#ppotrainer)
2. [HeavyTradingEnv](#heavytradingenv)
3. [FeatureRegistry](#featureregistry)

## PPOTrainer

PPO (Proximal Policy Optimization) アルゴリズムを使用した強化学習トレーナー。1Mステップの長時間トレーニングをサポート。

### PPOTrainerの基本的な使用例

```python
from ztb.trading.ppo_trainer import PPOTrainer
import pandas as pd

# データ読み込み
df = pd.read_csv('ml-dataset-enhanced.csv')

# トレーナー初期化
trainer = PPOTrainer(
    df=df,
    config={
        'total_timesteps': 1000000,
        'batch_size': 64,
        'n_steps': 2048,
        'learning_rate': 0.0003,
        'ent_coef': 0.0,
        'verbose': 1
    }
)

# トレーニング実行
trainer.train()

# モデル保存
trainer.save('models/scalping_model.zip')
```

### PPOTrainerの高度な使用例（ストリーミング対応）

```python
from ztb.data.streaming_pipeline import StreamingPipeline

# ストリーミングパイプライン設定
streaming_pipeline = StreamingPipeline(
    batch_size=256,
    buffer_policy='drop_oldest'
)

# トレーナー初期化（ストリーミングモード）
trainer = PPOTrainer(
    df=None,  # ストリーミング時はNone
    streaming_pipeline=streaming_pipeline,
    config={
        'total_timesteps': 1000000,
        'enable_streaming': True,
        'stream_batch_size': 256,
        'checkpoint_interval': 10000,
        'async_checkpoint': True,
        'checkpoint_compression': 'zstd'
    }
)

# トレーニング実行
trainer.train()
```

### 設定パラメータ

| パラメータ | 型 | 説明 | デフォルト値 |
|----------|----|------|------------|
| `total_timesteps` | int | 総トレーニングステップ数 | 1000000 |
| `batch_size` | int | バッチサイズ | 64 |
| `n_steps` | int | 各環境のステップ数 | 2048 |
| `learning_rate` | float | 学習率 | 0.0003 |
| `ent_coef` | float | エントロピー係数 | 0.0 |
| `clip_range` | float | PPOクリップ範囲 | 0.2 |
| `n_epochs` | int | エポック数 | 10 |
| `gae_lambda` | float | GAEラムダ | 0.95 |
| `max_grad_norm` | float | 最大勾配ノルム | 0.5 |
| `vf_coef` | float | 価値関数係数 | 0.5 |
| `verbose` | int | ログレベル | 1 |
| `seed` | int | 乱数シード | 42 |

### チェックポイント機能

```python
# 非同期チェックポイント有効化
trainer = PPOTrainer(
    df=df,
    config={
        'checkpoint_interval': 10000,
        'async_checkpoint': True,
        'checkpoint_compression': 'zstd',
        'max_pending_checkpoints': 1
    }
)

# トレーニング再開
trainer.resume_from_checkpoint('checkpoints/checkpoint_50000.zip')
```

### リソース監視

```python
# リソース監視有効化
trainer = PPOTrainer(
    df=df,
    config={
        'resource_monitoring': True,
        'resource_log_interval': 10,
        'alert_thresholds': {
            'cpu_pct': 80.0,
            'memory_pct': 85.0,
            'gpu_util_pct': 90.0
        }
    }
)
```

## HeavyTradingEnv

強化学習用のカスタム取引環境。66次元の特徴量を使用した高度な状態表現を提供。

### HeavyTradingEnvの基本的な使用例

```python
import gym
from ztb.trading.environment import HeavyTradingEnv
import pandas as pd

# データ読み込み
df = pd.read_csv('ml-dataset-enhanced.csv')

# 環境初期化
env = HeavyTradingEnv(
    df=df,
    config={
        'reward_scaling': 1.0,
        'transaction_cost': 0.0005,
        'max_position_size': 0.5,
        'reward_clip_value': 5.0
    }
)

# 環境情報
print(f"状態空間: {env.observation_space}")
print(f"行動空間: {env.action_space}")

# エピソード実行
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # ランダム行動
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"総報酬: {total_reward}")
```

### HeavyTradingEnvの高度な使用例（カスタム報酬関数）

```python
from ztb.trading.environment import HeavyTradingEnv

# カスタム報酬設定
env = HeavyTradingEnv(
    df=df,
    config={
        'reward_type': 'scalping',
        'reward_position_penalty_scale': 20.0,
        'reward_inventory_penalty_scale': 3.0,
        'reward_trade_frequency_penalty': 3.0,
        'reward_max_consecutive_trades': 10,
        'reward_consecutive_trade_penalty': 0.1
    }
)

# 報酬関数のカスタマイズ
def custom_reward_function(self, action, pnl, position, atr):
    # カスタム報酬計算ロジック
    base_reward = pnl * position / (atr + 1e-6)

    # 追加ペナルティ
    penalty = 0
    if abs(position) > 0.8:  # 過度なポジション
        penalty += 0.1

    return base_reward - penalty

# 環境に適用
env.reward_function = custom_reward_function.__get__(env, HeavyTradingEnv)
```

### ストリーミング対応

```python
from ztb.data.streaming_pipeline import StreamingPipeline

# ストリーミングパイプライン
streaming_pipeline = StreamingPipeline(
    batch_size=256,
    buffer_policy='drop_oldest'
)

# ストリーミング環境
env = HeavyTradingEnv(
    df=None,  # ストリーミング時はNone
    streaming_pipeline=streaming_pipeline,
    stream_batch_size=256
)

# リアルタイムデータ処理
for batch in streaming_pipeline:
    obs = env.process_streaming_batch(batch)
    # 強化学習ステップ
    action = model.predict(obs)
    next_obs, reward, done, info = env.step(action)
```

### HeavyTradingEnvの設定パラメータ

| パラメータ | 型 | 説明 | デフォルト値 |
|----------|----|------|------------|
| `reward_scaling` | float | 報酬スケーリング | 1.0 |
| `transaction_cost` | float | 取引コスト | 0.001 |
| `max_position_size` | float | 最大ポジションサイズ | 0.05 |
| `reward_clip_value` | float | 報酬クリップ値 | 5.0 |
| `reward_position_penalty_scale` | float | ポジション保持ペナルティ | 20.0 |
| `reward_inventory_penalty_scale` | float | 在庫ペナルティ | 3.0 |
| `reward_trade_frequency_penalty` | float | 取引頻度ペナルティ | 3.0 |
| `reward_max_consecutive_trades` | int | 最大連続取引数 | 10 |
| `reward_consecutive_trade_penalty` | float | 連続取引ペナルティ | 0.1 |

### 状態表現

```python
# 状態ベクトルの構造
state = env.reset()
print(f"状態次元: {len(state)}")

# 特徴量の確認
feature_names = env.get_feature_names()
print(f"特徴量: {feature_names[:10]}...")  # 最初の10個

# 状態の正規化
normalized_state = env.normalize_state(state)
```

## FeatureRegistry

特徴量計算関数の一元管理レジストリ。66種類のテクニカル指標と価格ベースの特徴量を提供。

### 基本的な使用例

```python
from ztb.features.registry import FeatureRegistry
import pandas as pd

# レジストリ初期化
FeatureRegistry.initialize(
    seed=42,
    cache_enabled=True,
    parallel_enabled=True
)

# データ読み込み
df = pd.read_csv('ml-dataset-enhanced.csv')

# 利用可能な特徴量一覧
available_features = FeatureRegistry.list_features()
print(f"利用可能特徴量: {len(available_features)}")
print(f"例: {available_features[:5]}")

# 特徴量計算
features_to_compute = ['rsi', 'sma_short', 'sma_long', 'adx', 'macd']
computed_features = FeatureRegistry.compute_features(
    df=df,
    feature_names=features_to_compute,
    parallel=True
)

print(f"計算された特徴量: {list(computed_features.keys())}")
```

### 高度な使用例（カスタム特徴量追加）

```python
from ztb.features.registry import FeatureRegistry

# カスタム特徴量関数の定義
def custom_rsi_divergence(prices, period=14):
    """カスタムRSIダイバージェンス指標"""
    rsi = FeatureRegistry.compute_single_feature('rsi', prices, period=period)
    price_change = prices.pct_change(period)

    # ダイバージェンス計算ロジック
    divergence = rsi - price_change * 100
    return divergence

# レジストリに登録
FeatureRegistry.register_feature('custom_rsi_divergence', custom_rsi_divergence)

# 使用
df_with_custom = FeatureRegistry.compute_features(
    df=df,
    feature_names=['rsi', 'custom_rsi_divergence']
)
```

### パフォーマンス最適化

```python
# キャッシュ有効化
FeatureRegistry.initialize(
    cache_enabled=True,
    parallel_enabled=True,
    config={
        'max_parallel_workers': 4,
        'parallel_batch_size': 20,
        'feature_chunk_size': 20,
        'memory_monitor_enabled': True,
        'gc_collect_interval': 10
    }
)

# 大規模データ処理
large_df = pd.read_csv('large_dataset.csv')

# メモリ効率的な処理
features = FeatureRegistry.compute_features_efficient(
    df=large_df,
    feature_names=['rsi', 'sma_short', 'sma_long', 'adx'],
    chunk_size=10000,  # 10k行ずつ処理
    use_cache=True
)
```

### FeatureRegistryの設定パラメータ

| パラメータ | 型 | 説明 | デフォルト値 |
|----------|----|------|------------|
| `seed` | int | 乱数シード | 42 |
| `cache_enabled` | bool | キャッシュ有効化 | True |
| `parallel_enabled` | bool | 並列処理有効化 | True |
| `max_parallel_workers` | int | 最大並列ワーカー数 | CPUコア数 |
| `parallel_batch_size` | int | 並列バッチサイズ | 20 |
| `feature_chunk_size` | int | 特徴量チャンクサイズ | 20 |
| `memory_monitor_enabled` | bool | メモリ監視有効化 | False |
| `gc_collect_interval` | int | GC収集間隔 | 10 |

### 特徴量一覧

#### 価格ベース特徴量

- `close`, `high`, `low`, `open`, `volume`
- `price`, `qty`, `pnl`, `win`

#### テクニカル指標

- **トレンド**: `sma_short`, `sma_long`, `ema_5`, `tema`
- **オシレーター**: `rsi`, `stoch`, `cci`, `mfi`
- **ボラティリティ**: `atr`, `bb_upper`, `bb_lower`, `hv`
- **モメンタム**: `macd`, `roc`, `williams_r`
- **出来高**: `obv`, `vwap`, `price_volume_corr`

#### 詳細な使用例

```python
# 特定の特徴量の詳細設定
rsi_features = FeatureRegistry.compute_single_feature(
    'rsi',
    prices=df['close'],
    period=14,
    normalize=True
)

# 複数タイムフレーム特徴量
multi_tf_features = FeatureRegistry.compute_multi_timeframe_features(
    df=df,
    base_features=['rsi', 'sma'],
    timeframes=['5m', '15m', '1h']
)

# 特徴量の相関分析
correlation_matrix = FeatureRegistry.analyze_feature_correlations(
    df=df,
    features=['rsi', 'sma_short', 'adx', 'macd']
)
print(correlation_matrix)
```

このAPIリファレンスは継続的に更新されます。新しい機能が追加された場合は適宜更新してください。
