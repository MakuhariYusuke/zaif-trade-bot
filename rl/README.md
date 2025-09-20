# Heavy Trading RL Project

重特徴量ベースの取引環境での強化学習プロジェクト

## 概要

このプロジェクトは、Parquet形式の特徴量データを用いた重特徴量ベースの取引環境で、PPO (Proximal Policy Optimization) を使用した強化学習を実装しています。

### 主な特徴

- **重特徴量状態ベクトル**: 価格系・テクニカル系・リスク系のすべての特徴量を使用
- **リスク調整リワード**: ATRベースのボラティリティスケーリング
- **PPO学習**: 安定した学習のための最新の強化学習アルゴリズム
- **包括的な評価**: 複数指標でのパフォーマンス評価と可視化
- **ハイパーパラメータ最適化**: Optunaを使用した自動最適化

## ファイル構成

```
rl/
├── main.py                 # メインスクリプト
├── train_ppo.py           # PPOトレーニングスクリプト
├── evaluate_model.py      # 評価と可視化スクリプト
├── optimize_params.py     # ハイパーパラメータ最適化スクリプト
├── envs/
│   └── heavy_trading_env.py # 重特徴量取引環境
└── __pycache__/

rl_config.json            # 設定ファイル
```

## インストール

### 必要条件

- Python 3.11+
- pip

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. データの準備

特徴量データはParquetまたはCSV形式で以下の構造である必要があります：

```python
# 必須列
- ts: タイムスタンプ
- price: 価格
- その他特徴量（テクニカル指標など）
```

### 2. 設定ファイル

`rl_config.json` で設定をカスタマイズ：

```json
{
  "data": {
    "train_data": "./data/train_features.parquet",
    "test_data": "./data/test_features.parquet"
  },
  "training": {
    "total_timesteps": 200000,
    "learning_rate": 0.0003
  }
}
```

### 3. トレーニング

#### 基本トレーニング

```bash
python rl/main.py --mode train --data ./data/train_features.parquet
```

#### ハイパーパラメータ最適化

```bash
python rl/main.py --mode optimize --data ./data/train_features.parquet --n-trials 100
```

### 4. 評価

#### モデルの評価

```bash
python rl/main.py --mode evaluate --model ./models/best_model --data ./data/test_features.parquet
```

#### モデル比較

```bash
python rl/main.py --mode compare --models model1.zip model2.zip --data ./data/test_features.parquet
```

## 環境仕様

### 状態空間

- **特徴量ベクトル**: すべての利用可能な特徴量を使用
- **次元**: 特徴量数に応じて動的に決定
- **型**: float32

### 行動空間

- **離散行動**: 3つの選択肢
  - 0: ホールド (ポジション維持)
  - 1: 買い (ロングポジション)
  - 2: 売り (ショートポジション)

### リワード関数

```
reward = (position * pnl) / (atr_14 + ε) * reward_scaling
```

- `position`: -1 (ショート), 0 (フラット), 1 (ロング)
- `pnl`: ポジションごとの損益
- `atr_14`: 14期間ATR（リスク調整）
- `ε`: 数値安定性のための小さな値

## トレーニングスクリプト

### PPOトレーニング

```bash
python rl/train_ppo.py --data ./data/train_features.parquet
```

### 評価スクリプト

```bash
python rl/evaluate_model.py --model ./models/best_model --data ./data/test_features.parquet
```

### 最適化スクリプト

```bash
python rl/optimize_params.py --data ./data/train_features.parquet --n-trials 100
```

## 設定パラメータ

### トレーニングパラメータ

- `total_timesteps`: 総トレーニングステップ数 (デフォルト: 200,000)
- `learning_rate`: 学習率 (デフォルト: 0.0003)
- `batch_size`: バッチサイズ (デフォルト: 64)
- `n_steps`: 各更新でのステップ数 (デフォルト: 2048)
- `gamma`: 割引率 (デフォルト: 0.99)
- `gae_lambda`: GAEパラメータ (デフォルト: 0.95)

### 環境パラメータ

- `reward_scaling`: リワードスケーリング係数 (デフォルト: 1.0)
- `transaction_cost`: 取引コスト (デフォルト: 0.001 = 0.1%)
- `max_position_size`: 最大ポジションサイズ (デフォルト: 1.0)

## 出力とログ

### ログディレクトリ

```
logs/               # トレーニングログ
models/            # 保存されたモデル
results/           # 評価結果
optimization/      # 最適化結果
tensorboard/       # TensorBoardログ
```

### 評価指標

- **平均リワード**: エピソードごとの平均リワード
- **平均PnL**: エピソードごとの平均損益
- **シャープレシオ**: リスク調整リターン
- **総取引数**: 実行された取引の総数
- **勝率**: ポジティブリワードの割合

## TensorBoard監視

トレーニング中にTensorBoardで監視可能：

```bash
tensorboard --logdir ./tensorboard/
```

## トラブルシューティング

### よくある問題

1. **メモリ不足**: `batch_size` または `n_steps` を減らす
2. **学習が不安定**: `learning_rate` を調整するか、`clip_range` を変更
3. **収束しない**: 特徴量のスケーリングを確認するか、`reward_scaling` を調整

### パフォーマンス最適化

- GPU使用: PyTorchが自動的にGPUを使用
- 並列評価: `n_eval_episodes` を増やす
- メモリ効率: データ型をfloat32に維持

## 拡張性

### カスタム環境

`heavy_trading_env.py` を継承してカスタム環境を作成：

```python
from rl.envs.heavy_trading_env import HeavyTradingEnv

class CustomTradingEnv(HeavyTradingEnv):
    def _calculate_reward(self, pnl, old_position):
        # カスタムリワード計算
        return custom_reward_function(pnl, old_position)
```

### 新しい特徴量

特徴量データを更新するだけで、新しい特徴量が自動的に使用されます。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 連絡先

質問やフィードバックはIssueを作成してください。