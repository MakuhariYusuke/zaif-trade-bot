# Unified Training System (v3.5.0)

このシステムは、Zaif Trade Bot の複数のトレーニングアプローチを統合したものです。様々なアルゴリズムとトレード戦略を統一的なインターフェースで扱えます。

## 🎯 最新機能 (v3.4.0)

### 包括的評価フレームワーク統合

トレーニング完了後のモデル評価に、6つの専門分析モジュールを統合：

- **Performance Attribution**: 収益源泉の詳細分解
- **Monte Carlo Simulation**: 確率的リスク評価
- **Strategy Robustness**: 市場変動耐性テスト
- **Benchmark Comparison**: 業界標準との比較
- **Risk Parity Analysis**: ポートフォリオ最適化
- **Cost Sensitivity**: 取引コスト影響分析

### 評価実行例

```bash
# トレーニング済みモデルの包括的評価
python comprehensive_benchmark.py --data ml-dataset-enhanced.csv --single-model models/trained_model.zip --episodes 10 --output-dir evaluation_results

# 進捗バー付きクロスバリデーション
python comprehensive_benchmark.py --data ml-dataset-enhanced.csv --single-model models/trained_model.zip --cv-folds 5 --output-dir cv_results
```

詳細: [comprehensive_benchmark.py](../comprehensive_benchmark.py), [CHANGELOG.md](../CHANGELOG.md)

## 🚀 実行マニュアル

### クイックスタート

#### 1. 環境準備
```bash
# リポジトリのクローン
git clone https://github.com/MakuhariYusuke/zaif-trade-bot.git
cd zaif-trade-bot

# 依存関係のインストール
pip install -r requirements.txt
```

#### 2. 設定確認
現在の設定はスキャルピングモードに最適化されています：
```json
{
  "algorithm": "ppo",
  "trading_mode": "scalping",
  "timeframe": "1d",
  "total_timesteps": 10000,
  "iterations": 1,
  "steps_per_iteration": 10000,
  "batch_size": 32,
  "n_steps": 1024,
  "reward_scaling": 1.2,
  "ent_coef": 0.05,
  "max_position_size": 0.3,
  "transaction_cost": 0.002,
  "curriculum_stage": "profit_only"
}
```

#### 3. トレーニング実行

**ローカル実行（メモリ8GB以上推奨）:**
```bash
# 基本実行
python -m ztb.training.unified_trainer --config unified_training_config.json --force

# ストリーミング有効化（メモリ節約）
python -m ztb.training.unified_trainer --config unified_training_config.json --force --enable-streaming --stream-batch-size 64
```

**分割実行（メモリ不足時）:**
```bash
# 1Mステップを10kステップ×100チャンクに分割
chmod +x split_training.sh
./split_training.sh
```

**クラウド実行（Google Colab推奨）:**
```bash
# Google Colabで実行
!git clone https://github.com/MakuhariYusuke/zaif-trade-bot.git
%cd zaif-trade-bot
!pip install -r requirements.txt
!python -m ztb.training.unified_trainer --config unified_training_config.json --force
```

### 環境変数設定（メモリ最適化）

```bash
# PyTorchメモリ最適化
export PYTORCH_DISABLE_TORCH_DYNAMO=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CPU専用モード（メモリ節約）
export CUDA_VISIBLE_DEVICES=""
```

### 結果確認

```bash
# チェックポイント確認
ls -la checkpoints/scalping_training_v2/

# ログ確認
tail -f logs/scalping_training_v2.log

# TensorBoard（オプション）
tensorboard --logdir logs/
```

## � ライブトレーディング

### 準備

1. **APIキー設定** (オプション - 設定しない場合はデモモード)
```bash
export COINCHECK_API_KEY="your_api_key"
export COINCHECK_API_SECRET="your_api_secret"
export DISCORD_WEBHOOK="your_webhook_url"  # 通知用（オプション）
```

### クロスプラットフォーム対応

WindowsおよびRaspberry Piで動作可能です。自動的に環境を検知して最適化されます。

### ログ確認

```bash
# ライブトレーディングログ
tail -f logs/live_trading_*.log
```

## �🔧 トラブルシューティング

### メモリ不足エラー
```
KeyboardInterrupt during PyTorch import
```

**解決策:**
1. **クラウド環境を使用**（推奨）
2. **RAMを16GB以上に増設**
3. **不要プロセスを終了**
4. **スワップファイル設定**（Linux/Mac）

### CUDA関連エラー
```
CUDA out of memory
```

**解決策:**
```bash
# CPU専用モード
export CUDA_VISIBLE_DEVICES=""
python -m ztb.training.unified_trainer --config unified_training_config.json --force
```

### データ読み込みエラー
```
FileNotFoundError: ml-dataset-enhanced.csv
```

**解決策:**
```bash
# データファイルの存在確認
ls -la ml-dataset-enhanced.csv

# またはデータパスを指定
python -m ztb.training.unified_trainer --config unified_training_config.json --data-path /path/to/data.csv --force
```

### トレーニング中断時の再開
```bash
# 同じセッションIDで再実行（自動的にチェックポイントから再開）
python -m ztb.training.unified_trainer --config unified_training_config.json --force
```

### パフォーマンス監視
```bash
# メモリ使用量監視
watch -n 1 'ps aux | grep python'

# GPU使用量監視（nvidia-smiがある場合）
watch -n 1 nvidia-smi
```

## 📋 コマンドチートシート

### 基本実行
```bash
# 通常実行
python -m ztb.training.unified_trainer --config unified_training_config.json --force

# ストリーミング有効
python -m ztb.training.unified_trainer --config unified_training_config.json --force --enable-streaming --stream-batch-size 64

# 分割実行
./split_training.sh

# クラウド実行
./cloud_training.sh
```

### 設定変更
```bash
# タイムフレーム変更
sed -i 's/"timeframe": "1d"/"timeframe": "5m"/g' unified_training_config.json

# ステップ数変更
sed -i 's/"total_timesteps": 10000/"total_timesteps": 100000/g' unified_training_config.json

# セッションID変更
sed -i 's/"session_id": "scalping_training_v2"/"session_id": "my_training_session"/g' unified_training_config.json
```

### 結果確認
```bash
# 最新チェックポイント
ls -lt checkpoints/scalping_training_v2/checkpoint_* | head -5

# ログ監視
tail -f logs/scalping_training_v2.log

# トレーニング進捗
grep "Training scalping_training_v2" logs/scalping_training_v2.log | tail -1
```

## ⚠️ 重要: メモリ要件について

**現在の設定では8GB以上のRAMを推奨します**

ローカル環境でのトレーニング実行時にメモリ不足が発生する場合があります。特に：
- PyTorchの初期化に失敗する
- 大規模データセットの読み込みで中断される
- CUDA/CPUメモリが不足する

### 推奨される解決策

1. **クラウド環境の使用**（推奨）
   - Google Colab Pro+ (有料、高メモリ)
   - AWS EC2, GCP, Azure VM
   - メモリ16GB以上のインスタンス

2. **ローカル環境の改善**
   - RAMを16GB以上に増強
   - スワップファイルの設定
   - 不要なプロセスを終了

3. **設定の最適化**
   - `timeframe: "1d"`（データ量を最小化）
   - `batch_size: 32`（メモリ使用量削減）
   - `n_steps: 1024`（ステップ数を削減）

### クラウド実行スクリプト

`cloud_training.sh` を使用してクラウド環境で実行してください。

```bash
# クラウド環境での実行例
chmod +x cloud_training.sh
./cloud_training.sh
```

## 特徴

- **アルゴリズム選択**: PPO, Base ML, Iterativeトレーニングをサポート
- **トレードモード**: スキャルピングと通常トレードを自動設定
- **長時間実行警告**: 安全な長時間実行のための確認機能
- **柔軟な設定**: JSON設定ファイルによるカスタマイズ
- **メモリ最適化**: 大規模トレーニングに対応
- **チェックポイント管理**: トレーニングの中断・再開が可能

## トレードモード

### スキャルピングモード (`trading_mode: "scalping"`)

- **概要**: 高頻度取引向けの最適化設定
- **特徴量セット**: `scalping`（高速取引向け指標）
- **時間枠**: `15s`（短期分析）
- **ポジションサイズ**: 0.3（小規模ポジション）
- **取引コスト**: 0.002（高コスト考慮）
- **学習ステップ**: 1,000,000（長時間トレーニング）
- **ユースケース**: 短期スキャルピング戦略

### 通常トレードモード (`trading_mode: "normal"`)

- **概要**: 標準的な取引向けの汎用設定
- **特徴量セット**: `full`（全特徴量使用）
- **時間枠**: `1m`（中期分析）
- **ポジションサイズ**: 1.0（フルポジション）
- **取引コスト**: 0.001（標準コスト）
- **学習ステップ**: 100,000（標準トレーニング）
- **ユースケース**: 一般的な取引戦略

## サポートされるアルゴリズム

### 1. PPO Training (`algorithm: "ppo"`)

- **概要**: Stable Baselines3 の PPO アルゴリズムを使用した標準的な強化学習トレーニング
- **特徴**:
  - 評価機能付きトレーニング
  - 定期的なチェックポイント保存
  - TensorBoard ログ出力
  - メモリ最適化
- **ユースケース**: 標準的な強化学習トレーニング
- **基盤**: `ztb/trading/ppo_trainer.py`

### 2. Base ML Reinforcement (`algorithm: "base_ml"`)

- **概要**: ベース ML 強化学習実験フレームワーク
- **特徴**:
  - シンプルなステップベースのトレーニングループ
  - 実験管理機能（チェックポイント、再開）
  - 拡張可能なベースクラス
- **ユースケース**: カスタム強化学習実験、プロトタイピング
- **基盤**: `ztb/training/entrypoints/base_ml_reinforcement.py`
- **注意**: **現在はダミー実装（ランダム報酬）** - 開発中
- **ステータス**: 非推奨（PPOを使用してください）

### 3. Iterative Training (`algorithm: "iterative"`)

- **概要**: 反復トレーニングセッション（1M timesteps 用）
- **特徴**:
  - 複数イテレーションのトレーニング
  - 再開機能
  - データ検証
  - Discord 通知
  - ストリーミングデータ対応
- **ユースケース**: 長時間実行トレーニング、本番環境トレーニング
- **基盤**: `ztb/training/run_1m.py`

## 推奨アルゴリズム

### 本番環境での使用

1. **PPO** (`algorithm: "ppo"`) - **推奨**
   - 安定した学習と高いパフォーマンス
   - メモリ最適化済み
   - TensorBoard統合

2. **Iterative** (`algorithm: "iterative"`) - **本番推奨**
   - 長時間トレーニング対応
   - チェックポイント自動保存
   - Discord通知機能

3. **Base ML** (`algorithm: "base_ml"`) - **非推奨**
   - 現在ダミー実装
   - 開発・実験用のみ

### ユースケース別推奨

- **新規プロジェクト**: PPOから開始
- **本番デプロイ**: Iterativeを使用
- **研究・実験**: PPOまたはBase ML（開発後）

## 使用方法

### 基本的な使用法

```bash
python -m ztb.training.unified_trainer --config unified_training_config.json
```

### トレードモード別の実行

#### スキャルピングトレーニング
```bash
python -m ztb.training.unified_trainer --config unified_training_config.json
# デフォルトでスキャルピングモード
```

#### 通常トレーニング
```bash
python -m ztb.training.unified_trainer --config unified_training_config_normal.json
```

### アルゴリズムの指定

```bash
python -m ztb.training.unified_trainer --config config.json --algorithm ppo
```

### 設定の上書き

```bash
python -m ztb.training.unified_trainer \
  --config unified_training_config.json \
  --data-path your_data.csv \
  --total-timesteps 500000 \
  --session-id my_training_session
```

### 追加オプション

- `--force`: 長時間実行警告をスキップ
- `--dry-run`: 設定検証のみ実行（トレーニングなし）
- `--enable-streaming`: ストリーミングデータ使用
- `--max-features N`: 最大特徴量数を制限

## 設定ファイル

設定ファイルは JSON 形式で、以下の構造を持ちます：

```json
{
  "algorithm": "ppo",
  "data_path": "ml-dataset-enhanced.csv",
  "session_id": "training_session",
  "total_timesteps": 100000,
  "checkpoint_dir": "checkpoints",
  "log_dir": "logs",
  "model_dir": "models",
  "tensorboard_log": "tensorboard",
  "verbose": 1,
  "seed": 42,
  "learning_rate": 0.0003,
  "n_steps": 2048,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "ent_coef": 0.0,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "reward_scaling": 1.0,
  "transaction_cost": 0.001,
  "max_position_size": 0.05,
  "eval_freq": 10000,
  "n_eval_episodes": 10,
  "checkpoint_interval": 5000
}
```

## 既存スクリプトの統合

この統合スクリプトは以下の既存スクリプトを置き換えます：

- `ztb/trading/ppo_trainer.py` (PPO トレーニング)
- `ztb/training/entrypoints/base_ml_reinforcement.py` (ベース ML)
- `ztb/training/run_1m.py` (反復トレーニング)

既存スクリプトは後方互換性のために残されますが、新規開発ではこの統合スクリプトを使用することを推奨します。

## 利点

- **単一のインターフェース**: 異なるトレーニングアプローチを統一された方法で実行
- **設定の一元化**: JSON 設定ファイルによる柔軟な構成
- **保守性の向上**: コード重複の削減
- **拡張性**: 新しいアルゴリズムの容易な追加
 
 
 
 
 
 
