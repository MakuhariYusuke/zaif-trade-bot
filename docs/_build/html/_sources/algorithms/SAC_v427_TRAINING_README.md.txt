# SAC v427 学習実行ガイド

## 概要
このガイドでは、報酬関数とパラメータを最適化したSAC v427モデルの学習実行方法について説明します。

## 前提条件
- Python 3.11+
- 必要なパッケージがインストール済み
- 学習データが準備済み (`data/btc_jpy_real_dataset.csv`)

## クイックスタート

### 1. 完全自動学習実行
```bash
# 全フェーズを自動実行（推奨）
python scripts/sac_v427_training_executor.py --config configs/sac_v427_market_adaptive_ensemble.json
```

### 2. 個別フェーズ実行

#### Phase 1: 初期学習のみ
```bash
python scripts/sac_v427_training_executor.py --phase 1 --config configs/sac_v427_market_adaptive_ensemble.json
```

#### Phase 2: ハイパーパラメータ最適化のみ
```bash
python scripts/sac_v427_training_executor.py --phase 2 --config configs/sac_v427_market_adaptive_ensemble.json
```

#### Phase 3: 微調整学習のみ
```bash
python scripts/sac_v427_training_executor.py --phase 3 --config configs/sac_v427_market_adaptive_ensemble.json
```

#### Phase 4: 最終検証のみ
```bash
python scripts/sac_v427_training_executor.py --phase 4 --config configs/sac_v427_market_adaptive_ensemble.json
# モデルパスを入力するよう求められます
```

## 学習パラメータのカスタマイズ

### 設定ファイルの編集
`configs/sac_v427_market_adaptive_ensemble.json` を編集してパラメータを変更できます：

```json
{
  "total_timesteps": 100000,  // Phase 1のステップ数
  "sac_hyperparameters": {
    "learning_rate": 0.0003,
    "buffer_size": 50000,
    "batch_size": 256
  }
}
```

### 環境変数での制御
```bash
# CPU学習の場合
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4

# メモリ最適化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

## 学習監視

### TensorBoardでの監視
```bash
# TensorBoard起動
tensorboard --logdir ./tensorboard/sac_v427

# ブラウザで http://localhost:6006 にアクセス
```

### 主要監視指標
- `episode_reward`: エピソードごとの報酬
- `eval/mean_reward`: 評価時の平均報酬
- `train/learning_rate`: 学習率の推移
- `train/ent_coef`: エントロピー係数の推移

## 学習結果の確認

### 学習ログの確認
```bash
# 学習ログ確認
tail -f logs/training_*.log
```

### 結果ファイルの確認
学習結果は `results/training/` ディレクトリに保存されます：

```
results/training/
├── sac_v427_training_results_full.json    # 完全実行結果
├── sac_v427_training_results_1.json       # Phase 1結果
├── sac_v427_training_results_2.json       # Phase 2結果
└── ...
```

### モデルファイルの確認
学習済みモデルは `models/` ディレクトリに保存されます：

```
models/
├── sac_v427_foundation.zip      # Phase 1モデル
├── sac_v427_optimized.zip       # Phase 2最適化モデル
└── sac_v427_final.zip          # Phase 3最終モデル
```

## トラブルシューティング

### メモリ不足の場合
```bash
# バッファサイズを減らす
export SAC_BUFFER_SIZE=25000

# バッチサイズを減らす
export SAC_BATCH_SIZE=128
```

### 学習が不安定な場合
```bash
# 学習率を下げる
export SAC_LEARNING_RATE=0.0001

# エントロピー係数を調整
export SAC_ENT_COEF=0.1
```

### GPUメモリ不足の場合
```bash
# CPU学習に切り替え
export CUDA_VISIBLE_DEVICES=""
export SAC_DEVICE=cpu
```

## パフォーマンス最適化

### 並列学習の有効化
```bash
# Ray Tune並列数設定
export RAY_NUM_CPUS=4
export RAY_NUM_GPUS=0

# 学習実行
python scripts/sac_v427_training_executor.py --phase 2
```

### メモリ使用量の監視
```bash
# メモリ監視スクリプト実行
python scripts/monitor_memory.py
```

## バックテスト検証

学習完了後にバックテストを実行：

```bash
# 最終モデルのバックテスト
python scripts/run_backtest.py \
  --model-path models/sac_v427_final.zip \
  --data-path data/btc_jpy_real_dataset.csv \
  --config configs/sac_v427_market_adaptive_ensemble.json
```

## 評価基準

学習成功の目安：
- **Sharpe Ratio**: > 1.5
- **Total Return**: > 50% (1年シミュレーション)
- **Win Rate**: > 55%
- **Maximum Drawdown**: < 20%

## 次のステップ

1. **モデル改善**: 学習結果に基づいて報酬関数を改良
2. **特徴拡張**: 追加の市場指標を統合
3. **アンサンブル学習**: 複数モデルの組み合わせを検討
4. **本番展開**: 検証完了後に本番環境への展開

## サポート

学習中に問題が発生した場合：
1. ログファイルを確認 (`logs/training_*.log`)
2. TensorBoardで学習曲線を確認
3. 設定パラメータを調整
4. 必要に応じてGitHub Issuesで報告

---

**最終更新**: 2025年10月19日
**バージョン**: SAC v427 Training Executor v1.0</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\SAC_v427_TRAINING_README.md
