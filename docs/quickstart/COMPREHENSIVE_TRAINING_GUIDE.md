# 包括的学習ガイド - Zaif Trade Bot

**最終更新**: 2025年10月7日
**対象**: 100k～2M学習、アンサンブル集計、ローリング評価
**目的**: 完全なエンド・ツー・エンドの学習ワークフロー

---

## 📋 目次

1. [概要](#概要)
2. [準備](#準備)
3. [100kテスト（15-30分）](#100kテスト15-30分)
4. [1M学習（3-5時間）](#1m学習3-5時間)
5. [2M学習（6-10時間）](#2m学習6-10時間)
6. [ローリング評価](#ローリング評価)
7. [アンサンブル集計](#アンサンブル集計)
8. [ツール一覧](#ツール一覧)
9. [トラブルシューティング](#トラブルシューティング)
10. [ベストプラクティス](#ベストプラクティス)

---

## 概要

### 🎯 学習フロー

```
100kテスト (15-30分)
    ↓ 動作確認OK
1M学習 (3-5時間) × 3モデル
    ↓ 性能評価
ローリング評価
    ↓ 過学習検出
アンサンブル集計
    ↓ 最終モデル
本番デプロイ
```

### 📦 実装済み機能

#### ✅ コア機能
- **ログレベル制御**: `--log-level INFO/WARNING/ERROR/DEBUG`
- **Checkpoint間隔制御**: `checkpoint_interval`パラメータ
- **Custom PPO統合**: PAN + Target Entropy
- **SELL バイアス対策**: 報酬倍率調整

#### ✅ 新規ツール（本セッション実装）
1. **設定ファイルテンプレート**: `configs/train/template.json`, `template_2M.json`
2. **設定ファイル検証**: `scripts/check_config_consistency.py`
3. **学習進捗監視**: `scripts/watch_training.py`
4. **チェックポイント比較**: `scripts/compare_checkpoints.py`
5. **ローリング評価**: `scripts/rolling_evaluation.py`
6. **アンサンブル集計**: `scripts/ensemble_aggregator.py`

#### ✅ 補足機能（本セッション実装）
1. **Gradient Probe Guard**: SELL勾配ゼロ張り付き検出＆自動停止
   - ゼロ勾配検出 → 自動停止 → 診断データアーカイブ
   - 設定: `enable_grad_probe_guard`, `grad_probe_config`
   - 詳細: [SUPPLEMENTAL_FEATURES_GUIDE.md](SUPPLEMENTAL_FEATURES_GUIDE.md)

2. **Enhanced Ensemble Aggregator**: 失格モデル自動除外
   - Confidence-weighted voting（Sharpe × 信頼度）
   - all-masked多発/低Sharpe → weight=0化
   - 詳細: [SUPPLEMENTAL_FEATURES_GUIDE.md](SUPPLEMENTAL_FEATURES_GUIDE.md)
3. **学習進捗監視**: `scripts/watch_training.py`
4. **チェックポイント比較**: `scripts/compare_checkpoints.py`
5. **ローリング評価**: `scripts/rolling_evaluation.py`
6. **アンサンブル集計**: `scripts/ensemble_aggregator.py`

---

## 準備

### データ確認

```powershell
# データファイル確認
if (Test-Path "ml-dataset-enhanced.csv") {
    echo "✅ Data file exists"
} else {
    echo "❌ Data file not found"
}
```

### 設定ファイル作成

```bash
# テンプレートをコピー
cp configs/train/template.json configs/train/my_training_100k.json

# YOUR_SESSION_ID を置換
# sed, VSCode検索置換、または手動編集
```

### 設定ファイル検証

```bash
# 不整合検出
python scripts/check_config_consistency.py configs/train/my_training_*.json
```

---

## 100kテスト（15-30分）

### 🎯 目的
- unified_trainer.py + CustomPPO動作確認
- パラメータ効果確認
- 問題早期発見

### 実行方法

#### 単一モデルテスト

```bash
# INFOレベル（推奨）
python -m ztb.training.unified_trainer \
    --config configs/train/ensemble_B_100k_test.json \
    --log-level INFO
```

#### 並列実行（3モデル）

```powershell
# WARNINGレベル（静かに実行）
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level WARNING"
```

### 監視

#### TensorBoard

```bash
tensorboard --logdir logs --port 6006
# ブラウザ: http://localhost:6006
```

#### コマンドライン監視（新機能）

```bash
# リアルタイム監視
python scripts/watch_training.py \
    --log-dir logs/ensemble_B_100k_test \
    --interval 10

# コンパクトモード
python scripts/watch_training.py \
    --log-dir logs/ensemble_B_100k_test \
    --compact

# 利用可能なメトリック一覧
python scripts/watch_training.py \
    --log-dir logs/ensemble_B_100k_test \
    --list
```

### 成功基準

**必須条件**:
- ✅ 学習完了（エラーなし）
- ✅ CustomPPO動作（`pan_total_samples > 0`）
- ✅ SELL発生（`legal_sell_rate ≥ 0.05`）
- ✅ 勾配正常（`grad_norm(SELL) ≠ 0`）

**推奨条件**:
- ✅ SELL率 ≥ 10%
- ✅ sharpe_proxy > 0
- ✅ entropy > 0.5

### チェックポイント比較（新機能）

```bash
# 最良チェックポイント特定
python scripts/compare_checkpoints.py \
    --checkpoint-dir checkpoints/ensemble_B_100k_test

# Top 5のみ表示
python scripts/compare_checkpoints.py \
    --checkpoint-dir checkpoints/ensemble_B_100k_test \
    --top 5

# CSV出力
python scripts/compare_checkpoints.py \
    --checkpoint-dir checkpoints/ensemble_B_100k_test \
    --export results.csv
```

---

## 1M学習（3-5時間）

### 🎯 目的
- 本格的な学習（100kの10倍）
- アンサンブル用の多様なモデル作成
- 最終性能評価

### 実行方法

#### シーケンシャル実行

```bash
# モデルA
python -m ztb.training.unified_trainer \
    --config configs/train/ensemble_A_1M.json \
    --log-level INFO

# モデルB
python -m ztb.training.unified_trainer \
    --config configs/train/ensemble_B_1M.json \
    --log-level INFO

# モデルC
python -m ztb.training.unified_trainer \
    --config configs/train/ensemble_C_1M.json \
    --log-level INFO
```

**所要時間**: 9-15時間（3モデル直列）

#### 並列実行（推奨）

```powershell
# WARNINGレベルで静かに実行
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_A_1M.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_B_1M.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_C_1M.json --log-level WARNING"
```

**所要時間**: 3-5時間（3モデル並列）

### チェックポイント設定

```json
{
  "total_timesteps": 1000000,
  "checkpoint_interval": 25000,
  "evaluation": {
    "enabled": true,
    "eval_freq": 25000,
    "n_eval_episodes": 10
  }
}
```

**チェックポイント数**: 40個（25k毎）

### 監視（バックグラウンド実行時）

```bash
# ログファイル確認
tail -f logs/ensemble_B_1M/train.log

# 進捗監視（別ターミナル）
python scripts/watch_training.py \
    --log-dir logs/ensemble_B_1M \
    --interval 30 \
    --compact
```

---

## 2M学習（6-10時間）

### 🎯 目的
- 長期学習による性能向上
- ローリング評価での過学習検出
- 最高性能モデルの探索

### 設定ファイル作成

```bash
# テンプレートをコピー
cp configs/train/template_2M.json configs/train/my_model_2M.json

# YOUR_SESSION_ID_2M を置換
```

### 重要な設定

```json
{
  "total_timesteps": 2000000,
  "checkpoint_interval": 50000,
  "evaluation": {
    "enabled": true,
    "eval_freq": 50000,
    "n_eval_episodes": 20
  },
  "resume": {
    "enabled": true,
    "checkpoint_path": null,
    "comment": "中断時に自動再開する場合、checkpoint_pathを指定"
  },
  "rolling_evaluation": {
    "enabled": true,
    "eval_interval": 100000,
    "eval_data_path": "ml-dataset-enhanced.csv",
    "eval_episodes": 50,
    "save_results": true,
    "results_dir": "eval_results/YOUR_SESSION_ID_2M"
  }
}
```

### 実行方法

```bash
# 通常実行
python -m ztb.training.unified_trainer \
    --config configs/train/my_model_2M.json \
    --log-level WARNING

# 中断後の再開（実装予定）
python -m ztb.training.unified_trainer \
    --config configs/train/my_model_2M.json \
    --resume checkpoints/my_model_2M/checkpoint_1000000 \
    --log-level WARNING
```

### ローリング評価（新機能）

学習完了後、全チェックポイントを評価:

```bash
python scripts/rolling_evaluation.py \
    --checkpoint-dir checkpoints/my_model_2M \
    --data-path ml-dataset-enhanced.csv \
    --n-episodes 100 \
    --output-dir eval_results
```

**出力例**:
```
📊 Rolling Evaluation Summary
================================================================================

Checkpoint                Step       Mean Reward            Std        Sharpe
--------------------------------------------------------------------------------
⭐ checkpoint_1500000     1500000          250.50          50.20        4.9900
   checkpoint_2000000     2000000          245.30          48.10        5.0978  ← 過学習
   checkpoint_1000000     1000000          230.10          55.30        4.1612
   ...
```

**過学習検出**:
- 最後の3チェックポイントで性能悪化 → 早めに学習停止すべき

---

## アンサンブル集計

### 🎯 目的
- 複数モデルの予測を集計
- Confidence-weighted voting
- 最終モデルの性能向上

### 基本的な使い方

#### 3モデルのアンサンブル

```bash
python scripts/ensemble_aggregator.py \
    --model-dirs \
        checkpoints/ensemble_A_1M/checkpoint_1000000 \
        checkpoints/ensemble_B_1M/checkpoint_1000000 \
        checkpoints/ensemble_C_1M/checkpoint_1000000 \
    --method confidence_weighted \
    --eval-data ml-dataset-enhanced.csv \
    --calibrate \
    --n-eval 100 \
    --output ensemble_results.json
```

### 集計方法

#### 1. Majority Vote（多数決）

```bash
--method majority_vote
```

- 各モデルが投票
- 最多票の行動を選択
- シンプルだが、モデルの信頼度を考慮しない

#### 2. Confidence-Weighted Voting（推奨）

```bash
--method confidence_weighted --calibrate
```

- 各モデルの信頼度で重み付け
- 評価データでSharpe ratioを計算し、重みを校正
- 最も性能の良いモデルの影響が大きくなる

#### 3. Soft Voting

```bash
--method soft_voting
```

- 確率分布を平均
- 滑らかな予測

### 重み校正

```bash
# 評価データで各モデルのSharpe ratioを計算し、重みを設定
python scripts/ensemble_aggregator.py \
    --model-dirs checkpoints/ensemble_*/checkpoint_1000000 \
    --method confidence_weighted \
    --eval-data ml-dataset-enhanced.csv \
    --calibrate \
    --n-eval 50
```

**出力例**:
```
📊 Calibrated weights:
  Model 1 (ensemble_A_1M): 0.3500
  Model 2 (ensemble_B_1M): 0.4200  ← 最も性能が良い
  Model 3 (ensemble_C_1M): 0.2300
```

### 評価結果

```json
{
  "method": "confidence_weighted",
  "n_models": 3,
  "model_weights": [0.35, 0.42, 0.23],
  "evaluation": {
    "mean_reward": 275.50,
    "sharpe_ratio": 5.2500,
    "action_distribution": {
      "0": 0.35,  // BUY
      "1": 0.45,  // HOLD
      "2": 0.20   // SELL
    }
  }
}
```

---

## ツール一覧

### 設定管理

| ツール | 用途 | 実行例 |
|--------|------|--------|
| **check_config_consistency.py** | 設定ファイル不整合検出 | `python scripts/check_config_consistency.py configs/train/*.json` |

### 学習監視

| ツール | 用途 | 実行例 |
|--------|------|--------|
| **watch_training.py** | リアルタイム進捗監視 | `python scripts/watch_training.py --log-dir logs/my_model --interval 10` |
| **compare_checkpoints.py** | チェックポイント性能比較 | `python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/my_model` |

### 評価

| ツール | 用途 | 実行例 |
|--------|------|--------|
| **rolling_evaluation.py** | ローリング評価（過学習検出） | `python scripts/rolling_evaluation.py --checkpoint-dir checkpoints/my_model --data-path ml-dataset-enhanced.csv` |

### アンサンブル

| ツール | 用途 | 実行例 |
|--------|------|--------|
| **ensemble_aggregator.py** | アンサンブル集計（confidence-weighted） | `python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_*/checkpoint_1000000 --method confidence_weighted` |

---

## トラブルシューティング

### 問題1: SELL率が低い（< 5%）

**原因**:
- SELL報酬倍率が低すぎる
- データセット内のSELL機会が少ない

**解決策**:

```json
// 設定ファイル修正
{
  "reward_profit_bonus_multipliers": [1.0, 1.0, 1.2]  // SELL倍率を上げる
}
```

または

```bash
# Stratified Sampling有効化
{
  "custom_ppo": {
    "enable_stratified_sampling": true
  }
}
```

---

### 問題2: grad_norm(SELL) = 0（勾配消失）

**原因**:
- SELL行動が全く発生していない
- 勾配が流れていない

**解決策**:

```json
{
  "custom_ppo": {
    "enable_stratified_sampling": true,
    "stratified_min_samples_per_class": 50
  },
  "reward_profit_bonus_multipliers": [1.0, 1.0, 1.5]
}
```

---

### 問題3: メモリ不足

**原因**:
- バッチサイズが大きすぎる
- GPUメモリ不足

**解決策**:

```json
{
  "training": {
    "batch_size": 32,  // 64 → 32
    "n_steps": 1024    // 2048 → 1024
  }
}
```

---

### 問題4: 学習が進まない（reward横ばい）

**原因**:
- 学習率が不適切
- entropy係数が高すぎる/低すぎる

**解決策**:

```json
{
  "training": {
    "learning_rate": 1.0e-4,  // 3.0e-4 → 1.0e-4
    "ent_coef": 0.5            // 0.7 → 0.5
  }
}
```

---

### 問題5: 過学習

**検出方法**:

```bash
# ローリング評価で検出
python scripts/rolling_evaluation.py \
    --checkpoint-dir checkpoints/my_model \
    --data-path ml-dataset-enhanced.csv
```

**対策**:
- 早めのチェックポイントを採用
- 正則化強化（ent_coef上げる）
- データ拡張

---

## ベストプラクティス

### 1. 段階的検証

```
100kテスト（15-30分）
    ↓ 成功
1M学習（3-5時間）
    ↓ 成功
2M学習（6-10時間）
```

**100kで失敗したら1M/2Mに進まない**

---

### 2. 並列実行

```powershell
# 3モデル並列（時間短縮）
Start-Process powershell -ArgumentList "-NoExit", "-Command", "..."
```

**メリット**:
- 時間短縮（3-5時間 vs 9-15時間）
- 多様性確保

---

### 3. ログレベル制御

| 状況 | 推奨レベル |
|------|------------|
| 初回実行 | INFO |
| 並列実行 | WARNING |
| エラー調査 | DEBUG |

---

### 4. チェックポイント戦略

| 学習規模 | checkpoint_interval | チェックポイント数 |
|----------|---------------------|---------------------|
| 100k | 10,000 | 10 |
| 1M | 25,000 | 40 |
| 2M | 50,000 | 40 |

**理由**: チェックポイント数40前後が管理しやすい

---

### 5. アンサンブル設定

```json
// 多様性を持たせる
{
  "ensemble_A": {"ent_coef": 0.6, "seed": 101, "SELL倍率": 0.8},
  "ensemble_B": {"ent_coef": 0.7, "seed": 202, "SELL倍率": 0.9},
  "ensemble_C": {"ent_coef": 0.8, "seed": 303, "SELL倍率": 1.0, "allow_reverse": true}
}
```

---

### 6. 評価データ分離

```
学習データ: ml-dataset-enhanced.csv の 0-80%
検証データ: ml-dataset-enhanced.csv の 80-100%
```

**ローリング評価・アンサンブル校正は検証データで実施**

---

## まとめ

### ✅ 実装済み機能

1. **100k/1M/2M学習** - テンプレート完備
2. **ログレベル制御** - 視認性向上
3. **学習進捗監視** - TensorBoardなしで監視
4. **チェックポイント比較** - 最良モデル特定
5. **ローリング評価** - 過学習検出
6. **アンサンブル集計** - Confidence-weighted voting

### 📊 推奨ワークフロー

```bash
# Week 1: 100kテスト
python -m ztb.training.unified_trainer --config configs/train/ensemble_B_100k_test.json --log-level INFO

# Week 2: 1M学習（3モデル並列）
Start-Process powershell ...

# Week 3: ローリング評価
python scripts/rolling_evaluation.py ...

# Week 4: アンサンブル集計 → 本番デプロイ
python scripts/ensemble_aggregator.py ...
```

---

**すべてのツールとテンプレートが整いました！儲かるモデルを見つけましょう！** 🚀
