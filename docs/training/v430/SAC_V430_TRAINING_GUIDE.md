# SAC v430 Advanced Training Guide

## 🚀 効率的な学習方法の提案

SAC v430の学習をより効率的に行うための高度な手法を実装しました。

### 📚 利用可能な学習モード

#### 1. **カリキュラム学習 (Curriculum Learning)**
段階的に難易度を上げる学習方法。初心者から上級者への学習をシミュレート。

```bash
python train_v430_advanced.py --config configs/v430/sac_v430_optimized.json --mode curriculum
```

**特徴:**
- 4段階のカリキュラム（warmup → foundation → optimization → refinement）
- 各段階で成功基準を満たすまで進まない
- 徐々に複雑な報酬関数と学習パラメータを使用

#### 2. **マルチステージ学習 (Multi-Stage Training)**
異なる目的で段階的に学習。

```bash
python train_v430_advanced.py --config configs/v430/sac_v430_optimized.json --mode multi_stage
```

**特徴:**
- Stage 1: 探索重視（高エントロピー）
- Stage 2: 活用重視（低エントロピー）
- Stage 3: 微調整（低学習率）

#### 3. **アンサンブル学習 (Ensemble Training)**
複数のモデルを並列学習し、アンサンブル化。

```bash
python train_v430_advanced.py --config configs/v430/sac_v430_optimized.json --mode ensemble
```

**特徴:**
- 5つの異なるシードでモデルを学習
- 並列処理で効率化
- アンサンブル設定ファイルを自動生成

#### 4. **標準学習 (Standard Training)**
最適化された設定で通常の学習。

```bash
python train_v430_advanced.py --config configs/v430/sac_v430_optimized.json --mode standard
```

### 🛠️ 追加の最適化機能

`sac_v430_training_optimizations.py` で提供される高度な最適化:

#### **グラジエント蓄積 (Gradient Accumulation)**
メモリが限られている場合に有効なバッチサイズを大きくする手法。

#### **動的学習率スケジューリング (Dynamic LR Scheduling)**
学習の停滞を検知して学習率を自動調整。

#### **早期停止 (Early Stopping)**
検証損失が改善しなくなったら学習を停止。

#### **メモリ効率的なデータローディング**
メモリ使用量を最適化して大きなデータセットを扱えるように。

#### **並列環境評価 (Parallel Evaluation)**
複数の環境で並列して評価を実行。

### 📊 推奨学習フロー

#### **初回学習時:**
```bash
# 1. カリキュラム学習で基礎を築く
python train_v430_advanced.py --mode curriculum

# 2. アンサンブル学習で多様性を持たせる
python train_v430_advanced.py --mode ensemble
```

#### **継続学習時:**
```bash
# マルチステージ学習で効率的に改善
python train_v430_advanced.py --mode multi_stage
```

#### **クイックテスト時:**
```bash
# 標準学習で高速検証
python train_v430_advanced.py --mode standard
```

### 🎯 各モードの推定学習時間

| モード | 推定時間 | 特徴 |
|--------|----------|------|
| curriculum | 2-3時間 | 段階的学習、安定性重視 |
| multi_stage | 1.5-2時間 | 目的別学習、効率性重視 |
| ensemble | 2-4時間 | 並列学習、多様性重視 |
| standard | 30-60分 | 高速学習、検証用 |

### 📁 出力ファイル構造

```
runs/sac_v430_{mode}_{timestamp}/
├── stage_warmup/models/final_model.zip          # カリキュラム学習時
├── stage_foundation/models/final_model.zip
├── stage_optimization/models/final_model.zip
├── stage_refinement/models/final_model.zip
├── final_model/models/final_model.zip
├── ensemble_model_0/models/final_model.zip      # アンサンブル学習時
├── ensemble_model_1/models/final_model.zip
├── ...
├── ensemble_config.json                         # アンサンブル設定
├── training_summary.json                        # 学習サマリー
└── logs/                                        # 詳細ログ
```

### 🔧 高度な設定

#### **メモリ最適化:**
```python
from sac_v430_training_optimizations import setup_efficient_training
setup_efficient_training()
```

#### **カスタム最適化設定:**
```python
from sac_v430_training_optimizations import create_optimized_config
config = create_optimized_config(base_config)
```

### 📈 性能監視

各学習モードで以下のメトリクスを監視:
- **学習曲線**: 報酬の推移
- **メモリ使用量**: GPU/CPUメモリの効率性
- **学習速度**: ステップ/秒
- **収束性**: 損失関数の減少率

### 🚨 注意事項

1. **GPUメモリ**: 大規模モデルでは16GB以上のGPUを推奨
2. **学習時間**: カリキュラム学習は時間がかかるが、最良の結果が期待できる
3. **並列処理**: アンサンブル学習時はCPUコア数が重要
4. **チェックポイント**: 長時間学習時は定期的にチェックポイントを保存

### 🎉 推奨開始方法

```bash
# 初回はカリキュラム学習から
python train_v430_advanced.py --mode curriculum --config configs/v430/sac_v430_optimized.json
```

この学習方法により、SAC v430はより安定して高性能な取引戦略を学習できます！
