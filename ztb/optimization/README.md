# ハイパーパラメータ最適化フレームワーク

このディレクトリには、強化学習アルゴリズム（特にSAC）のハイパーパラメータを効率的に探索するための様々な最適化手法が含まれています。

## 📁 ディレクトリ構造

```
ztb/optimization/
├── __init__.py                 # パッケージ初期化
├── base.py                     # 基底クラスとデータ構造
├── sac_utils.py               # SAC専用ユーティリティ
├── compare_methods.py         # 手法比較スクリプト
├── methods/                   # 最適化手法の実装
│   ├── grid_search.py         # Grid Search
│   ├── random_search.py       # Random Search
│   ├── bayesian_optimization.py  # Bayesian Optimization
│   └── binary_search.py       # Binary Search
├── configs/                   # 最適化設定ファイル
└── results/                   # 結果保存先
```

## 🎯 利用可能な最適化手法

### 1. Grid Search（グリッドサーチ）
**特徴**: 全組み合わせを網羅的に探索  
**利点**: 最適解を見逃さない、体系的  
**欠点**: 組み合わせ爆発、計算コスト大  
**推奨**: パラメータ数が少ない（2-3個）、確実に最適解を見つけたい場合

```python
from ztb.optimization.methods.grid_search import GridSearchOptimizer
from ztb.optimization.sac_utils import get_sac_parameter_spaces

param_spaces = list(get_sac_parameter_spaces('essential').values())
optimizer = GridSearchOptimizer(
    parameter_spaces=param_spaces,
    objective_function=my_objective,
    grid_resolution={
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [64, 128, 256]
    }
)
result = optimizer.optimize()
```

### 2. Random Search（ランダムサーチ）
**特徴**: ランダムサンプリングで探索  
**利点**: Grid Searchより効率的、高次元に強い  
**欠点**: 運に左右される  
**推奨**: パラメータ数が多い（3個以上）、計算リソースが限られている場合

```python
from ztb.optimization.methods.random_search import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    parameter_spaces=param_spaces,
    objective_function=my_objective,
    n_trials=30
)
result = optimizer.optimize()
```

### 3. Bayesian Optimization（ベイズ最適化）
**特徴**: ガウス過程で効率的に探索  
**利点**: 少ない試行回数で良い結果、過去の結果を活用  
**欠点**: 高次元では性能低下、要scikit-optimize  
**推奨**: 評価コストが高い、中次元（~10次元）の探索

```python
from ztb.optimization.methods.bayesian_optimization import BayesianOptimizer

optimizer = BayesianOptimizer(
    parameter_spaces=param_spaces,
    objective_function=my_objective,
    n_trials=30,
    n_initial_points=10,
    acquisition_function='EI'
)
result = optimizer.optimize()
```

### 4. Binary Search（二分探索）
**特徴**: 黄金分割探索で単一パラメータを最適化  
**利点**: 非常に効率的（O(log n)）  
**欠点**: 単一パラメータのみ、単峰性の仮定  
**推奨**: Learning Rateなど単一パラメータの微調整

```python
from ztb.optimization.methods.binary_search import BinarySearchOptimizer
from ztb.optimization.base import ParameterSpace, ParameterType

param_space = ParameterSpace(
    'learning_rate', 
    ParameterType.LOG_UNIFORM, 
    low=1e-5, 
    high=1e-2
)

optimizer = BinarySearchOptimizer(
    parameter_space=param_space,
    objective_function=my_objective,
    tolerance=1e-5,
    max_iterations=20
)
result = optimizer.optimize()
```

## 🚀 クイックスタート

### 1. モック実験（動作確認）

```bash
# 手法比較実験（テスト用モック目的関数）
python -m ztb.optimization.compare_methods --preset essential --n-trials 20
```

### 2. 実際のSAC訓練での最適化

```python
from pathlib import Path
from ztb.optimization.methods.random_search import RandomSearchOptimizer
from ztb.optimization.sac_utils import (
    get_sac_parameter_spaces,
    create_sac_objective_function
)

# パラメータ空間を定義
param_spaces = list(get_sac_parameter_spaces('essential').values())

# 目的関数を作成（実際にSACを訓練）
objective_func = create_sac_objective_function(
    base_config_path=Path('configs/sac_v395i_complete_fix.json'),
    total_timesteps=5000,
    metric='critic_loss',
    lower_is_better=True
)

# 最適化を実行
optimizer = RandomSearchOptimizer(
    parameter_spaces=param_spaces,
    objective_function=objective_func,
    n_trials=10
)

result = optimizer.optimize()
result.save(Path('ztb/optimization/results/my_optimization.json'))
```

## 📊 SAC用パラメータプリセット

### `essential` - 最重要パラメータ（推奨）
- `learning_rate`: 学習率
- `batch_size`: バッチサイズ
- `gamma`: 割引率
- `tau`: Soft update係数

### `learning` - 学習率関連
- `learning_rate`
- `learning_starts`
- `train_freq`
- `gradient_steps`

### `buffer` - バッファ関連
- `buffer_size`
- `batch_size`
- `learning_starts`

### `full` - 全パラメータ（計算コスト大）
上記全て + `target_entropy`, `target_update_interval`等

## 📈 結果の分析

最適化結果は以下の形式で保存されます：

```json
{
  "optimizer_name": "RandomSearchOptimizer",
  "best_parameters": {
    "learning_rate": 0.0003,
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 0.005
  },
  "best_objective_value": 0.0805,
  "best_metrics": {
    "critic_loss": 0.0805,
    "actor_loss": -4.26,
    "ent_coef": 0.45
  },
  "n_trials": 20,
  "success_rate": 0.95,
  "total_duration_seconds": 1234.5,
  "all_trials": [...]
}
```

結果を読み込んで分析：

```python
from ztb.optimization.base import OptimizationResult

result = OptimizationResult.load('ztb/optimization/results/my_optimization.json')
result.print_summary()
```

## 🔬 手法の選び方

| 状況 | 推奨手法 | 理由 |
|------|---------|------|
| パラメータ2-3個、少数の選択肢 | Grid Search | 全組み合わせを網羅できる |
| パラメータ3個以上、連続値 | Random Search | 効率的、実装簡単 |
| 評価コストが高い（長時間訓練） | Bayesian Optimization | 少ない試行で最適化 |
| 単一パラメータの微調整 | Binary Search | 最も効率的 |
| 不明、とりあえず試したい | Random Search | バランスが良い |

## 📝 依存関係

- **必須**: numpy
- **オプション**: 
  - `scikit-optimize` - Bayesian Optimizationに必要
  - `matplotlib` - 結果の可視化に使用可能

```bash
# オプションの依存関係をインストール
pip install scikit-optimize matplotlib
```

## 💡 ベストプラクティス

1. **まずはRandom Searchから**: 手軽で効果的
2. **essential プリセットを使用**: 重要なパラメータに絞る
3. **段階的に探索**: 
   - Phase 1: Random Search（20試行）で大まかな範囲を把握
   - Phase 2: Binary Searchで微調整
   - Phase 3: Bayesian Optimizationで仕上げ
4. **結果を保存**: 必ず結果をJSON保存して後で分析
5. **複数シードで検証**: 最良パラメータを異なるシードで再検証

## 🎓 参考文献

- Bergstra & Bengio (2012): "Random Search for Hyper-Parameter Optimization"
- Snoek et al. (2012): "Practical Bayesian Optimization of Machine Learning Algorithms"
- Li et al. (2017): "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"

## 🆘 トラブルシューティング

### Q: Bayesian Optimizationが使えない
A: `pip install scikit-optimize` でインストール

### Q: 最適化が遅い
A: `n_trials`を減らす、またはより効率的な手法（Binary Search, Bayesian Optimization）を使用

### Q: 結果が不安定
A: 複数シードで実験、成功率をチェック

### Q: カスタム目的関数を作りたい
A: `sac_utils.py`の`create_sac_objective_function`を参考に実装
