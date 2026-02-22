"""
Walk-Forward Analysis モジュール

時系列安全な過学習検出と複数ウィンドウでのモデル評価を提供します。

## アーキテクチャ

```
WalkForwardSplitter
    ↓ (generates)
TimeSeriesWindow (List)
    ↓ (for each window)
WalkForwardModelEvaluator.train_and_evaluate_window()
    ↓ (generates)
WindowPerformance (List)
    ↓ (aggregates)
WalkForwardResult
    ↓ (reports)
WalkForwardReporter
```

## 使用例（UnifiedEvaluator 推奨）

```python
from ztb.evaluation.walk_forward import WalkForwardSplitter
from ztb.evaluation.unified_evaluation import EvaluationType, UnifiedEvaluator
import pandas as pd

# ウィンドウ分割
splitter = WalkForwardSplitter()
windows = splitter.split(df)

# 統合評価
config = {
    "walk_forward_windows": [w._asdict() for w in windows],
    "walk_forward_timesteps": 50000,
}
evaluator = UnifiedEvaluator(config=config)
evaluation = evaluator.evaluate_model(
    model_path="walk_forward",
    data_path="trading_data.csv",
    evaluation_type=EvaluationType.WALK_FORWARD,
)

print(f"Average Test ROI: {evaluation.performance_metrics['average_test_roi']:.2%}")
print(f"Overfitting Ratio: {evaluation.performance_metrics['overfitting_ratio']:.2%}")
```

## 主要クラス

- `TimeSeriesWindow`: ウィンドウ定義（train/val/test インデックス）
- `WindowPerformance`: ウィンドウ単位の性能メトリクス
- `WalkForwardSplitter`: ウィンドウ生成
- `WalkForwardModelEvaluator`: SAC 訓練・評価（旧API）
- `WalkForwardResult`: 集計結果
- `WalkForwardReporter`: レポート機能
"""

from .evaluator import WalkForwardModelEvaluator
from .reporter import WalkForwardReporter
from .types import WalkForwardResult
from .splitter import WalkForwardSplitter
from .types import TimeSeriesWindow, WindowPerformance

__all__ = [
    # Types
    "TimeSeriesWindow",
    "WindowPerformance",
    # Splitter
    "WalkForwardSplitter",
    # Evaluator
    "WalkForwardModelEvaluator",
    # Result
    "WalkForwardResult",
    # Reporter
    "WalkForwardReporter",
]
