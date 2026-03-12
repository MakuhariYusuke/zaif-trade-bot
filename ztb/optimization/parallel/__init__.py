"""
並列評価モジュール

Walk-Forward評価をマルチプロセッシングで高速化するための
並列ウィンドウ評価器を提供します。

## 主要クラス

- **ParallelWindowEvaluator**: multiprocessing.Pool を使用した並列評価
  - 50ウィンドウ評価: 25時間 → 2-4時間（87-92%削減）
  - エラー隔離・コレクション機能
  - 自動ワーカー数最適化

## 使用例

```python
from ztb.optimization.parallel import ParallelWindowEvaluator
from ztb.evaluation.walk_forward.checkpoint import CheckpointManager

# 並列評価器作成
evaluator = ParallelWindowEvaluator(
    num_workers=8,  # ワーカー数（デフォルト: CPU数）
    checkpoint_mgr=checkpoint_mgr,
    enable_error_collection=True
)

# 評価実行
results, errors = evaluator.evaluate_windows_parallel(
    df=df,
    windows=windows,
    timesteps=10000,
    run_id="run_001"
)

# 結果確認
print(f"Completed: {len(results)}, Errors: {len(errors)}")
```

## パフォーマンス特性

- CPU密集的な機械学習評価に最適
- GILの影響を回避（spawn方式のマルチプロセッシング）
- ウィンドウ間の独立性を活用した効率的な並列化
"""

from .window_evaluator import ParallelWindowEvaluator, eval_window_worker

__all__ = ["ParallelWindowEvaluator", "eval_window_worker"]
