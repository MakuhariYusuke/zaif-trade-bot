"""Evaluation Module - Robust Model Performance Assessment

This package provides comprehensive tools for evaluating reinforcement learning
trading models with protection against overfitting.

## Walk-Forward Analysis

Walk-Forward Analysis is a time-series safe evaluation methodology that prevents
look-ahead bias and detects overfitting by:

1. **Multiple Windows**: Testing across multiple time periods
2. **Expanding Training**: Each window uses more training data than the previous
3. **Out-of-Sample Testing**: Test sets are always future data (never seen before)
4. **Overfitting Detection**: Comparing validation (in-sample) vs test (out-of-sample) performance

## Components

### WalkForwardSplitter
Generates rolling time-series windows with configurable train/val/test splits.

```python
from ztb.evaluation import WalkForwardSplitter

splitter = WalkForwardSplitter(
    initial_train_pct=0.50,  # First window: 50% training
    val_pct=0.15,            # 15% validation
    test_pct=0.15,           # 15% testing
    step_pct=0.15            # Each window: +15% training
)
windows = splitter.split(df)  # Generates 3-5 windows
```

### UnifiedEvaluator (Walk-Forward)
Runs walk-forward evaluation through the unified entrypoint.

```python
from ztb.evaluation.unified_evaluation import EvaluationType, UnifiedEvaluator

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
```

### WalkForwardResult & WalkForwardReporter
Aggregates results and generates reports.

```python
from ztb.evaluation import WalkForwardResult, WalkForwardReporter
from ztb.evaluation.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator(config={"walk_forward_timesteps": 50000})
evaluation, performances, errors = evaluator.evaluate_walk_forward_details_from_df(
    df=df,
    windows=windows,
    model_name="walk_forward",
)

result = WalkForwardResult(windows, performances)
reporter = WalkForwardReporter(result)
reporter.report()  # Console output
reporter.save_results('results.json')  # Persistent storage
```

## Key Metrics

- **ROI**: Return on Investment (profitability)
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst-case loss
- **Overfitting Ratio**: |val_roi - test_roi| / |val_roi|
  - Low values (< 0.2) indicate robust models
  - High values indicate overfitting to validation data

## Example Workflow

```python
from ztb.evaluation import WalkForwardSplitter
from ztb.evaluation.unified_evaluation import EvaluationType, UnifiedEvaluator
import pandas as pd

# Load data
df = pd.read_csv('trading_data.csv')

# Create windows
splitter = WalkForwardSplitter()
windows = splitter.split(df)

# Train and evaluate
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

## Compatibility

Re-exports `unified_evaluation` module for backward compatibility with
older code that imports from `ztb.evaluation.unified_evaluation`.
"""

from . import unified_evaluation  # noqa: F401

# Walk-Forward evaluation modules（新しいサブパッケージから import）
from .walk_forward import (
    TimeSeriesWindow,
    WindowPerformance,
    WalkForwardModelEvaluator,
    WalkForwardReporter,
    WalkForwardResult,
    WalkForwardSplitter,
)
from .unified_evaluation import EvaluationType, UnifiedEvaluator

__all__ = [
    "unified_evaluation",
    # Walk-Forward types
    "TimeSeriesWindow",
    "WindowPerformance",
    # Walk-Forward components
    "WalkForwardSplitter",
    "WalkForwardModelEvaluator",
    "WalkForwardResult",
    "WalkForwardReporter",
    "UnifiedEvaluator",
    "EvaluationType",
]
