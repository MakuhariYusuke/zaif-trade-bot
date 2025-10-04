"""
Common rolling operations and missing value handling.
共通ローリング処理と欠損値処理
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ztb.utils.memory.dtypes import OptimizationReport, optimize_dtypes


def rolling_mean(
    series: pd.Series, window: int, min_periods: Optional[int] = None
) -> pd.Series:
    """Rolling mean with proper handling"""
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).mean()


def rolling_std(
    series: pd.Series, window: int, min_periods: Optional[int] = None
) -> pd.Series:
    """Rolling standard deviation with proper handling"""
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).std()


def fillna(series: pd.Series, value: Union[float, int] = 0) -> pd.Series:
    """Fill NaN values"""
    return series.fillna(value)


def ffill(series: pd.Series) -> pd.Series:
    """Forward fill NaN values"""
    return series.ffill()


def bfill(series: pd.Series) -> pd.Series:
    """Backward fill NaN values"""
    return series.bfill()


def safe_divide(
    numerator: pd.Series, denominator: pd.Series, default: float = 0.0
) -> pd.Series:
    """Safe division avoiding division by zero"""
    result = numerator / denominator.astype(float).replace(0, np.nan)
    return result.fillna(default)


def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    *,
    chunk_size: int | None = None,
    memory_report: bool = False,
    convert_objects_to_category: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with memory-efficient dtypes applied.

    The function leverages :func:`ztb.utils.memory.dtypes.optimize_dtypes` to perform
    safe downcasting of numeric columns, convert low-cardinality object columns to
    categorical dtype, and optionally report the memory savings. It attaches the
    resulting :class:`OptimizationReport` to ``df.attrs['memory_optimization']``
    for downstream diagnostics.
    """

    effective_chunk = (
        chunk_size
        if chunk_size is not None
        else min(max(1, len(df.columns) // 8 or 1), 64)
    )

    optimized, report = optimize_dtypes(
        df,
        chunk_size=effective_chunk,
        convert_objects_to_category=convert_objects_to_category,
        memory_report=memory_report,
    )

    if isinstance(report, OptimizationReport):
        optimized.attrs.setdefault("memory_optimization", {})
        optimized.attrs["memory_optimization"].update(
            {
                "before_bytes": report.memory_before_bytes,
                "after_bytes": report.memory_after_bytes,
                "saved_bytes": report.memory_saved_bytes,
                "percent_reduction": report.percent_reduction,
            }
        )

    return optimized


def generate_intermediate_report(
    step: int,
    feature_times: Dict[str, float],
    memory_usage: float,
    nan_rates: Dict[str, float],
) -> None:
    """Generate intermediate report for feature computation progress"""
    import json
    from pathlib import Path

    report = {
        "step": step,
        "timestamp": pd.Timestamp.now().isoformat(),
        "memory_usage_mb": memory_usage,
        "feature_performance": {
            feature: {
                "computation_time_ms": feature_time * 1000,
                "nan_rate": nan_rates.get(feature, 0.0),
            }
            for feature, feature_time in feature_times.items()
        },
        "summary": {
            "total_features": len(feature_times),
            "avg_computation_time_ms": (
                sum(feature_times.values()) * 1000 / len(feature_times)
                if feature_times
                else 0
            ),
            "max_nan_rate": max(nan_rates.values()) if nan_rates else 0.0,
        },
    }

    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report_path = reports_dir / f"intermediate_features_step_{step:05d}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
