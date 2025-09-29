"""
Common rolling operations and missing value handling.
共通ローリング処理と欠損値処理
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


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


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for memory efficiency.
    Note: Converts float64 columns to float32, which may result in loss of precision.
    精度が必要な場合はfloat32への変換による精度低下に注意してください。
    """
    df_optimized = df.copy()

    # Convert int64 to int32 with overflow check
    int_cols = df_optimized.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        min_val = df_optimized[col].min()
        max_val = df_optimized[col].max()
        if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            df_optimized[col] = df_optimized[col].astype("int32")
        else:
            # Leave as int64 and optionally log a warning
            print(
                f"Warning: Column '{col}' has values outside int32 range and will remain as int64."
            )

    # Convert int64 to int32
    int_cols = df_optimized.select_dtypes(include=["int64"]).columns
    df_optimized[int_cols] = df_optimized[int_cols].astype("int32")

    return df_optimized


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
            "avg_computation_time_ms": sum(feature_times.values())
            * 1000
            / len(feature_times)
            if feature_times
            else 0,
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
