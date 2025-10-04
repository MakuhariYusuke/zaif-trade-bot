"""Time series lag correlation analysis"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ztb.utils.errors import safe_operation


def compute_lag_correlations(frames: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Compute lag correlations for feature pairs.

    Lags: [1, 5, 10, 20]
    Returns top 10 pairs by absolute correlation.
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _compute_lag_correlations_impl(frames),
        context="lag_correlation_analysis",
        default_result=[],  # Return empty list on error
    )


def _compute_lag_correlations_impl(frames: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Implementation of lag correlation computation."""
    if not frames:
        return []

    # Combine frames
    combined = pd.concat(frames.values(), axis=1, keys=frames.keys())
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = combined.columns.droplevel(0)

    # Filter valid columns
    valid_cols = []
    for col in combined.columns:
        series = combined[col]
        if series.isna().mean() > 0.3 or series.nunique() <= 1:
            continue
        valid_cols.append(col)

    if len(valid_cols) < 2:
        return []

    filtered_df = combined[valid_cols].dropna()

    lags = [1, 5, 10, 20]
    results = []

    for i, col1 in enumerate(valid_cols):
        for j, col2 in enumerate(valid_cols):
            if i >= j:  # avoid duplicates and self
                continue
            series1 = filtered_df[col1]
            series2 = filtered_df[col2]
            for lag in lags:
                shifted = series1.shift(lag)
                valid = pd.concat([shifted, series2], axis=1).dropna()
                if len(valid) > 10:
                    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                    if not np.isnan(corr):
                        results.append(
                            {
                                "feature1": col1,
                                "feature2": col2,
                                "lag": lag,
                                "correlation": corr,
                            }
                        )

    # Sort by absolute correlation, take top 10
    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)  # type: ignore
    return results[:10]
