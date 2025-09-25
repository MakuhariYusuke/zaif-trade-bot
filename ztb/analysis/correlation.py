"""Correlation analysis utilities"""
import pandas as pd
from typing import Dict, Any, Optional
import numpy as np

def compute_correlations(frames: Dict[str, pd.DataFrame]) -> Dict[str, Optional[pd.DataFrame]]:
    """Compute correlations for successful feature frames.

    Filters out high NaN columns (>30%) and constant columns.
    Returns pearson and spearman correlation matrices.
    """
    if not frames:
        return {"pearson": None, "spearman": None}

    # Combine all frames into one DataFrame
    combined = pd.concat(frames.values(), axis=1, keys=frames.keys())
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = combined.columns.droplevel(0)  # flatten if MultiIndex

    # Filter columns
    valid_cols = []
    for col in combined.columns:
        series = combined[col]
        nan_rate = series.isna().mean()
        if nan_rate > 0.3:
            continue  # skip high NaN
        if series.nunique() <= 1:
            continue  # skip constant
        valid_cols.append(col)

    if len(valid_cols) < 2:
        return {"pearson": None, "spearman": None}

    filtered_df = combined[valid_cols].dropna()  # drop rows with any NaN for correlation

    if filtered_df.empty or len(filtered_df.columns) < 2:
        return {"pearson": None, "spearman": None}

    pearson = filtered_df.corr(method='pearson')
    spearman = filtered_df.corr(method='spearman')
    return {"pearson": pearson, "spearman": spearman}
