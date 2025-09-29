"""Correlation analysis utilities"""

from typing import Dict, Optional

import pandas as pd


def compute_correlations(
    frames: Dict[str, pd.DataFrame],
    nan_strategy: str = "drop",  # "drop", "fill", or "none"
    fill_value: float = 0.0,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Compute correlations for successful feature frames.

    Filters out high NaN columns (>30%) and constant columns.
    Returns pearson and spearman correlation matrices.
    nan_strategy: "drop" to drop rows with NaN, "fill" to fill NaN with fill_value, "none" to leave NaN as is.
    fill_value: value to use when nan_strategy is "fill".
    Returns:
        Dict with 'pearson' and 'spearman' DataFrames or None if not computable.
    """
    if not frames:
        return {"pearson": None, "spearman": None}

    # Combine all frames into one DataFrame
    # Align indexes and ensure unique column names before concatenation
    aligned_frames = {}
    for key, df in frames.items():
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
        df.columns = [
            f"{key}_{col}" for col in df.columns
        ]  # prefix to ensure uniqueness
        aligned_frames[key] = df

    # Reindex all DataFrames to the union of all indexes
    all_indexes = sorted(set().union(*(df.index for df in aligned_frames.values())))
    for key in aligned_frames:
        aligned_frames[key] = aligned_frames[key].reindex(all_indexes)

    combined = pd.concat(aligned_frames.values(), axis=1)

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

    filtered_df = combined[valid_cols]
    if nan_strategy == "drop":
        filtered_df = filtered_df.dropna()
    elif nan_strategy == "fill":
        filtered_df = filtered_df.fillna(fill_value)
    # if nan_strategy == "none", do nothing

    if filtered_df.empty or len(filtered_df.columns) < 2:
        return {"pearson": None, "spearman": None}

    pearson = filtered_df.corr(method="pearson")
    spearman = filtered_df.corr(method="spearman")
    return {"pearson": pearson, "spearman": spearman}
