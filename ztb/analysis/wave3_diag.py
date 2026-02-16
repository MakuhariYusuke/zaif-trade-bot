#!/usr/bin/env python3
"""
Wave 3 Diagnostic Functions for Analysis.

This module provides diagnostic functions for analyzing correlations,
mutual information, variance inflation factors, and data leakage detection.
Used in advanced market analysis and feature engineering.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_correlations(
    data: pd.DataFrame, method: str = "pearson"
) -> Dict[str, pd.DataFrame]:
    """
    Calculate correlation matrix for the given data.

    Args:
        data: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Dictionary containing correlation matrices for different methods
    """
    if data.empty:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame()}

    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame()}

    results = {}

    # Calculate Pearson correlation
    try:
        results["pearson"] = numeric_data.corr(method="pearson")
    except Exception:
        results["pearson"] = pd.DataFrame()

    # Calculate Spearman correlation
    try:
        results["spearman"] = numeric_data.corr(method="spearman")
    except Exception:
        results["spearman"] = pd.DataFrame()

    return results


def calculate_mutual_info(
    X: pd.DataFrame, horizons: List[int], random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Calculate mutual information between features and target at different horizons.

    Args:
        X: Feature DataFrame
        horizons: List of forecast horizons
        random_state: Random state for reproducibility

    Returns:
        Dictionary with mutual information results for each horizon
    """
    results = {}

    if X.empty or not horizons:
        return results

    numeric_data = X.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 2:
        return results

    # Assume last column is target
    feature_cols = numeric_data.columns[:-1]
    target_col = numeric_data.columns[-1]

    X_features = numeric_data[feature_cols].values
    y_target = numeric_data[target_col].values

    for horizon in horizons:
        try:
            # Create lagged target for this horizon
            if len(y_target) > horizon:
                y_lagged = y_target[horizon:]
                X_lagged = X_features[:-horizon]

                mi_scores = mutual_info_regression(
                    X_lagged, y_lagged, random_state=random_state
                )

                result_df = pd.DataFrame(
                    {"feature": feature_cols, "mutual_info": mi_scores}
                )

                results[f"h{horizon}"] = result_df
        except Exception:
            # Skip failed calculations
            continue

    return results


def calculate_vif(data: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for features.

    Args:
        data: Input DataFrame with numeric features
        threshold: VIF threshold for multicollinearity detection

    Returns:
        DataFrame with VIF values and multicollinearity flags
    """
    if data.empty:
        return pd.DataFrame()

    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 1:
        return pd.DataFrame()

    # If only a single numeric feature is present, VIF is 1.0 by definition
    if numeric_data.shape[1] == 1:
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_data.columns
        vif_data["vif"] = [1.0]
        vif_data["multicollinear"] = [False]
        return vif_data

    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_data.columns
    vif_data["vif"] = [
        variance_inflation_factor(numeric_data.values, i)
        for i in range(numeric_data.shape[1])
    ]

    vif_data["multicollinear"] = vif_data["vif"] > threshold

    return vif_data


def check_leaks(
    data: pd.DataFrame, target_col: str = "return", correlation_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Check for data leakage by analyzing correlations with target.

    Args:
        data: Input DataFrame
        target_col: Name of target column (default: "return")
        correlation_threshold: Correlation threshold for leakage detection

    Returns:
        DataFrame with correlation analysis results
    """
    results = []

    if data.empty:
        return pd.DataFrame()

    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 1:
        return pd.DataFrame()

    if target_col not in numeric_data.columns:
        # For testing purposes, analyze all features without target correlation
        for col in numeric_data.columns:
            results.append(
                {
                    "feature": col,
                    # return NaN when there is no target column to indicate undefined correlation
                    "corr_current": np.nan,
                    "corr_future": np.nan,
                    "potential_leak": False,
                }
            )
        return pd.DataFrame(results)

    target_series = numeric_data[target_col]

    for col in numeric_data.columns:
        if col == target_col:
            continue

        try:
            # Calculate correlation with current target
            corr_current = numeric_data[col].corr(target_series)

            # Calculate correlation with future target (shifted)
            corr_future = numeric_data[col].corr(target_series.shift(-1))

            results.append(
                {
                    "feature": col,
                    "corr_current": corr_current if not np.isnan(corr_current) else 0.0,
                    "corr_future": corr_future if not np.isnan(corr_future) else 0.0,
                    "potential_leak": abs(
                        corr_future if not np.isnan(corr_future) else 0.0
                    )
                    > correlation_threshold,
                }
            )
        except Exception:
            results.append(
                {
                    "feature": col,
                    "corr_current": 0.0,
                    "corr_future": 0.0,
                    "potential_leak": False,
                }
            )

    return pd.DataFrame(results)


def generate_synthetic_data(
    n_samples: int = 1000,
    n_rows: int | None = None,
    n_features: int = 10,
    n_informative: int = 5,
    noise: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic data for testing and validation.

    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        noise: Amount of noise to add
        random_state: Random state for reproducibility

    Returns:
        DataFrame with synthetic market data
    """
    np.random.seed(random_state)

    # Support legacy callers using 'n_rows' keyword
    rows = n_samples if n_rows is None else n_rows

    # Generate timestamps
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="1H")

    # Generate price data with trend and noise
    base_price = 100.0
    trend = np.linspace(0, 10, rows)  # Slight upward trend
    noise_component = np.random.normal(0, 2, rows)
    close_prices = base_price + trend + noise_component

    # Generate OHLC data
    highs = close_prices + np.abs(np.random.normal(0, 1, rows))
    lows = close_prices - np.abs(np.random.normal(0, 1, rows))
    opens = close_prices + np.random.normal(0, 0.5, rows)

    # Generate volume
    volume = np.random.lognormal(10, 1, rows)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "ts": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_prices,
            "volume": volume,
            "episode_id": np.repeat(range(rows // 100 + 1), 100)[:rows],
        }
    )

    return df
