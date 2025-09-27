"""
Cross-validation utilities for ZTB evaluation.

This module provides time series cross-validation functionality
including rolling forward validation and multiple testing correction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from statsmodels.stats.multitest import fdrcorrection  # type: ignore
from typing import Dict, List, Any, Tuple
from ztb.metrics.metrics import calculate_all_metrics


def rolling_forward_cv(returns: pd.Series,
                      n_splits: int = 5,
                      test_size: int = 60,
                      gap: int = 0) -> List[Dict[str, Any]]:
    """
    Perform rolling forward cross-validation on time series data.

    Args:
        returns: Time series of returns
        n_splits: Number of CV splits
        test_size: Size of test set in periods
        gap: Gap between train and test sets

    Returns:
        List of CV results for each split
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have DatetimeIndex")

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(len(returns)).reshape(-1, 1))):
        train_returns = returns.iloc[train_idx]  # pd.Series
        test_returns = returns.iloc[test_idx]    # pd.Series

        # Calculate metrics for this fold
        train_metrics = calculate_all_metrics(train_returns)
        test_metrics = calculate_all_metrics(test_returns)

        cv_results.append({
            'fold': fold,
            'train_period': {
                'start': train_returns.index[0],
                'end': train_returns.index[-1],
                'length': len(train_returns)
            },
            'test_period': {
                'start': test_returns.index[0],
                'end': test_returns.index[-1],
                'length': len(test_returns)
            },
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'sharpe_ratio': test_metrics.get('sharpe_ratio', 0)
        })

    return cv_results


def aggregate_cv_results(cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate cross-validation results across all folds.

    Args:
        cv_results: List of CV results from rolling_forward_cv

    Returns:
        Aggregated statistics
    """
    if not cv_results:
        return {}

    sharpe_ratios = [r['sharpe_ratio'] for r in cv_results]

    return {
        'mean_sharpe': np.mean(sharpe_ratios),
        'std_sharpe': np.std(sharpe_ratios),
        'min_sharpe': np.min(sharpe_ratios),
        'max_sharpe': np.max(sharpe_ratios),
        'sharpe_ci_95': (
            np.mean(sharpe_ratios) - 1.96 * np.std(sharpe_ratios),
            np.mean(sharpe_ratios) + 1.96 * np.std(sharpe_ratios)
        ),
        'n_folds': len(cv_results),
        'fold_results': cv_results
    }


def apply_multiple_testing_correction(p_values: List[float],
                                    method: str = 'fdr_bh',
                                    alpha: float = 0.1) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Apply multiple testing correction to p-values.

    Args:
        p_values: List of p-values to correct
        method: Correction method ('fdr_bh' for Benjamini-Hochberg)
        alpha: Significance level

    Returns:
        Tuple of (rejected, corrected_p_values)
    """
    if method == 'fdr_bh':
        rejected, corrected_p = fdrcorrection(p_values, alpha=alpha)
    else:
        raise ValueError(f"Unsupported correction method: {method}")

    return rejected, corrected_p


def evaluate_with_cv(feature_df: pd.DataFrame,
                    returns: pd.Series,
                    n_splits: int = 5,
                    test_size: int = 60) -> Dict[str, Any]:
    """
    Evaluate features with cross-validation.

    Args:
        feature_df: DataFrame of features
        returns: Series of returns
        n_splits: Number of CV splits
        test_size: Test set size

    Returns:
        Evaluation results with CV metrics
    """
    # Align data
    aligned_data = pd.concat([feature_df, returns.rename('returns')], axis=1).dropna()

    if len(aligned_data) < test_size * 2:
        return {
            'status': 'insufficient_data_for_cv',
            'available_periods': len(aligned_data),
            'required_periods': test_size * 2
        }

    # Perform CV
    cv_results = rolling_forward_cv(aligned_data['returns'], n_splits, test_size)
    cv_summary = aggregate_cv_results(cv_results)

    # Calculate feature correlations with CV-adjusted significance
    correlations = {}
    p_values = []

    for col in feature_df.columns:
        if col in aligned_data.columns:
            corr = aligned_data[col].corr(aligned_data['returns'])
            # Simple p-value approximation (for demonstration)
            n = len(aligned_data)
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
            p_val = 2 * (1 - abs(t_stat) / np.sqrt(n - 2)) if n > 2 else 1.0
            p_values.append(p_val)
            correlations[col] = {
                'correlation': corr,
                'p_value': p_val
            }

    # Apply multiple testing correction
    if p_values:
        rejected, corrected_p = apply_multiple_testing_correction(p_values, alpha=0.1)
        for i, col in enumerate(feature_df.columns):
            if col in correlations:
                correlations[col]['corrected_p_value'] = corrected_p[i]
                correlations[col]['significant'] = rejected[i]

    return {
        'status': 'success',
        'cv_summary': cv_summary,
        'correlations': correlations,
        'total_features': len(feature_df.columns),
        'aligned_periods': len(aligned_data)
    }