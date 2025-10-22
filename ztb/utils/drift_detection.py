"""
Feature drift detection using PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) test.

This module detects distribution shifts between training and evaluation datasets,
which can indicate model degradation or data quality issues.

Thresholds:
- PSI > 0.2: Significant drift detected
- KS p-value < 0.01: Distributions are significantly different
"""

from pathlib import Path
from typing import Tuple, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from ztb.trading.environment.constants import EPSILON


class DriftResultDict(TypedDict):
    """Drift detection result dictionary."""

    feature_name: str
    psi: float
    psi_drift: bool
    ks_statistic: float
    ks_p_value: float
    ks_drift: bool
    drift_detected: bool
    train_mean: float
    eval_mean: float
    train_std: float
    eval_std: float


def calculate_psi(
    expected: NDArray[np.float32],
    actual: NDArray[np.float32],
    bins: int = 10,
    epsilon: float = EPSILON,
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    PSI measures the shift in distribution between two datasets.
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.2: Moderate change
    - PSI ≥ 0.2: Significant change (ACTION REQUIRED)

    Args:
        expected: Expected distribution (e.g., training data)
        actual: Actual distribution (e.g., evaluation data)
        bins: Number of bins for discretization
        epsilon: Small constant to avoid log(0)

    Returns:
        PSI value (higher = more drift)

    Examples:
        >>> expected = np.random.normal(0, 1, 1000)
        >>> actual = np.random.normal(0, 1, 1000)  # Same distribution
        >>> calculate_psi(expected, actual)
        0.05  # Low PSI (similar distributions)

        >>> actual_shifted = np.random.normal(1, 1, 1000)  # Shifted distribution
        >>> calculate_psi(expected, actual_shifted)
        0.35  # High PSI (significant drift)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    # Handle case where all values are identical
    if len(breakpoints) <= 2:
        return 0.0  # No variation, no drift

    # Count frequencies in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Add epsilon to avoid division by zero
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def calculate_ks(
    expected: NDArray[np.float32],
    actual: NDArray[np.float32],
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test statistic and p-value.

    KS test checks if two samples come from the same distribution.
    - p-value < 0.01: Distributions are significantly different (ACTION REQUIRED)
    - p-value ≥ 0.01: No significant difference

    Args:
        expected: Expected distribution (e.g., training data)
        actual: Actual distribution (e.g., evaluation data)

    Returns:
        Tuple of (statistic, p_value)
        - statistic: KS statistic (0 to 1, higher = more different)
        - p_value: Probability that distributions are the same

    Examples:
        >>> expected = np.random.normal(0, 1, 1000)
        >>> actual = np.random.normal(0, 1, 1000)  # Same distribution
        >>> stat, p = calculate_ks(expected, actual)
        >>> p > 0.01  # True (no significant difference)

        >>> actual_shifted = np.random.normal(1, 1, 1000)  # Shifted distribution
        >>> stat, p = calculate_ks(expected, actual_shifted)
        >>> p < 0.01  # True (significant difference)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return (np.nan, np.nan)

    # Run KS test
    statistic, p_value = stats.ks_2samp(expected, actual)

    return (float(statistic), float(p_value))


def detect_drift_single_feature(
    train_values: NDArray[np.float32],
    eval_values: NDArray[np.float32],
    feature_name: str,
    psi_threshold: float = 0.2,
    ks_p_threshold: float = 0.01,
) -> DriftResultDict:
    """
    Detect drift for a single feature.

    Args:
        train_values: Feature values from training data
        eval_values: Feature values from evaluation data
        feature_name: Name of the feature
        psi_threshold: PSI threshold for drift detection
        ks_p_threshold: KS p-value threshold for drift detection

    Returns:
        Dictionary with drift statistics:
        - feature_name: Name of the feature
        - psi: PSI value
        - psi_drift: Whether PSI indicates drift (bool)
        - ks_statistic: KS test statistic
        - ks_p_value: KS test p-value
        - ks_drift: Whether KS indicates drift (bool)
        - drift_detected: Overall drift flag (PSI OR KS)
        - train_mean: Mean of training data
        - eval_mean: Mean of evaluation data
        - train_std: Std of training data
        - eval_std: Std of evaluation data
    """
    # Calculate PSI
    psi = calculate_psi(train_values, eval_values)
    psi_drift = psi >= psi_threshold if not np.isnan(psi) else False

    # Calculate KS
    ks_stat, ks_p = calculate_ks(train_values, eval_values)
    ks_drift = ks_p < ks_p_threshold if not np.isnan(ks_p) else False

    # Overall drift flag
    drift_detected = psi_drift or ks_drift

    # Basic statistics
    train_mean = float(np.nanmean(train_values))
    eval_mean = float(np.nanmean(eval_values))
    train_std = float(np.nanstd(train_values))
    eval_std = float(np.nanstd(eval_values))

    return {
        "feature_name": feature_name,
        "psi": psi,
        "psi_drift": psi_drift,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "ks_drift": ks_drift,
        "drift_detected": drift_detected,
        "train_mean": train_mean,
        "eval_mean": eval_mean,
        "train_std": train_std,
        "eval_std": eval_std,
    }


def detect_drift_all_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_p_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Detect drift for all features in datasets.

    Args:
        train_df: Training dataset (features as columns)
        eval_df: Evaluation dataset (features as columns)
        psi_threshold: PSI threshold for drift detection
        ks_p_threshold: KS p-value threshold for drift detection

    Returns:
        DataFrame with drift statistics for each feature
    """
    results = []

    for feature_name in train_df.columns:
        if feature_name not in eval_df.columns:
            continue

        train_values = np.asarray(train_df[feature_name].values)
        eval_values = np.asarray(eval_df[feature_name].values)

        result = detect_drift_single_feature(
            train_values,
            eval_values,
            feature_name,
            psi_threshold,
            ks_p_threshold,
        )
        results.append(result)

    return pd.DataFrame(results)


def generate_drift_report_html(
    drift_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate HTML drift report.

    Args:
        drift_df: Drift statistics DataFrame (from detect_drift_all_features)
        output_path: Path to save HTML report
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Drift Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .drift { background-color: #ffcccc; font-weight: bold; }
            .no-drift { background-color: #ccffcc; }
            .summary { margin: 20px 0; padding: 10px; background-color: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>Feature Drift Report</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Features:</strong> {total_features}</p>
            <p><strong>Features with Drift:</strong> {drift_count} ({drift_pct:.1f}%)</p>
            <p><strong>PSI Threshold:</strong> 0.2 (Significant drift)</p>
            <p><strong>KS p-value Threshold:</strong> 0.01 (Significant difference)</p>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>PSI</th>
                    <th>PSI Drift</th>
                    <th>KS Statistic</th>
                    <th>KS p-value</th>
                    <th>KS Drift</th>
                    <th>Overall Drift</th>
                    <th>Train Mean</th>
                    <th>Eval Mean</th>
                    <th>Train Std</th>
                    <th>Eval Std</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add summary stats
    total_features = len(drift_df)
    drift_count = drift_df["drift_detected"].sum()
    drift_pct = (drift_count / total_features * 100) if total_features > 0 else 0

    html = html.format(
        total_features=total_features,
        drift_count=drift_count,
        drift_pct=drift_pct,
    )

    # Add rows
    for _, row in drift_df.iterrows():
        row_class = "drift" if row["drift_detected"] else "no-drift"
        html += f"""
                <tr class="{row_class}">
                    <td>{row['feature_name']}</td>
                    <td>{row['psi']:.4f}</td>
                    <td>{'✓' if row['psi_drift'] else ''}</td>
                    <td>{row['ks_statistic']:.4f}</td>
                    <td>{row['ks_p_value']:.4f}</td>
                    <td>{'✓' if row['ks_drift'] else ''}</td>
                    <td>{'<strong>DRIFT</strong>' if row['drift_detected'] else 'OK'}</td>
                    <td>{row['train_mean']:.4f}</td>
                    <td>{row['eval_mean']:.4f}</td>
                    <td>{row['train_std']:.4f}</td>
                    <td>{row['eval_std']:.4f}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")
