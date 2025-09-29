#!/usr/bin/env python3
"""
Leakage detector for time series data.

Detects data leakage using lagged cross-correlations and permutation tests.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class LeakageDetector:
    """Detects data leakage in time series using lagged correlations."""

    def __init__(self, threshold: float = 0.2, max_lag: int = 5, n_permutations: int = 200, random_seed: int = 42):
        """
        Initialize leakage detector.

        Args:
            threshold: Correlation threshold for flagging leakage
            max_lag: Maximum lag to test (negative lags only)
            n_permutations: Number of permutations for significance test
            random_seed: Random seed for reproducibility
        """
        self.threshold = threshold
        self.max_lag = max_lag
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(random_seed)

    def detect_leakage(self, target: pd.Series, features: pd.DataFrame) -> List[dict]:
        """
        Detect leakage between target and features using lagged correlations.

        Args:
            target: Target time series
            features: Feature DataFrame

        Returns:
            List of leakage detections with lag, correlation, p-value, and significance
        """
        detections = []

        for feature_name in features.columns:
            feature = features[feature_name]

            # Skip if insufficient data
            if len(target) < 10 or len(feature) < 10:
                continue

            # Test negative lags (feature predicts future target)
            for lag in range(1, self.max_lag + 1):
                try:
                    # Align series with lag
                    lagged_feature = feature.shift(lag)
                    aligned_target = target.iloc[lag:]

                    # Remove NaN values
                    valid_idx = ~(lagged_feature.isna() | aligned_target.isna())
                    if valid_idx.sum() < 10:
                        continue

                    lagged_feature_clean = lagged_feature[valid_idx]
                    aligned_target_clean = aligned_target[valid_idx]

                    # Calculate correlation
                    corr, _ = stats.pearsonr(lagged_feature_clean, aligned_target_clean)

                    # Permutation test for significance
                    p_value = self._permutation_test(lagged_feature_clean.values,
                                                    aligned_target_clean.values,
                                                    abs(corr))

                    # Flag if correlation exceeds threshold and is significant
                    is_significant = p_value < 0.05
                    exceeds_threshold = abs(corr) > self.threshold

                    if exceeds_threshold and is_significant:
                        detections.append({
                            'feature': feature_name,
                            'lag': -lag,  # Negative lag indicates feature precedes target
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': True,
                            'description': f'Potential leakage: {feature_name} at lag {-lag} (ρ={corr:.3f}, p={p_value:.3f})'
                        })

                except Exception as e:
                    # Skip problematic calculations
                    continue

        return detections

    def _permutation_test(self, x: np.ndarray, y: np.ndarray, observed_corr: float) -> float:
        """
        Perform permutation test for correlation significance.

        Args:
            x, y: Arrays to test
            observed_corr: Observed correlation coefficient

        Returns:
            p-value
        """
        n = len(x)
        observed_corr_abs = abs(observed_corr)

        # Count permutations where absolute correlation exceeds observed
        count = 0
        for _ in range(self.n_permutations):
            # Permute y
            y_permuted = self.rng.permutation(y)
            perm_corr, _ = stats.pearsonr(x, y_permuted)
            if abs(perm_corr) >= observed_corr_abs:
                count += 1

        return count / self.n_permutations

    def test_synthetic_leakage(self) -> bool:
        """
        Test with synthetic data containing known leakage.

        Returns:
            True if leakage is detected (test passes), False otherwise
        """
        # Create synthetic data with known leakage
        np.random.seed(42)
        n = 1000

        # Feature that predicts future target
        noise = np.random.normal(0, 1, n)
        feature = pd.Series(noise + np.random.normal(0, 0.1, n))

        # Target with leakage from feature (lag -1)
        target = 0.5 * feature.shift(-1) + noise + np.random.normal(0, 0.1, n)
        target = target.fillna(method='bfill')  # Fill NaN at end

        # Create DataFrame
        df = pd.DataFrame({
            'target': target,
            'feature': feature
        })

        # Detect leakage
        detections = self.detect_leakage(df['target'], df[['feature']])

        # Should detect leakage
        return len(detections) > 0

    def test_no_leakage(self) -> bool:
        """
        Test with synthetic data containing no leakage.

        Returns:
            True if no leakage is detected (test passes), False otherwise
        """
        # Create synthetic data with no leakage
        np.random.seed(123)
        n = 1000

        # Independent series
        target = np.random.normal(0, 1, n)
        feature = np.random.normal(0, 1, n)

        # Create DataFrame
        df = pd.DataFrame({
            'target': target,
            'feature': feature
        })

        # Detect leakage
        detections = self.detect_leakage(df['target'], df[['feature']])

        # Should not detect leakage
        return len(detections) == 0


def run_leakage_tests() -> dict:
    """
    Run synthetic leakage tests.

    Returns:
        Test results
    """
    detector = LeakageDetector()

    synthetic_leakage_detected = detector.test_synthetic_leakage()
    no_leakage_detected = detector.test_no_leakage()

    results = {
        'synthetic_leakage_test': 'PASS' if synthetic_leakage_detected else 'FAIL',
        'no_leakage_test': 'PASS' if not no_leakage_detected else 'FAIL',
        'overall': 'PASS' if synthetic_leakage_detected and not no_leakage_detected else 'FAIL'
    }

    return results


if __name__ == '__main__':
    # Run tests
    results = run_leakage_tests()
    print("Leakage Detector Tests:")
    for test, result in results.items():
        print(f"  {test}: {result}")

    if results['overall'] != 'PASS':
        exit(1)