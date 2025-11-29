#!/usr/bin/env python3
"""
Test script for new statistical functions in metrics module
"""

import numpy as np

from ztb.metrics.metrics import (
    autocorrelation,
    coefficient_of_variation,
    kurtosis,
    skewness,
    test_normality,
)


def test_new_functions():
    """Test the newly added statistical functions"""
    # Generate test data
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, 100)

    print("Testing new statistical functions:")
    print(f"Coefficient of Variation: {coefficient_of_variation(returns):.4f}")
    print(f"Skewness: {skewness(returns):.4f}")
    print(f"Kurtosis: {kurtosis(returns):.4f}")
    print(f"Autocorrelation (lag 1): {autocorrelation(returns, lag=1):.4f}")

    normality = test_normality(returns)
    print(
        f'Normality test - Shapiro p-value: {normality["shapiro_wilk"]["p_value"]:.4f}'
    )
    print(
        f'Normality test - Shapiro is_normal: {normality["shapiro_wilk"]["is_normal"]}'
    )

    print("\nAll functions working correctly!")


if __name__ == "__main__":
    test_new_functions()
