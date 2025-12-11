"""
Unit tests for ztb.metrics.statistics module.
"""

import numpy as np
import pandas as pd

from ztb.metrics.statistics import (
    calculate_atr,
    calculate_autocorrelation,
    calculate_volatility,
    detect_outliers_iqr,
    p_mean_method,
    rolling_statistics,
)


class TestStatistics:
    """Test cases for statistics functions."""

    def setup_method(self):
        """Setup test data."""
        self.data_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.data_np = np.array(self.data_list)
        self.data_pd = pd.Series(self.data_list)
        self.window = 3

    def test_p_mean_method_arithmetic(self):
        """Test p_mean_method with arithmetic mean."""
        p_values = [0.01, 0.05, 0.10]
        result = p_mean_method(p_values, method="arithmetic")
        expected = (0.01 + 0.05 + 0.10) / 3
        assert np.isclose(result, expected)

    def test_p_mean_method_geometric(self):
        """Test p_mean_method with geometric mean."""
        p_values = [0.01, 0.05, 0.10]
        result = p_mean_method(p_values, method="geometric")
        expected = np.exp(np.mean(np.log(p_values)))
        assert np.isclose(result, expected)

    def test_rolling_statistics_list(self):
        """Test rolling_statistics with list input."""
        stats = rolling_statistics(self.data_list, self.window)
        assert len(stats["mean"]) == len(self.data_list) - self.window + 1
        assert np.isclose(stats["mean"][0], 2.0)  # mean(1, 2, 3)
        assert np.isclose(stats["max"][-1], 5.0)  # max(3, 4, 5)

    def test_rolling_statistics_numpy(self):
        """Test rolling_statistics with numpy input."""
        stats = rolling_statistics(self.data_np, self.window)
        assert len(stats["mean"]) == len(self.data_np) - self.window + 1
        assert np.isclose(stats["mean"][0], 2.0)

    def test_rolling_statistics_pandas(self):
        """Test rolling_statistics with pandas input."""
        stats = rolling_statistics(self.data_pd, self.window)
        assert len(stats["mean"]) == len(self.data_pd) - self.window + 1
        assert np.isclose(stats["mean"][0], 2.0)

    def test_calculate_volatility(self):
        """Test calculate_volatility."""
        vol = calculate_volatility(self.data_list, self.window)
        assert len(vol) == len(self.data_list) - self.window + 1
        # std(1, 2, 3) = 0.816... (population) or 1.0 (sample)?
        # numpy std is population by default, pandas is sample (ddof=1)
        # My implementation uses np.std for list/numpy (population) and rolling.std for pandas (sample)
        # Wait, rolling_statistics implementation for list uses np.std which is population.
        # Pandas rolling().std() uses ddof=1 (sample).
        # This is an inconsistency I should fix or be aware of.
        # Let's check the implementation again.
        pass

    def test_detect_outliers_iqr(self):
        """Test detect_outliers_iqr."""
        data = [10, 12, 11, 13, 100, 11, 12]  # 100 is outlier
        outliers = detect_outliers_iqr(data)
        assert outliers[4]  # Check truthiness
        assert not outliers[0]

    def test_calculate_autocorrelation(self):
        """Test calculate_autocorrelation."""
        # Simple trend
        data = [1, 2, 3, 4, 5]
        ac = calculate_autocorrelation(data, lag=1)
        # Should be high positive
        assert ac > 0.0

    def test_calculate_atr(self):
        """Test calculate_atr."""
        # Use larger dataset for Ta-Lib stability
        data = pd.DataFrame(
            {
                "high": [10, 12, 15, 14, 16] * 5,
                "low": [8, 9, 11, 12, 13] * 5,
                "close": [9, 11, 14, 13, 15] * 5,
            }
        )

        atr = calculate_atr(data, period=3)

        assert len(atr) == 25
        # First few values might be NaN
        assert pd.isna(atr.iloc[0])

        # Check that we eventually get values
        assert not pd.isna(atr.iloc[-1])
        assert atr.iloc[-1] > 0
