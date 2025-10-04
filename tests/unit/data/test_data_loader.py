import numpy as np
import pandas as pd

from ztb.data.data_loader import detect_outliers_iqr, detect_outliers_zscore


class TestDetectOutliersIQR:
    """Test IQR-based outlier detection."""

    def test_detect_outliers_iqr_normal_data(self):
        """Test outlier detection with normal data."""
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        outliers, _, _ = detect_outliers_iqr(data, "value")

        assert len(outliers) == 0

    def test_detect_outliers_iqr_with_outliers(self):
        """Test outlier detection with clear outliers."""
        data = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}  # 100 is outlier
        )
        outliers, _, _ = detect_outliers_iqr(data, "value")

        assert len(outliers) == 1
        assert outliers.iloc[0]["value"] == 100

    def test_detect_outliers_iqr_empty_data(self):
        """Test outlier detection with empty data."""
        data = pd.DataFrame({"value": []})
        outliers, lower, upper = detect_outliers_iqr(data, "value")

        assert len(outliers) == 0
        assert pd.isna(lower)
        assert pd.isna(upper)


class TestDetectOutliersZscore:
    """Test Z-score-based outlier detection."""

    def test_detect_outliers_zscore_normal_data(self):
        """Test z-score outlier detection with normal data."""
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        outliers = detect_outliers_zscore(data, "value", threshold=3)

        assert len(outliers) == 0

    def test_detect_outliers_zscore_with_outliers(self):
        """Test z-score outlier detection with outliers."""
        data = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}  # 100 is outlier
        )
        outliers = detect_outliers_zscore(data, "value", threshold=2)

        assert len(outliers) == 1
        assert outliers.iloc[0]["value"] == 100

    def test_detect_outliers_zscore_custom_threshold(self):
        """Test z-score with custom threshold."""
        data = pd.DataFrame(
            {
                "value": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    15,
                ]  # 15 might be outlier with lower threshold
            }
        )
        outliers = detect_outliers_zscore(data, "value", threshold=1.5)

        assert len(outliers) >= 1  # Should detect 15 as outlier

    def test_detect_outliers_zscore_with_nans(self):
        """Test z-score outlier detection with NaN values."""
        data = pd.DataFrame({"value": [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 100]})
        outliers = detect_outliers_zscore(data, "value", threshold=2)

        # Should still detect 100 as outlier despite NaN
        assert len(outliers) == 1
        assert outliers.iloc[0]["value"] == 100
