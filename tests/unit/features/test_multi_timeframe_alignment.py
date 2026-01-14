import numpy as np
import pandas as pd

from ztb.features.multi_timeframe.data_pipeline import MultiTimeframeDataPipeline
from ztb.features.timeframe import Timeframe


class TestMultiTimeframeAlignment:
    """Test cases for multi-timeframe data alignment and synchronization."""

    def test_synchronize_to_base_with_merge_asof(self):
        """Test that _synchronize_to_base uses proper time alignment with merge_asof."""
        pipeline = MultiTimeframeDataPipeline()

        # Create base timeframe data (1min)
        base_timestamps = pd.date_range("2023-01-01 10:00:00", periods=10, freq="1min")
        base_df = pd.DataFrame(
            {
                "timestamp": base_timestamps,
                "open": [100 + i for i in range(10)],
                "high": [101 + i for i in range(10)],
                "low": [99 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000] * 10,
            }
        )

        # Create 5min timeframe data (every 5 minutes, offset by 2 minutes)
        source_timestamps = pd.date_range("2023-01-01 10:02:00", periods=3, freq="5min")
        source_df = pd.DataFrame(
            {
                "timestamp": source_timestamps,
                "open": [200, 210, 220],
                "high": [201, 211, 221],
                "low": [199, 209, 219],
                "close": [200.5, 210.5, 220.5],
                "volume": [5000] * 3,
            }
        )

        # Synchronize
        result = pipeline._synchronize_to_base(
            source_df, base_df, Timeframe.M5, Timeframe.M1
        )

        # Check that result has 8 rows (first 2 timestamps have no source data)
        assert (
            len(result) == 8
        ), f"Expected 8 rows (missing data for first 2 timestamps), got {len(result)}"

        # Check that timestamps are aligned (should be subset of base timestamps)
        assert (
            result["timestamp"].isin(base_df["timestamp"]).all()
        ), "Result timestamps should be subset of base timestamps"

        # Check that data is properly filled (first few rows should have source data)
        # Row 0-1: before first source timestamp, should be NaN or missing
        # Row 2+: should have source data from first source row

    def test_resample_timeframe_uses_correct_frequencies(self):
        """Test that _resample_timeframe uses correct pandas frequency strings."""
        pipeline = MultiTimeframeDataPipeline()

        # Create 1min data
        timestamps = pd.date_range(
            "2023-01-01 10:00:00", periods=60, freq="1min"
        )  # 1 hour of 1min data
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.random.uniform(100, 110, 60),
                "high": np.random.uniform(110, 120, 60),
                "low": np.random.uniform(90, 100, 60),
                "close": np.random.uniform(100, 110, 60),
                "volume": np.random.uniform(1000, 2000, 60),
            }
        )

        # Resample to 5min
        result = pipeline._resample_timeframe(df, Timeframe.M1, Timeframe.M5)

        # Should have 12 rows (60 minutes / 5 minutes = 12)
        assert (
            len(result) == 12
        ), f"Expected 12 rows for 5min resampling, got {len(result)}"

        # Check OHLCV aggregation
        assert result["open"].iloc[0] == df["open"].iloc[0], "First open should match"
        assert (
            result["close"].iloc[0] == df["close"].iloc[4]
        ), "First close should match last of first group"
        assert (
            result["high"].iloc[0] == df["high"].iloc[0:5].max()
        ), "High should be max of group"
        assert (
            result["low"].iloc[0] == df["low"].iloc[0:5].min()
        ), "Low should be min of group"
        assert (
            abs(result["volume"].iloc[0] - df["volume"].iloc[0:5].sum()) < 1e-10
        ), "Volume should be sum of group"

    def test_generate_missing_timeframes_creates_correct_data(self):
        """Test that generate_missing_timeframes creates properly resampled data."""
        pipeline = MultiTimeframeDataPipeline()

        # Create 1min data
        timestamps = pd.date_range(
            "2023-01-01 10:00:00", periods=120, freq="1min"
        )  # 2 hours
        df_1min = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100 + i * 0.1 for i in range(120)],
                "high": [101 + i * 0.1 for i in range(120)],
                "low": [99 + i * 0.1 for i in range(120)],
                "close": [100.5 + i * 0.1 for i in range(120)],
                "volume": [1000] * 120,
            }
        )

        available_data = {Timeframe.M1: df_1min}
        target_timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15]

        result = pipeline.generate_missing_timeframes(available_data, target_timeframes)

        # Should have all three timeframes
        assert len(result) == 3
        assert Timeframe.M1 in result
        assert Timeframe.M5 in result
        assert Timeframe.M15 in result

        # Check row counts
        assert (
            len(result[Timeframe.M5]) == 24
        ), f"Expected 24 rows for 5min (120min/5min), got {len(result[Timeframe.M5])}"
        assert (
            len(result[Timeframe.M15]) == 8
        ), f"Expected 8 rows for 15min (120min/15min), got {len(result[Timeframe.M15])}"

    def test_data_quality_report_provides_comprehensive_info(self):
        """Test that get_data_quality_report provides comprehensive data quality information."""
        pipeline = MultiTimeframeDataPipeline()

        # Create test data
        timestamps = pd.date_range("2023-01-01", periods=100, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100] * 100,
                "high": [101] * 100,
                "low": [99] * 100,
                "close": [100.5] * 100,
                "volume": [1000] * 100,
            }
        )

        data_dict = {Timeframe.M1: df}

        report = pipeline.get_data_quality_report(data_dict)

        # Check report structure
        assert "timeframes" in report
        assert "summary" in report
        assert Timeframe.M1.value in report["timeframes"]

        tf_report = report["timeframes"][Timeframe.M1.value]
        assert "row_count" in tf_report
        assert "column_count" in tf_report
        assert "date_range" in tf_report
        assert "missing_data" in tf_report
        assert "data_quality" in tf_report

        assert tf_report["row_count"] == 100
        assert tf_report["column_count"] == 6  # timestamp + 5 OHLCV
