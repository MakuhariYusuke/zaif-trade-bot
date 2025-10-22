"""Tests for DataProcessor component."""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.environment.components.data_processor import DataProcessor


@pytest.fixture
def data_processor():
    """DataProcessor instance for testing."""
    return DataProcessor(
        preprocess_chunk_size=32,
        memory_logging_enabled=False,
        gc_step_interval=0,
    )


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "open": [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)],
        "high": [105 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)],
        "low": [95 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)],
        "close": [102 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)],
        "volume": [1000 + i * 10 + np.random.normal(0, 50) for i in range(100)],
        "rsi": [50 + np.sin(i * 0.1) * 20 for i in range(100)],
        "macd": [np.cos(i * 0.1) * 5 for i in range(100)],
        "ts": dates,
        "timestamp": dates,
        "exchange": ["binance"] * 100,
        "pair": ["BTC/USDT"] * 100,
        "episode_id": list(range(100)),
        "side": ["buy"] * 100,
        "source": ["api"] * 100,
    }
    df = pd.DataFrame(data, index=dates)
    # Add some NaN values for testing
    df.iloc[10:16, df.columns.get_loc("close")] = np.nan
    df.iloc[20:26, df.columns.get_loc("volume")] = np.nan
    return df


class TestDataProcessorInitialization:
    """Test DataProcessor initialization."""

    def test_initialization(self):
        """Test proper initialization."""
        processor = DataProcessor(
            preprocess_chunk_size=64,
            memory_logging_enabled=True,
            gc_step_interval=1000,
        )
        assert processor._preprocess_chunk_size == 64
        assert processor._memory_logging_enabled is True
        assert processor._gc_step_interval == 1000

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        processor = DataProcessor()
        assert processor._preprocess_chunk_size == 32
        assert processor._memory_logging_enabled is False
        assert processor._gc_step_interval == 0


class TestDataProcessorPreprocessData:
    """Test preprocess_data method."""

    def test_preprocess_data_basic(self, data_processor, sample_dataframe):
        """Test basic data preprocessing."""
        result = data_processor.preprocess_data(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == len(sample_dataframe)

        # Check that excluded columns are removed
        excluded_cols = [
            "ts",
            "timestamp",
            "exchange",
            "pair",
            "episode_id",
            "side",
            "source",
        ]
        for col in excluded_cols:
            assert col not in result.columns

        # Check that NaN values are filled
        assert not result.isnull().any().any()

    def test_preprocess_data_empty_dataframe(self, data_processor):
        """Test preprocessing with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = data_processor.preprocess_data(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_preprocess_data_numeric_conversion(self, data_processor):
        """Test numeric column conversion."""
        df = pd.DataFrame(
            {
                "float_col": [1.0, 2.0, 3.0],
                "int_col": [1, 2, 3],
                "bool_col": [True, False, True],
                "str_col": ["a", "b", "c"],
            }
        )

        result = data_processor.preprocess_data(df)

        # Check float conversion
        assert result["float_col"].dtype == np.float32

        # Check bool conversion
        assert result["bool_col"].dtype == np.int8

    def test_preprocess_data_index_reset(self, data_processor):
        """Test index reset functionality."""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            },
            index=[10, 20, 30],
        )  # Non-standard index

        result = data_processor.preprocess_data(df)

        # Index should be reset to RangeIndex
        assert isinstance(result.index, pd.RangeIndex)
        assert list(result.index) == [0, 1, 2]


class TestDataProcessorStreamingMethods:
    """Test streaming-related methods."""

    def test_fetch_streaming_snapshot_no_pipeline(self, data_processor):
        """Test fetch_streaming_snapshot with no pipeline."""
        result = data_processor.fetch_streaming_snapshot(
            streaming_pipeline=None,
            required_rows=100,
            stream_batch_size=32,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_prepare_stream_batch_empty(self, data_processor):
        """Test prepare_stream_batch with empty batch."""
        empty_batch = pd.DataFrame()
        base_columns = ["close", "volume"]
        base_df = pd.DataFrame()

        result = data_processor.prepare_stream_batch(empty_batch, base_columns, base_df)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_prepare_stream_batch_with_data(self, data_processor):
        """Test prepare_stream_batch with data."""
        batch = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "volume": [1000, 1100],
            }
        )
        base_columns = ["close", "volume", "rsi"]
        base_df = pd.DataFrame(columns=["close", "volume", "rsi"])

        result = data_processor.prepare_stream_batch(batch, base_columns, base_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "rsi" in result.columns  # Should be added with default value

    def test_prepare_stream_batch_missing_columns(self, data_processor):
        """Test prepare_stream_batch with missing columns."""
        batch = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                # Missing volume column
            }
        )
        base_columns = ["close", "volume"]
        base_df = pd.DataFrame()

        result = data_processor.prepare_stream_batch(batch, base_columns, base_df)

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert "volume" in result.columns
        # Missing column should be filled with 0.0
        assert (result["volume"] == 0.0).all()


class TestDataProcessorFeatureStorage:
    """Test feature storage dtype methods."""

    def test_apply_feature_storage_dtype_float32(self, data_processor):
        """Test feature storage dtype application with float32."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "rsi": [50.0, 51.0, 52.0],
            }
        )
        features = ["close", "volume", "rsi"]
        config = {"feature_storage_dtype": "float32"}

        data_processor.apply_feature_storage_dtype(df, features, config)

        for feature in features:
            assert df[feature].dtype == np.float32

    def test_apply_feature_storage_dtype_float16(self, data_processor):
        """Test feature storage dtype application with float16."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "rsi": [50.0, 51.0, 52.0],
            }
        )
        features = ["close", "volume", "rsi"]
        config = {"feature_storage_dtype": "float16"}

        data_processor.apply_feature_storage_dtype(df, features, config)

        for feature in features:
            assert df[feature].dtype == np.float16

    def test_apply_feature_storage_dtype_with_protected_columns(self, data_processor):
        """Test feature storage with protected columns."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "rsi": [50.0, 51.0, 52.0],
            }
        )
        # First preprocess the data to convert to float32
        df = data_processor.preprocess_data(df)

        features = ["close", "volume", "rsi"]
        config = {
            "feature_storage_dtype": "float16",
            "precision_columns": ["close"],  # Protect close from conversion
        }

        data_processor.apply_feature_storage_dtype(df, features, config)

        # Protected column should remain float32
        assert df["close"].dtype == np.float32
        # Others should be converted to float16
        assert df["volume"].dtype == np.float16
        assert df["rsi"].dtype == np.float16

    def test_apply_feature_storage_dtype_overflow_protection(self, data_processor):
        """Test overflow protection for float16."""
        # Create values that would overflow float16
        large_value = 100000.0  # Large value that exceeds float16 range
        df = pd.DataFrame(
            {
                "close": [large_value, large_value, large_value],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )
        # First preprocess the data
        df = data_processor.preprocess_data(df)

        features = ["close", "volume"]
        config = {"feature_storage_dtype": "float16"}

        data_processor.apply_feature_storage_dtype(df, features, config)

        # Large values should not be converted to avoid overflow
        assert df["close"].dtype == np.float32  # Should remain float32
        assert df["volume"].dtype == np.float16  # Should be converted
