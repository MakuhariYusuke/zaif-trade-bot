"""
Unit tests for ztb.cache.parquet_io module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

try:
    from ztb.cache.parquet_io import (
        analyze_column_dependencies,
        convert_to_parquet,
        load_config,
        load_features_config,
        read_parquet,
        read_parquet_with_features,
        smart_column_detection,
        write_parquet,
    )
except ImportError:
    pytest.skip("ztb.cache.parquet_io module not available", allow_module_level=True)


class TestParquetIO:
    """Test cases for parquet_io functions."""

    def test_load_config(self, tmp_path):
        """Test load_config function."""
        config_file = tmp_path / "test.yaml"
        config_data = {"parquet": {"compression": "snappy"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(config_file)
        assert result == config_data

    def test_load_config_file_not_found(self):
        """Test load_config when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_load_features_config(self, tmp_path):
        """Test load_features_config function."""
        features_file = tmp_path / "features.yaml"
        features_data = {"features": {"technical": ["rsi", "macd"]}}
        with open(features_file, "w") as f:
            yaml.dump(features_data, f)

        result = load_features_config(features_file)
        assert result == features_data

    def test_load_features_config_not_found(self):
        """Test load_features_config when file doesn't exist."""
        with patch("ztb.cache.parquet_io.print") as mock_print:
            result = load_features_config(Path("nonexistent.yaml"))
            assert result == {}
            mock_print.assert_called_once()

    def test_analyze_column_dependencies(self):
        """Test analyze_column_dependencies function."""
        features_config = {
            "features": {
                "technical": [
                    {"name": "rsi", "dependencies": ["close", "high", "low"]},
                    {"name": "sma_20", "dependencies": ["close"]},
                    "macd",
                ]
            }
        }

        result = analyze_column_dependencies(features_config)

        # Should include base columns plus dependencies
        expected = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "close",
            "high",
            "low",
        }
        assert result == expected

    def test_analyze_column_dependencies_target_features(self):
        """Test analyze_column_dependencies with target features."""
        features_config = {
            "features": {
                "technical": [
                    {"name": "rsi", "dependencies": ["close"]},
                    {"name": "macd", "dependencies": ["close", "volume"]},
                ]
            }
        }

        result = analyze_column_dependencies(features_config, ["rsi"])

        # Should only include dependencies for rsi
        expected = {"open", "high", "low", "close", "volume", "timestamp", "close"}
        assert result == expected

    @patch("ztb.cache.parquet_io.pq.ParquetFile")
    def test_smart_column_detection(self, mock_parquet_file):
        """Test smart_column_detection function."""
        mock_schema = MagicMock()
        mock_schema.names = ["col1", "col2", "col3", "col4"]
        mock_parquet_file.return_value.schema = mock_schema

        required_columns = {"col1", "col3", "missing"}

        result = smart_column_detection(Path("test.parquet"), required_columns)

        assert result == ["col1", "col3"]

    @patch("ztb.cache.parquet_io.pq.ParquetFile")
    def test_smart_column_detection_no_required(self, mock_parquet_file):
        """Test smart_column_detection without required columns."""
        mock_schema = MagicMock()
        mock_schema.names = ["col1", "col2", "col3"]
        mock_parquet_file.return_value.schema = mock_schema

        result = smart_column_detection(Path("test.parquet"), None)

        assert result == ["col1", "col2", "col3"]

    @patch("ztb.cache.parquet_io.pq.ParquetFile")
    @patch("ztb.cache.parquet_io.print")
    def test_smart_column_detection_error(self, mock_print, mock_parquet_file):
        """Test smart_column_detection with error."""
        mock_parquet_file.side_effect = Exception("Read error")

        result = smart_column_detection(Path("test.parquet"), {"col1"})

        assert result == []
        mock_print.assert_called_once()

    @patch("ztb.cache.parquet_io.load_config")
    @patch("ztb.cache.parquet_io.pq.write_table")
    @patch("ztb.cache.parquet_io.pa")
    def test_write_parquet(self, mock_pa, mock_write_table, mock_load_config):
        """Test write_parquet function."""
        mock_load_config.return_value = {"parquet": {"compression": "gzip"}}
        mock_table = MagicMock()
        mock_pa.Table.from_pandas.return_value = mock_table

        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        write_parquet(df, Path("test.parquet"))

        mock_pa.Table.from_pandas.assert_called_once_with(df)
        mock_write_table.assert_called_once()

    @patch("ztb.cache.parquet_io.load_config")
    @patch("ztb.cache.parquet_io.psutil.Process")
    @patch("ztb.cache.parquet_io.pd.read_parquet")
    def test_read_parquet(self, mock_read_parquet, mock_process, mock_load_config):
        """Test read_parquet function."""
        mock_load_config.return_value = {"limits": {"peak_memory_mb": 1000}}
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500MB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        df = pd.DataFrame({"col1": [1, 2]})
        mock_read_parquet.return_value = df

        result = read_parquet(Path("test.parquet"))

        assert isinstance(result, pd.DataFrame)
        mock_read_parquet.assert_called_once()

    @patch("ztb.cache.parquet_io.load_config")
    @patch("ztb.cache.parquet_io.load_features_config")
    @patch("ztb.cache.parquet_io.smart_column_detection")
    @patch("ztb.cache.parquet_io.read_parquet")
    def test_read_parquet_with_features(
        self,
        mock_read_parquet,
        mock_smart_detection,
        mock_load_features,
        mock_load_config,
    ):
        """Test read_parquet_with_features function."""
        mock_load_config.return_value = {}
        mock_load_features.return_value = {
            "rsi": {"dependencies": ["close"]},
            "macd": {"dependencies": ["close", "volume"]},
        }
        mock_smart_detection.return_value = ["close", "volume"]
        mock_read_parquet.return_value = pd.DataFrame(
            {"close": [1, 2], "volume": [3, 4]}
        )

        result = read_parquet_with_features(
            Path("test.parquet"), target_features=["rsi"]
        )

        assert isinstance(result, pd.DataFrame)
        mock_read_parquet.assert_called_once()

    @patch("ztb.cache.parquet_io.load_config")
    @patch("ztb.cache.parquet_io.write_parquet")
    @patch("ztb.cache.parquet_io.print")
    def test_convert_to_parquet_csv(
        self, mock_print, mock_write_parquet, mock_load_config
    ):
        """Test convert_to_parquet with CSV input."""
        mock_load_config.return_value = {}

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            tmp_file.write(b"col1,col2\n1,2\n3,4\n")
            tmp_file_path = Path(tmp_file.name)

        try:
            with patch("ztb.cache.parquet_io.pd.read_csv") as mock_read_csv:
                mock_read_csv.return_value = pd.DataFrame(
                    {"col1": [1, 3], "col2": [2, 4]}
                )

                convert_to_parquet(tmp_file_path, Path("output.parquet"))

                mock_read_csv.assert_called_once_with(tmp_file_path)
                mock_write_parquet.assert_called_once()
        finally:
            tmp_file_path.unlink(missing_ok=True)

    @patch("ztb.cache.parquet_io.load_config")
    @patch("ztb.cache.parquet_io.write_parquet")
    def test_convert_to_parquet_unsupported(self, mock_write_parquet, mock_load_config):
        """Test convert_to_parquet with unsupported format."""
        mock_load_config.return_value = {}

        with pytest.raises(ValueError, match="Unsupported input format"):
            convert_to_parquet(Path("test.txt"), Path("output.parquet"))
