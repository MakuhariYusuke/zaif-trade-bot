"""
Tests for Schema Environment Factory
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from ztb.trading.environment.schema_env_factory import (
    create_env_from_schema,
    create_env_from_model_path
)
from ztb.utils.config import ZTBConfig


class TestSchemaEnvironmentFactory:
    """Test cases for schema-based environment creation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'open': np.random.randn(100),
        })

    @pytest.fixture
    def mock_schema_metadata(self):
        """Mock schema metadata."""
        from ztb.training.core.feature_schema_manager import FeatureSchemaMetadata

        return FeatureSchemaMetadata(
            model_name="test_model",
            num_features=5,
            feature_names=['close', 'volume', 'high', 'low', 'open'],
            schema_hash="test_hash",
            created_at="2025-01-01T00:00:00",
            training_config={"learning_rate": 0.003},
        )

    @patch('ztb.trading.environment.schema_env_factory.FeatureSchemaManager')
    @patch('ztb.trading.environment.schema_env_factory.HeavyTradingEnv')
    def test_create_env_from_schema_success(self, mock_env_class, mock_manager_class, sample_df, mock_schema_metadata):
        """Test successful environment creation from schema."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.load_schema.return_value = mock_schema_metadata
        mock_manager.load_scaler.return_value = None
        mock_manager_class.return_value = mock_manager

        mock_env = MagicMock()
        mock_env_class.return_value = mock_env

        # Execute
        result = create_env_from_schema("test_model", sample_df)

        # Verify
        assert result == mock_env
        # models_dirのデフォルト値はPath("models")
        from pathlib import Path
        mock_manager_class.assert_called_once_with("test_model", Path("models"))
        mock_manager.load_schema.assert_called_once()
        mock_manager.load_scaler.assert_called_once()

        # Check environment creation with correct config
        call_args = mock_env_class.call_args
        config = call_args[1]['config']
        assert config['feature_names'] == ['close', 'volume', 'high', 'low', 'open']
        assert config['num_features'] == 5
        assert config['schema_hash'] == "test_hash"
        assert config['model_name'] == "test_model"

    @patch('ztb.trading.environment.schema_env_factory.FeatureSchemaManager')
    def test_create_env_from_schema_missing_features(self, mock_manager_class, sample_df):
        """Test environment creation fails when required features are missing."""
        from ztb.training.core.feature_schema_manager import FeatureSchemaMetadata

        # Schema expects features that don't exist in dataframe
        mock_schema_metadata = FeatureSchemaMetadata(
            model_name="test_model",
            num_features=3,
            feature_names=['close', 'volume', 'missing_feature'],
            schema_hash="test_hash",
            created_at="2025-01-01T00:00:00",
            training_config={},
        )

        mock_manager = MagicMock()
        mock_manager.load_schema.return_value = mock_schema_metadata
        mock_manager.load_scaler.return_value = None
        mock_manager_class.return_value = mock_manager

        # Execute and expect ValueError
        with pytest.raises(ValueError, match="Dataset is missing required features"):
            create_env_from_schema("test_model", sample_df)

    @patch('ztb.trading.environment.schema_env_factory.FeatureSchemaManager')
    def test_create_env_from_schema_with_scaler(self, mock_manager_class, sample_df, mock_schema_metadata):
        """Test environment creation with scaler data."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.load_schema.return_value = mock_schema_metadata
        mock_scaler_data = {"mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "std": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
        mock_manager.load_scaler.return_value = mock_scaler_data
        mock_manager_class.return_value = mock_manager

        with patch('ztb.trading.environment.schema_env_factory.HeavyTradingEnv') as mock_env_class:
            mock_env = MagicMock()
            mock_env_class.return_value = mock_env

            # Execute
            result = create_env_from_schema("test_model", sample_df)

            # Verify scaler data is included in config
            call_args = mock_env_class.call_args
            config = call_args[1]['config']
            assert 'scaler_mean' in config
            assert 'scaler_std' in config
            np.testing.assert_array_equal(config['scaler_mean'], mock_scaler_data['mean'])
            np.testing.assert_array_equal(config['scaler_std'], mock_scaler_data['std'])

    def test_create_env_from_model_path(self, sample_df):
        """Test environment creation from model path."""
        config = ZTBConfig()
        with patch('ztb.trading.environment.schema_env_factory.create_env_from_schema') as mock_create_env:
            mock_env = MagicMock()
            mock_create_env.return_value = mock_env

            # Execute
            result = create_env_from_model_path(config.get_model_path("test_model.zip"), sample_df)

            # Verify
            assert result == mock_env
            mock_create_env.assert_called_once()
            call_args = mock_create_env.call_args
            assert call_args[0][0] == "test_model"  # model_name extracted from path
            # Use pandas.testing to compare DataFrames
            import pandas.testing as pdt
            pdt.assert_frame_equal(call_args[0][1], sample_df)
            assert call_args[0][2] is None  # config
            assert isinstance(call_args[0][3], Path)  # models_dir