"""Tests for ObservationBuilder component."""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.environment.components.observation_builder import ObservationBuilder


@pytest.fixture
def sample_features():
    """Sample feature names for testing."""
    return ["close", "volume", "rsi", "macd"]


@pytest.fixture
def sample_feature_matrix(sample_features):
    """Sample feature matrix for testing."""
    rows = 100
    cols = len(sample_features)
    return np.random.randn(rows, cols).astype(np.float32)


@pytest.fixture
def sample_scaler():
    """Sample scaler parameters for testing."""
    features = 4
    return {
        "mean": np.random.randn(features).astype(np.float32),
        "std": np.abs(np.random.randn(features)).astype(np.float32)
        + 0.1,  # Ensure positive std
    }


@pytest.fixture
def observation_builder(sample_features, sample_feature_matrix, sample_scaler):
    """ObservationBuilder instance for testing."""
    return ObservationBuilder(
        features=sample_features,
        feature_matrix=sample_feature_matrix,
        nonfinite_rows=set(),
        nonfinite_warned_rows=set(),
        scaler_mean=sample_scaler["mean"],
        scaler_std=sample_scaler["std"],
    )


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    data = {
        "close": [100 + i * 0.1 for i in range(50)],
        "volume": [1000 + i * 10 for i in range(50)],
        "rsi": [50 + np.sin(i * 0.1) * 20 for i in range(50)],
        "macd": [np.cos(i * 0.1) * 5 for i in range(50)],
    }
    return pd.DataFrame(data, index=dates)


class TestObservationBuilderInitialization:
    """Test ObservationBuilder initialization."""

    def test_initialization(self, observation_builder, sample_features):
        """Test proper initialization."""
        assert observation_builder.features == sample_features
        assert observation_builder.scaler_mean is not None
        assert observation_builder.scaler_std is not None
        assert observation_builder._nonfinite_rows == set()
        assert observation_builder._nonfinite_warned_rows == set()

    def test_initialization_without_scaler(
        self, sample_features, sample_feature_matrix
    ):
        """Test initialization without scaler."""
        builder = ObservationBuilder(
            features=sample_features,
            feature_matrix=sample_feature_matrix,
            nonfinite_rows=set(),
            nonfinite_warned_rows=set(),
        )
        assert builder.scaler_mean is None
        assert builder.scaler_std is None


class TestObservationBuilderGetObservation:
    """Test get_observation method."""

    def test_get_observation_with_feature_matrix(
        self, observation_builder, sample_dataframe
    ):
        """Test observation retrieval with feature matrix."""
        obs = observation_builder.get_observation(
            current_step=10,
            n_steps=50,
            df=sample_dataframe,
        )
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == len(observation_builder.features)

    def test_get_observation_with_scaler(self, observation_builder, sample_dataframe):
        """Test observation with scaling applied."""
        obs_scaled = observation_builder.get_observation(
            current_step=10,
            n_steps=50,
            df=sample_dataframe,
        )

        # Create builder without scaler for comparison
        builder_no_scaler = ObservationBuilder(
            features=observation_builder.features,
            feature_matrix=observation_builder._feature_matrix,
            nonfinite_rows=set(),
            nonfinite_warned_rows=set(),
        )
        obs_unscaled = builder_no_scaler.get_observation(
            current_step=10,
            n_steps=50,
            df=sample_dataframe,
        )

        # Scaled and unscaled should be different
        assert not np.allclose(obs_scaled, obs_unscaled)

    def test_get_observation_boundary_cases(
        self, observation_builder, sample_dataframe
    ):
        """Test observation at boundary steps."""
        # Test step 0
        obs_0 = observation_builder.get_observation(
            current_step=0,
            n_steps=50,
            df=sample_dataframe,
        )
        assert isinstance(obs_0, np.ndarray)

        # Test step beyond n_steps
        obs_beyond = observation_builder.get_observation(
            current_step=60,
            n_steps=50,
            df=sample_dataframe,
        )
        assert isinstance(obs_beyond, np.ndarray)

    def test_get_observation_fallback_path(self, sample_features, sample_dataframe):
        """Test fallback path when feature matrix is empty."""
        # Create builder with empty feature matrix
        empty_matrix = np.array([]).reshape(0, len(sample_features)).astype(np.float32)
        builder = ObservationBuilder(
            features=sample_features,
            feature_matrix=empty_matrix,
            nonfinite_rows=set(),
            nonfinite_warned_rows=set(),
        )

        obs = builder.get_observation(
            current_step=10,
            n_steps=50,
            df=sample_dataframe,
        )
        assert isinstance(obs, np.ndarray)
        assert len(obs) == len(sample_features)


class TestObservationBuilderGetInfo:
    """Test get_info method."""

    def test_get_info_basic(self, observation_builder):
        """Test basic info retrieval."""
        from unittest.mock import Mock

        config = Mock()
        info = observation_builder.get_info(
            current_step=10,
            n_steps=100,
            position=0.5,
            total_pnl=1500.0,
            trades_count=5,
            features=observation_builder.features,
            config=config,
        )

        assert isinstance(info, dict)
        assert info["current_step"] == 10
        assert info["total_steps"] == 100
        assert info["position"] == 0.5
        assert info["total_pnl"] == 1500.0
        assert info["trades_count"] == 5
        assert info["features"] == observation_builder.features

    def test_get_info_with_different_values(self, observation_builder):
        """Test info with different parameter values."""
        from unittest.mock import Mock

        config = Mock()
        info = observation_builder.get_info(
            current_step=25,
            n_steps=200,
            position=-0.3,
            total_pnl=-500.0,
            trades_count=12,
            features=["custom_feature"],
            config=config,
        )

        assert info["current_step"] == 25
        assert info["position"] == -0.3
        assert info["total_pnl"] == -500.0


class TestObservationBuilderNonfiniteHandling:
    """Test handling of non-finite values."""

    def test_nonfinite_warning(self, sample_features, sample_feature_matrix):
        """Test warning for non-finite values."""
        # Add some non-finite rows
        nonfinite_rows = {5, 15}
        builder = ObservationBuilder(
            features=sample_features,
            feature_matrix=sample_feature_matrix,
            nonfinite_rows=nonfinite_rows,
            nonfinite_warned_rows=set(),
        )

        # This should trigger warning for row 5
        obs = builder.get_observation(
            current_step=5,
            n_steps=50,
            df=pd.DataFrame(),  # Not used in this path
        )
        assert isinstance(obs, np.ndarray)
        assert 5 in builder._nonfinite_warned_rows

    def test_nonfinite_no_duplicate_warnings(
        self, sample_features, sample_feature_matrix
    ):
        """Test that warnings are not duplicated for same row."""
        nonfinite_rows = {5}
        warned_rows = {5}  # Already warned
        builder = ObservationBuilder(
            features=sample_features,
            feature_matrix=sample_feature_matrix,
            nonfinite_rows=nonfinite_rows,
            nonfinite_warned_rows=warned_rows,
        )

        # Should not trigger additional warning
        obs = builder.get_observation(
            current_step=5,
            n_steps=50,
            df=pd.DataFrame(),
        )
        assert isinstance(obs, np.ndarray)
