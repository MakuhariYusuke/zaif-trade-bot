"""
Unit tests for SAC algorithm with advanced network architectures.
"""

from unittest.mock import Mock, patch

import pytest

from ztb.training.algorithms.sac.sac_algorithm import SACAlgorithm
from ztb.training.models.advanced_networks import LSTMPolicy, TransformerPolicy


class TestSACAlgorithmAdvanced:
    """Test cases for SAC algorithm with LSTM/Transformer support."""

    def test_sac_algorithm_default_config_includes_network_params(self):
        """Test that default config includes network architecture parameters."""
        algorithm = SACAlgorithm()
        config = algorithm.get_default_config()

        # Check network type parameters
        assert "network_type" in config
        assert config["network_type"] == "mlp"

        # Check LSTM parameters
        assert "lstm_hidden_size" in config
        assert "lstm_layers" in config
        assert "sequence_length" in config

        # Check Transformer parameters
        assert "transformer_d_model" in config
        assert "transformer_n_heads" in config
        assert "transformer_n_layers" in config
        assert "transformer_d_ff" in config

        # Check dropout
        assert "network_dropout" in config

    def test_validate_config_accepts_valid_network_types(self):
        """Test config validation accepts valid network types."""
        algorithm = SACAlgorithm()

        # Test MLP
        config_mlp = algorithm.get_default_config()
        config_mlp["network_type"] = "mlp"
        assert algorithm.validate_config(config_mlp)

        # Test LSTM
        config_lstm = algorithm.get_default_config()
        config_lstm["network_type"] = "lstm"
        assert algorithm.validate_config(config_lstm)

        # Test Transformer
        config_transformer = algorithm.get_default_config()
        config_transformer["network_type"] = "transformer"
        assert algorithm.validate_config(config_transformer)

    def test_validate_config_rejects_invalid_network_type(self):
        """Test config validation rejects invalid network types."""
        algorithm = SACAlgorithm()
        config = algorithm.get_default_config()
        config["network_type"] = "invalid"

        with pytest.raises(ValueError, match="Unsupported network_type"):
            algorithm.validate_config(config)

    def test_validate_config_lstm_parameters(self):
        """Test LSTM parameter validation."""
        algorithm = SACAlgorithm()
        config = algorithm.get_default_config()
        config["network_type"] = "lstm"

        # Test invalid lstm_hidden_size
        config["lstm_hidden_size"] = 0
        with pytest.raises(ValueError, match="lstm_hidden_size must be positive"):
            algorithm.validate_config(config)

        # Reset and test invalid lstm_layers
        config["lstm_hidden_size"] = 128
        config["lstm_layers"] = 0
        with pytest.raises(ValueError, match="lstm_layers must be positive"):
            algorithm.validate_config(config)

    def test_validate_config_transformer_parameters(self):
        """Test Transformer parameter validation."""
        algorithm = SACAlgorithm()
        config = algorithm.get_default_config()
        config["network_type"] = "transformer"

        # Test invalid d_model
        config["transformer_d_model"] = 0
        with pytest.raises(ValueError, match="transformer_d_model must be positive"):
            algorithm.validate_config(config)

        # Reset and test invalid n_heads
        config["transformer_d_model"] = 128
        config["transformer_n_heads"] = 0
        with pytest.raises(ValueError, match="transformer_n_heads must be positive"):
            algorithm.validate_config(config)

        # Reset and test non-divisible dimensions
        config["transformer_n_heads"] = 7  # 128 not divisible by 7
        with pytest.raises(
            ValueError,
            match="transformer_d_model must be divisible by transformer_n_heads",
        ):
            algorithm.validate_config(config)

    def test_validate_config_sequence_length(self):
        """Test sequence length validation for LSTM/Transformer."""
        algorithm = SACAlgorithm()

        # Test LSTM with invalid sequence length
        config_lstm = algorithm.get_default_config()
        config_lstm["network_type"] = "lstm"
        config_lstm["sequence_length"] = 0
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            algorithm.validate_config(config_lstm)

        # Test Transformer with invalid sequence length
        config_transformer = algorithm.get_default_config()
        config_transformer["network_type"] = "transformer"
        config_transformer["sequence_length"] = 0
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            algorithm.validate_config(config_transformer)

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_create_model_lstm_policy(self, mock_sac):
        """Test model creation with LSTM policy."""
        algorithm = SACAlgorithm()

        # Mock environment
        mock_env = Mock()
        mock_env.observation_space.shape = (100,)  # 10 timesteps * 10 features

        config = algorithm.get_default_config()
        config["network_type"] = "lstm"
        config["sequence_length"] = 10

        algorithm.create_model(mock_env, config)

        # Check that SAC was called with LSTMPolicy
        call_args = mock_sac.call_args
        assert call_args[1]["policy"] == LSTMPolicy

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_create_model_transformer_policy(self, mock_sac):
        """Test model creation with Transformer policy."""
        algorithm = SACAlgorithm()

        # Mock environment
        mock_env = Mock()
        mock_env.observation_space.shape = (100,)  # 10 timesteps * 10 features

        config = algorithm.get_default_config()
        config["network_type"] = "transformer"
        config["sequence_length"] = 10

        algorithm.create_model(mock_env, config)

        # Check that SAC was called with TransformerPolicy
        call_args = mock_sac.call_args
        assert call_args[1]["policy"] == TransformerPolicy

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC")
    def test_create_model_mlp_policy(self, mock_sac):
        """Test model creation with MLP policy (default)."""
        algorithm = SACAlgorithm()

        # Mock environment
        mock_env = Mock()
        mock_env.observation_space.shape = (100,)

        config = algorithm.get_default_config()
        config["network_type"] = "mlp"

        algorithm.create_model(mock_env, config)

        # Check that SAC was called with default policy
        call_args = mock_sac.call_args
        assert call_args[1]["policy"] == "MlpPolicy"

    def test_resolve_policy_kwargs_lstm(self):
        """Test policy kwargs resolution for LSTM."""
        policy_kwargs = SACAlgorithm._resolve_policy_kwargs(
            {"activation_fn": "relu"},
            network_type="lstm",
            sequence_length=10,
            lstm_hidden_size=64,
            lstm_layers=2,
            network_dropout=0.1,
        )

        assert policy_kwargs is not None
        assert "features_extractor_class" in policy_kwargs
        assert policy_kwargs["features_extractor_class"] == LSTMPolicy
        assert "features_extractor_kwargs" in policy_kwargs
        assert policy_kwargs["features_extractor_kwargs"]["lstm_hidden_size"] == 64
        assert policy_kwargs["features_extractor_kwargs"]["lstm_layers"] == 2
        assert policy_kwargs["features_extractor_kwargs"]["sequence_length"] == 10
        assert policy_kwargs["features_extractor_kwargs"]["dropout"] == 0.1

    def test_resolve_policy_kwargs_transformer(self):
        """Test policy kwargs resolution for Transformer."""
        policy_kwargs = SACAlgorithm._resolve_policy_kwargs(
            {"activation_fn": "relu"},
            network_type="transformer",
            sequence_length=10,
            transformer_d_model=64,
            transformer_n_heads=8,
            transformer_n_layers=4,
            transformer_d_ff=256,
            network_dropout=0.1,
        )

        assert policy_kwargs is not None
        assert "features_extractor_class" in policy_kwargs
        assert policy_kwargs["features_extractor_class"] == TransformerPolicy
        assert "features_extractor_kwargs" in policy_kwargs
        assert policy_kwargs["features_extractor_kwargs"]["d_model"] == 64
        assert policy_kwargs["features_extractor_kwargs"]["n_heads"] == 8
        assert policy_kwargs["features_extractor_kwargs"]["n_layers"] == 4
        assert policy_kwargs["features_extractor_kwargs"]["d_ff"] == 256
        assert policy_kwargs["features_extractor_kwargs"]["sequence_length"] == 10
        assert policy_kwargs["features_extractor_kwargs"]["dropout"] == 0.1

    def test_resolve_policy_kwargs_mlp(self):
        """Test policy kwargs resolution for MLP."""
        policy_kwargs = SACAlgorithm._resolve_policy_kwargs(
            {"activation_fn": "relu"},
            network_type="mlp",
        )

        assert policy_kwargs is not None
        assert "features_extractor_class" not in policy_kwargs
        assert "features_extractor_kwargs" not in policy_kwargs
