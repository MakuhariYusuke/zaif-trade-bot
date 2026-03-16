#!/usr/bin/env python3
"""
Unit tests for Ensemble functionality.

Tests cover:
- EnsemblePredictor initialization and model loading
- Prediction aggregation with weighted voting
- Error handling for failed model predictions
- Edge cases and robustness
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ztb.training.ensemble import EnsemblePredictor, ModelConfig


class TestEnsemblePredictor:
    """Test EnsemblePredictor functionality."""

    @pytest.fixture
    def sample_model_configs(self):
        """Sample model configurations for testing."""
        return [
            {"path": "/path/to/model1.zip", "weight": 1.0, "feature_set": "basic"},
            {"path": "/path/to/model2.zip", "weight": 2.0, "feature_set": "enhanced"},
            {"path": "/path/to/model3.zip", "weight": 1.5, "feature_set": "full"},
        ]

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        model1 = Mock()
        model1.predict.return_value = (np.array([0]), None)  # Action 0

        model2 = Mock()
        model2.predict.return_value = (np.array([1]), None)  # Action 1

        model3 = Mock()
        model3.predict.return_value = (np.array([2]), None)  # Action 2

        return [model1, model2, model3]

    @patch("ztb.training.ensemble.PPO.load")
    def test_initialization_success(
        self, mock_ppo_load, sample_model_configs, mock_models
    ):
        """Test successful ensemble initialization."""
        # Setup mock loading
        mock_ppo_load.side_effect = mock_models

        ensemble = EnsemblePredictor(sample_model_configs)

        assert len(ensemble.models) == 3
        assert len(ensemble.weights) == 3
        assert len(ensemble.feature_sets) == 3

        # Check weights are normalized (1.0 + 2.0 + 1.5 = 4.5)
        expected_weights = [1.0 / 4.5, 2.0 / 4.5, 1.5 / 4.5]
        np.testing.assert_array_almost_equal(ensemble.weights, expected_weights)

        assert ensemble.feature_sets == ["basic", "enhanced", "full"]

    @patch("ztb.training.ensemble.PPO.load")
    def test_initialization_partial_failure(self, mock_ppo_load, sample_model_configs):
        """Test ensemble initialization with some model loading failures."""
        # Make second model fail to load
        model1 = Mock()
        model3 = Mock()

        def load_side_effect(path):
            if "model2" in path:
                raise FileNotFoundError(f"Model not found: {path}")
            elif "model1" in path:
                return model1
            elif "model3" in path:
                return model3
            else:
                raise ValueError(f"Unexpected path: {path}")

        mock_ppo_load.side_effect = load_side_effect

        ensemble = EnsemblePredictor(sample_model_configs)

        # Should have only 2 models (model2 failed)
        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2

        # Weights should be normalized for remaining models
        expected_weights = [1.0 / 2.5, 1.5 / 2.5]  # (1.0 + 1.5) = 2.5
        np.testing.assert_array_almost_equal(ensemble.weights, expected_weights)

    @patch("ztb.training.ensemble.PPO.load")
    def test_initialization_all_failures(self, mock_ppo_load, sample_model_configs):
        """Test ensemble initialization when all models fail to load."""
        mock_ppo_load.side_effect = FileNotFoundError("Model not found")

        ensemble = EnsemblePredictor(sample_model_configs)

        assert len(ensemble.models) == 0
        assert len(ensemble.weights) == 0

    def test_predict_no_models(self, sample_model_configs):
        """Test prediction fails when no models are loaded."""
        ensemble = EnsemblePredictor([])  # Empty config list

        observation = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="No models loaded in ensemble"):
            ensemble.predict(observation)

    @patch("ztb.training.ensemble.PPO.load")
    def test_predict_discrete_actions_weighted_voting(
        self, mock_ppo_load, sample_model_configs
    ):
        """Test prediction with discrete actions using weighted voting."""
        # Setup models with different predictions
        model1 = Mock()
        model1.predict.return_value = (np.array([0]), None)  # Action 0, weight 1.0

        model2 = Mock()
        model2.predict.return_value = (np.array([1]), None)  # Action 1, weight 2.0

        model3 = Mock()
        model3.predict.return_value = (np.array([1]), None)  # Action 1, weight 1.5

        mock_ppo_load.side_effect = [model1, model2, model3]

        ensemble = EnsemblePredictor(sample_model_configs)
        observation = np.array([[1.0, 2.0, 3.0]])

        action, state = ensemble.predict(observation, deterministic=True)

        # Action 1 should win: weight 0 + 2.0 + 1.5 = 3.5 > weight 0: 1.0
        assert action[0] == 1
        assert state is None

        # Verify all models were called
        model1.predict.assert_called_once_with(observation, deterministic=True)
        model2.predict.assert_called_once_with(observation, deterministic=True)
        model3.predict.assert_called_once_with(observation, deterministic=True)

    @patch("ztb.training.ensemble.PPO.load")
    def test_predict_continuous_actions_weighted_average(
        self, mock_ppo_load, sample_model_configs
    ):
        """Test prediction with continuous actions using weighted averaging."""
        # Setup models with continuous predictions
        model1 = Mock()
        model1.predict.return_value = (np.array([1.0]), None)  # weight 1.0

        model2 = Mock()
        model2.predict.return_value = (np.array([3.0]), None)  # weight 2.0

        model3 = Mock()
        model3.predict.return_value = (np.array([5.0]), None)  # weight 1.5

        mock_ppo_load.side_effect = [model1, model2, model3]

        ensemble = EnsemblePredictor(sample_model_configs)
        observation = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        action, state = ensemble.predict(observation, deterministic=True)

        # Weighted average: (1.0*1.0 + 3.0*2.0 + 5.0*1.5) / (1.0+2.0+1.5) = (1.0 + 6.0 + 7.5) / 4.5 = 14.5 / 4.5 ≈ 3.222
        expected_action = np.array([14.5 / 4.5])
        np.testing.assert_array_almost_equal(action, expected_action)

    @patch("ztb.training.ensemble.PPO.load")
    def test_predict_with_model_failures(self, mock_ppo_load, sample_model_configs):
        """Test prediction robustness when some models fail."""
        # Setup models where middle one fails
        model1 = Mock()
        model1.predict.return_value = (np.array([0]), None)

        model3 = Mock()
        model3.predict.return_value = (np.array([2]), None)

        def predict_side_effect(observation, deterministic=True):
            if model1.predict.call_count > 0 and not model1.predict.called:
                pass  # model1 not called yet
            elif (
                len(
                    [
                        call
                        for call in model1.predict.call_args_list
                        if call[0][0] is observation
                    ]
                )
                > 0
            ):
                return model1.predict.return_value
            else:
                raise RuntimeError("Model prediction failed")

        mock_ppo_load.side_effect = [model1, RuntimeError("Load failed"), model3]

        ensemble = EnsemblePredictor(sample_model_configs)
        observation = np.array([[1.0, 2.0, 3.0]])

        action, state = ensemble.predict(observation, deterministic=True)

        # Should succeed with 2 models, action 2 should win (only action available)
        assert action[0] == 2
        assert state is None

    @patch("ztb.training.ensemble.PPO.load")
    def test_predict_all_models_fail(self, mock_ppo_load, sample_model_configs):
        """Test prediction fails when all models fail."""
        # All models fail during prediction
        model1 = Mock()
        model1.predict.side_effect = RuntimeError("Prediction failed")

        model2 = Mock()
        model2.predict.side_effect = RuntimeError("Prediction failed")

        model3 = Mock()
        model3.predict.side_effect = RuntimeError("Prediction failed")

        mock_ppo_load.side_effect = [model1, model2, model3]

        ensemble = EnsemblePredictor(sample_model_configs)
        observation = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="All 3 model predictions failed"):
            ensemble.predict(observation)

    @patch("ztb.training.ensemble.PPO.load")
    def test_predict_deterministic_parameter(
        self, mock_ppo_load, sample_model_configs, mock_models
    ):
        """Test deterministic parameter is passed to models."""
        mock_ppo_load.side_effect = mock_models

        ensemble = EnsemblePredictor(sample_model_configs)
        observation = np.array([[1.0, 2.0, 3.0]])

        # Test deterministic=True
        ensemble.predict(observation, deterministic=True)
        for model in mock_models:
            model.predict.assert_called_with(observation, deterministic=True)

        # Reset mocks
        for model in mock_models:
            model.reset_mock()

        # Test deterministic=False
        ensemble.predict(observation, deterministic=False)
        for model in mock_models:
            model.predict.assert_called_with(observation, deterministic=False)

    @patch("ztb.training.ensemble.PPO.load")
    def test_get_action_probabilities_not_implemented(
        self, mock_ppo_load, sample_model_configs, mock_models
    ):
        """Test get_action_probabilities method (currently not fully implemented)."""
        # Setup models with proper policy mocks
        for model in mock_models:
            model.policy = Mock()
            model.policy.obs_to_tensor.return_value = [Mock()]
            model.policy.return_value = (None, None, None)  # logits, values, entropy
            model.policy.get_distribution.return_value = (
                None  # No distribution available
            )

        mock_ppo_load.side_effect = mock_models

        ensemble = EnsemblePredictor(sample_model_configs)
        observation = np.array([[1.0, 2.0, 3.0]])

        # This method exists but may not be fully implemented
        # Just test it doesn't crash
        try:
            result = ensemble.get_action_probabilities(observation)
            # If implemented, should return tuple of arrays
            assert isinstance(result, tuple)
            assert len(result) == 2
        except NotImplementedError:
            # Acceptable if not implemented yet
            pass
        except ValueError as e:
            # Expected when no probabilities can be extracted
            assert "Could not get probabilities" in str(e)
        except Exception as e:
            # Other exceptions should be investigated
            pytest.fail(f"Unexpected exception: {e}")


class TestModelConfig:
    """Test ModelConfig TypedDict usage."""

    def test_model_config_creation(self):
        """Test ModelConfig can be created with valid data."""
        config: ModelConfig = {
            "path": "/path/to/model.zip",
            "weight": 1.5,
            "feature_set": "enhanced",
        }

        assert config["path"] == "/path/to/model.zip"
        assert config["weight"] == 1.5
        assert config["feature_set"] == "enhanced"

    def test_model_config_optional_fields(self):
        """Test ModelConfig with optional fields omitted."""
        config: ModelConfig = {
            "path": "/path/to/model.zip"
            # weight and feature_set are optional
        }

        assert config["path"] == "/path/to/model.zip"
        assert "weight" not in config
        assert "feature_set" not in config
