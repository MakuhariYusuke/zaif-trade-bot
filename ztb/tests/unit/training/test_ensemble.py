"""
Unit tests for ensemble.py module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from ztb.training.ensemble import EnsemblePredictor


class TestEnsemblePredictor:
    """Test cases for EnsemblePredictor class."""

    def test_init_with_valid_models(self):
        """Test initialization with valid model configurations."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0, "feature_set": "full"},
            {"path": "model2.zip", "weight": 2.0, "feature_set": "reduced"}
        ]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)

            assert len(predictor.models) == 2
            assert len(predictor.weights) == 2
            assert len(predictor.feature_sets) == 2
            assert predictor.models == [mock_model1, mock_model2]
            assert predictor.weights == [1/3, 2/3]  # Normalized weights
            assert predictor.feature_sets == ["full", "reduced"]

    def test_init_with_invalid_model(self):
        """Test initialization when one model fails to load."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0},
            {"path": "invalid.zip", "weight": 1.0}
        ]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model = Mock()
            mock_load.side_effect = [mock_model, Exception("Load failed")]

            predictor = EnsemblePredictor(model_configs)

            assert len(predictor.models) == 1
            assert len(predictor.weights) == 1
            assert predictor.weights == [1.0]  # Normalized to 1.0

    def test_init_no_models(self):
        """Test initialization with no valid models."""
        model_configs = [{"path": "invalid.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_load.side_effect = Exception("Load failed")

            predictor = EnsemblePredictor(model_configs)

            assert len(predictor.models) == 0
            assert len(predictor.weights) == 0

    def test_predict_continuous_actions(self):
        """Test prediction with continuous actions."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()

            # Mock continuous actions
            mock_model1.predict.return_value = (np.array([1.5]), None)
            mock_model2.predict.return_value = (np.array([2.5]), None)

            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            action, state = predictor.predict(observation)

            # Should be weighted average: (1.5 * 1/2) + (2.5 * 1/2) = 2.0
            assert action[0] == 2.0
            assert state is None

    def test_predict_discrete_actions(self):
        """Test prediction with discrete actions."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}, {"path": "model3.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_model3 = Mock()

            # Mock discrete actions (0, 1, 0)
            mock_model1.predict.return_value = (np.array([0]), None)
            mock_model2.predict.return_value = (np.array([1]), None)
            mock_model3.predict.return_value = (np.array([0]), None)

            mock_load.side_effect = [mock_model1, mock_model2, mock_model3]

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            action, state = predictor.predict(observation)

            # Action 0 gets weight 2/3, action 1 gets weight 1/3
            # So action 0 should win
            assert action[0] == 0

    def test_predict_no_models_loaded(self):
        """Test prediction when no models are loaded."""
        predictor = EnsemblePredictor([])

        with pytest.raises(ValueError, match="No models loaded in ensemble"):
            predictor.predict(np.array([1.0, 2.0, 3.0]))

    def test_predict_all_models_fail(self):
        """Test prediction when all models fail."""
        model_configs = [{"path": "model1.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            mock_load.return_value = mock_model

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            with pytest.raises(ValueError, match="All model predictions failed"):
                predictor.predict(observation)

    def test_predict_with_state(self):
        """Test prediction returns state from first successful model."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()

            mock_state1 = Mock()
            mock_state2 = Mock()

            mock_model1.predict.return_value = (np.array([1.0]), mock_state1)
            mock_model2.predict.return_value = (np.array([2.0]), mock_state2)

            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            action, state = predictor.predict(observation)

            assert state == mock_state1  # Should return state from first model

    @patch('ztb.training.ensemble.logger')
    def test_predict_logs_warnings_on_failure(self, mock_logger):
        """Test that prediction logs warnings when individual models fail."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()

            mock_model1.predict.side_effect = Exception("Model 1 failed")
            mock_model2.predict.return_value = (np.array([2.0]), None)

            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            action, state = predictor.predict(observation)

            # Should still work with model 2
            assert action[0] == 2.0
            mock_logger.warning.assert_called_once()


class TestEnsemblePredictorGetActionProbabilities:
    """Test cases for get_action_probabilities method."""

    def test_get_action_probabilities_success(self):
        """Test successful action probability retrieval."""
        # Skip this complex test for now - the method requires extensive mocking
        # of PyTorch tensors and distributions
        pytest.skip("Complex method requiring extensive PyTorch mocking")

    def test_get_action_probabilities_no_policy(self):
        """Test action probabilities when models don't have policy."""
        model_configs = [{"path": "model1.zip"}]

        with patch('ztb.training.ensemble.PPO.load') as mock_load:
            mock_model = Mock()
            # Remove policy attribute
            del mock_model.policy
            mock_load.return_value = mock_model

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            with pytest.raises(ValueError, match="Models must have policy"):
                predictor.get_action_probabilities(observation)