"""
Unit tests for ensemble.py module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ztb.training.models.ensemble import EnsemblePredictor


class TestEnsemblePredictor:
    """Test cases for EnsemblePredictor class."""

    def test_init_with_valid_models(self):
        """Test initialization with valid model configurations."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0, "feature_set": "full"},
            {"path": "model2.zip", "weight": 2.0, "feature_set": "reduced"},
        ]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)

            assert len(predictor.models) == 2
            assert len(predictor.weights) == 2
            assert len(predictor.feature_sets) == 2
            assert predictor.models == [mock_model1, mock_model2]
            assert predictor.weights == [1 / 3, 2 / 3]  # Normalized weights
            assert predictor.feature_sets == ["full", "reduced"]

    def test_init_with_invalid_model(self):
        """Test initialization when one model fails to load."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0},
            {"path": "invalid.zip", "weight": 1.0},
        ]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model = Mock()
            mock_load.side_effect = [mock_model, Exception("Load failed")]

            predictor = EnsemblePredictor(model_configs)

            assert len(predictor.models) == 1
            assert len(predictor.weights) == 1
            assert predictor.weights == [1.0]  # Normalized to 1.0

    def test_init_no_models(self):
        """Test initialization with no valid models."""
        model_configs = [{"path": "invalid.zip"}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_load.side_effect = Exception("Load failed")

            with pytest.raises(RuntimeError, match="Failed to load any models"):
                EnsemblePredictor(model_configs)

    def test_predict_continuous_actions(self):
        """Test prediction with continuous actions."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
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
        model_configs = [
            {"path": "model1.zip"},
            {"path": "model2.zip"},
            {"path": "model3.zip"},
        ]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
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
        with pytest.raises(ValueError, match="At least one model configuration required"):
            EnsemblePredictor([])

    def test_predict_all_models_fail(self):
        """Test prediction when all models fail."""
        model_configs = [{"path": "model1.zip"}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            mock_load.return_value = mock_model

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            with pytest.raises(ValueError, match=r"All \d+ model predictions failed"):
                predictor.predict(observation)

    def test_predict_with_state(self):
        """Test prediction returns state from first successful model."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
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

    @patch("ztb.training.ensemble.logger")
    def test_predict_logs_warnings_on_failure(self, mock_logger):
        """Test that prediction logs warnings when individual models fail."""
        model_configs = [{"path": "model1.zip"}, {"path": "model2.zip"}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
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

    def test_get_action_probabilities_no_policy(self):
        """Test action probabilities when models don't have policy."""
        model_configs = [{"path": "model1.zip"}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model = Mock()
            # Remove policy attribute
            del mock_model.policy
            mock_load.return_value = mock_model

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            with pytest.raises(ValueError, match="Models must have policy"):
                predictor.get_action_probabilities(observation)

    def test_predict_with_different_weights(self):
        """Test prediction with different model weights."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0},
            {"path": "model2.zip", "weight": 3.0},
        ]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model1 = Mock()
            mock_model1.predict.return_value = (np.array([0.1, 0.9]), None)
            mock_model2 = Mock()
            mock_model2.predict.return_value = (np.array([0.3, 0.7]), None)
            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            result = predictor.predict(observation)

            # Expected: (0.1 * 0.25 + 0.3 * 0.75, None) = (0.025 + 0.225, None) = (0.25, None)
            expected_action = np.array([0.25, 0.75])
            np.testing.assert_array_almost_equal(result[0], expected_action)

    def test_predict_deterministic_vs_stochastic(self):
        """Test prediction in deterministic vs stochastic modes."""
        model_configs = [{"path": "model1.zip", "weight": 1.0}]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = (np.array([0.2, 0.8]), None)
            mock_load.return_value = mock_model

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            # Test deterministic
            result_det = predictor.predict(observation, deterministic=True)
            # Test stochastic
            result_stoch = predictor.predict(observation, deterministic=False)

            # Both should call predict with correct deterministic flag
            assert mock_model.predict.call_count == 2
            mock_model.predict.assert_any_call(observation, deterministic=True)
            mock_model.predict.assert_any_call(observation, deterministic=False)

    def test_get_action_probabilities(self):
        """Test getting action probabilities from ensemble."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0},
            {"path": "model2.zip", "weight": 1.0},
        ]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_model1 = Mock()
            mock_model1.policy = Mock()
            # Mock obs_to_tensor to return a tuple
            mock_model1.policy.obs_to_tensor.return_value = (Mock(),)
            # Mock policy call
            mock_model1.policy.return_value = (None, None, None)
            # Mock get_distribution
            mock_distribution = Mock()
            mock_distribution.distribution = Mock()
            mock_distribution.distribution.probs = Mock()
            mock_distribution.distribution.probs.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.1, 0.9]])
            mock_model1.policy.get_distribution.return_value = mock_distribution
            
            mock_model2 = Mock()
            mock_model2.policy = Mock()
            mock_model2.policy.obs_to_tensor.return_value = (Mock(),)
            mock_model2.policy.return_value = (None, None, None)
            mock_distribution2 = Mock()
            mock_distribution2.distribution = Mock()
            mock_distribution2.distribution.probs = Mock()
            mock_distribution2.distribution.probs.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.3, 0.7]])
            mock_model2.policy.get_distribution.return_value = mock_distribution2
            
            mock_load.side_effect = [mock_model1, mock_model2]

            predictor = EnsemblePredictor(model_configs)
            observation = np.array([1.0, 2.0, 3.0])

            probs, weights = predictor.get_action_probabilities(observation)

            # Expected: weighted average based on entropy confidence
            # [0.1, 0.9] has lower entropy (higher confidence) than [0.3, 0.7]
            # So result should be closer to [0.1, 0.9]
            expected_probs = np.array([[0.191, 0.809]])  # Approximate weighted average
            np.testing.assert_array_almost_equal(probs, expected_probs, decimal=2)

    def test_empty_model_configs(self):
        """Test initialization with empty model configurations."""
        with pytest.raises(ValueError, match="At least one model configuration required"):
            EnsemblePredictor([])

    def test_model_loading_failure(self):
        """Test behavior when all models fail to load."""
        model_configs = [
            {"path": "model1.zip", "weight": 1.0},
            {"path": "model2.zip", "weight": 1.0},
        ]

        with patch("ztb.training.ensemble.PPO.load") as mock_load:
            mock_load.side_effect = [Exception("Load failed"), Exception("Load failed")]

            with pytest.raises(RuntimeError, match="Failed to load any models"):
                EnsemblePredictor(model_configs)


class TestEnsembleTradingSystem:
    """Test cases for EnsembleTradingSystem class."""

    @patch('ztb.training.ensemble.EnsemblePredictor')
    def test_init(self, mock_predictor_class):
        """Test EnsembleTradingSystem initialization."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        config = {
            "ensemble_models": [
                {"path": "model1.zip", "weight": 1.0},
                {"path": "model2.zip", "weight": 2.0},
            ]
        }

        from ztb.training.models.ensemble import EnsembleTradingSystem
        system = EnsembleTradingSystem([
            {"path": "model1.zip", "weight": 1.0},
            {"path": "model2.zip", "weight": 2.0},
        ])

        assert system.ensemble == mock_predictor
        mock_predictor_class.assert_called_once_with([
            {"path": "model1.zip", "weight": 1.0},
            {"path": "model2.zip", "weight": 2.0},
        ])

    @patch('ztb.training.ensemble.EnsemblePredictor')
    def test_trade(self, mock_predictor_class):
        """Test trading execution."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = (np.array([0.1, 0.9]), None)
        mock_predictor.get_action_probabilities.return_value = np.array([0.1, 0.9])
        mock_predictor_class.return_value = mock_predictor

        from ztb.training.models.ensemble import EnsembleTradingSystem
        system = EnsembleTradingSystem([
            {"path": "model1.zip", "weight": 1.0}
        ])

        observation = np.array([1.0, 2.0, 3.0])
        result = system.trade(observation)

        # Verify prediction was called
        mock_predictor.predict.assert_called_once_with(observation, deterministic=True)

        # Verify result structure
        assert "action" in result
        assert "confidence" in result
        assert "risk_check_passed" in result
        assert isinstance(result["action"], int)  # Action should be an integer (0, 1, or 2)
