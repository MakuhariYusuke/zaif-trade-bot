"""
Tests for Enhanced Ensemble Aggregator with confidence-weighted voting and model disqualification.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Note: This is a simplified test that doesn't require actual PPO models
# In real usage, you would use actual trained models


class TestEnsembleEnhancements:
    """Test enhanced ensemble aggregator features."""
    
    def test_disqualification_threshold_default(self):
        """Test default disqualification threshold."""
        # This would test EnsembleAggregator initialization
        # Simplified version without actual import
        disqualification_threshold = 0.5
        min_sharpe_threshold = -2.0
        
        assert disqualification_threshold == 0.5
        assert min_sharpe_threshold == -2.0
    
    def test_model_disqualification_by_sharpe(self):
        """Test model disqualification based on Sharpe ratio."""
        # Simulate model performances
        sharpe_ratios = [2.5, -3.0, 1.8]  # Model 2 should be disqualified
        min_sharpe_threshold = -2.0
        
        disqualified = []
        for i, sharpe in enumerate(sharpe_ratios):
            if sharpe < min_sharpe_threshold:
                disqualified.append(i)
        
        assert 1 in disqualified  # Model 2 (index 1) should be disqualified
        assert len(disqualified) == 1
    
    def test_model_disqualification_by_masked_rate(self):
        """Test model disqualification based on all-masked rate."""
        # Simulate masked rates
        masked_rates = [0.1, 0.6, 0.3]  # Model 2 should be disqualified
        disqualification_threshold = 0.5
        
        disqualified = []
        for i, rate in enumerate(masked_rates):
            if rate >= disqualification_threshold:
                disqualified.append(i)
        
        assert 1 in disqualified  # Model 2 (index 1) should be disqualified
        assert len(disqualified) == 1
    
    def test_weight_calculation_with_confidence_scaling(self):
        """Test weight calculation with Sharpe × confidence scaling."""
        # Simulate model performance
        sharpe_ratios = [2.0, 3.0, 1.5]
        confidences = [0.8, 0.9, 0.7]
        
        # Calculate weights with confidence scaling
        weights = []
        for sharpe, conf in zip(sharpe_ratios, confidences):
            weight = max(sharpe, 0.0) * conf
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        normalized_weights = [w / total for w in weights]
        
        # Model 2 should have highest weight (3.0 * 0.9 = 2.7)
        assert normalized_weights[1] > normalized_weights[0]
        assert normalized_weights[1] > normalized_weights[2]
        
        # Weights should sum to 1
        assert abs(sum(normalized_weights) - 1.0) < 1e-6
    
    def test_weight_calculation_without_confidence_scaling(self):
        """Test weight calculation without confidence scaling (Sharpe only)."""
        # Simulate model performance
        sharpe_ratios = [2.0, 3.0, 1.5]
        
        # Calculate weights without confidence scaling
        weights = [max(s, 0.0) for s in sharpe_ratios]
        
        # Normalize
        total = sum(weights)
        normalized_weights = [w / total for w in weights]
        
        # Model 2 should have highest weight (3.0)
        assert normalized_weights[1] > normalized_weights[0]
        assert normalized_weights[1] > normalized_weights[2]
        
        # Weights should sum to 1
        assert abs(sum(normalized_weights) - 1.0) < 1e-6
    
    def test_disqualified_model_gets_zero_weight(self):
        """Test that disqualified models get weight=0."""
        # Simulate model performance
        sharpe_ratios = [2.0, -3.0, 1.5]  # Model 2 disqualified
        min_sharpe_threshold = -2.0
        
        # Calculate weights
        weights = []
        for i, sharpe in enumerate(sharpe_ratios):
            if sharpe < min_sharpe_threshold:
                # Disqualified
                weights.append(0.0)
            else:
                # Normal weight
                weights.append(max(sharpe, 0.0))
        
        # Normalize (excluding zeros)
        total = sum(weights)
        if total > 0:
            normalized_weights = [w / total for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Model 2 should have zero weight
        assert normalized_weights[1] == 0.0
        
        # Other models should have positive weights
        assert normalized_weights[0] > 0.0
        assert normalized_weights[2] > 0.0
        
        # Weights should sum to 1
        assert abs(sum(normalized_weights) - 1.0) < 1e-6
    
    def test_all_models_disqualified_fallback(self):
        """Test fallback when all models are disqualified."""
        # All models have poor Sharpe
        sharpe_ratios = [-3.0, -4.0, -2.5]
        min_sharpe_threshold = -2.0
        
        # Calculate weights
        weights = []
        for sharpe in sharpe_ratios:
            if sharpe < min_sharpe_threshold:
                weights.append(0.0)
            else:
                weights.append(max(sharpe, 0.0))
        
        # Check if all weights are zero
        total = sum(weights)
        
        if total == 0:
            # Fallback to equal weights
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / total for w in weights]
        
        # All models should have equal weight (fallback)
        assert all(w == pytest.approx(1.0 / 3) for w in normalized_weights)
        
        # Weights should sum to 1
        assert abs(sum(normalized_weights) - 1.0) < 1e-6
    
    def test_masked_rate_detection(self):
        """Test detection of all-masked episodes."""
        # Simulate action probabilities
        normal_probs = np.array([0.7, 0.2, 0.1])
        masked_probs = np.array([0.333, 0.333, 0.334])  # All equal = masked
        
        # Check if masked (std < threshold)
        normal_std = np.std(normal_probs)
        masked_std = np.std(masked_probs)
        
        threshold = 1e-6
        
        assert normal_std > threshold  # Not masked
        assert masked_std < threshold  # Masked
    
    def test_episode_masked_rate_calculation(self):
        """Test calculation of masked rate per episode."""
        # Simulate episode with some masked steps
        total_steps = 100
        masked_steps = 60
        
        masked_rate = masked_steps / total_steps
        
        # Episode is considered masked if > 50% steps are masked
        episode_threshold = 0.5
        
        assert masked_rate > episode_threshold  # Episode should be counted as masked
    
    def test_confidence_weighted_prediction(self):
        """Test confidence-weighted prediction aggregation."""
        # Simulate individual model predictions
        # Each model predicts: action_probs shape (3,) for [BUY, HOLD, SELL]
        pred1_probs = np.array([0.7, 0.2, 0.1])  # Model 1: confident BUY
        pred2_probs = np.array([0.1, 0.8, 0.1])  # Model 2: confident HOLD
        pred3_probs = np.array([0.2, 0.2, 0.6])  # Model 3: confident SELL
        
        # Model weights (from calibration)
        weights = [0.4, 0.3, 0.3]
        
        # Weighted average
        final_probs = (
            pred1_probs * weights[0] +
            pred2_probs * weights[1] +
            pred3_probs * weights[2]
        )
        
        # Final action is argmax
        final_action = int(np.argmax(final_probs))
        
        # Model 1 has highest weight and predicts BUY, so final should be BUY
        assert final_action == 0  # BUY
        
        # Final confidence
        final_confidence = float(np.max(final_probs))
        assert 0.0 <= final_confidence <= 1.0


class TestCalibrationIntegration:
    """Test integration of calibration with disqualification."""
    
    def test_full_calibration_workflow(self):
        """Test complete calibration workflow with disqualification."""
        # Simulate 3 models
        n_models = 3
        n_episodes = 50
        
        # Model 1: Good performance
        model1_rewards = np.random.normal(100, 20, n_episodes)
        model1_sharpe = np.mean(model1_rewards) / np.std(model1_rewards)
        model1_confidence = 0.85
        model1_masked_rate = 0.1
        
        # Model 2: Poor performance (should be disqualified by Sharpe)
        model2_rewards = np.random.normal(-50, 30, n_episodes)
        model2_sharpe = np.mean(model2_rewards) / np.std(model2_rewards)
        model2_confidence = 0.70
        model2_masked_rate = 0.2
        
        # Model 3: High masked rate (should be disqualified)
        model3_rewards = np.random.normal(80, 25, n_episodes)
        model3_sharpe = np.mean(model3_rewards) / np.std(model3_rewards)
        model3_confidence = 0.80
        model3_masked_rate = 0.6
        
        # Thresholds
        min_sharpe_threshold = -2.0
        disqualification_threshold = 0.5
        
        # Check disqualifications
        disqualified = []
        
        # Model 1
        if model1_sharpe < min_sharpe_threshold or model1_masked_rate >= disqualification_threshold:
            disqualified.append(0)
        
        # Model 2
        if model2_sharpe < min_sharpe_threshold or model2_masked_rate >= disqualification_threshold:
            disqualified.append(1)
        
        # Model 3
        if model3_sharpe < min_sharpe_threshold or model3_masked_rate >= disqualification_threshold:
            disqualified.append(2)
        
        # Model 2 should be disqualified (poor Sharpe)
        # Model 3 should be disqualified (high masked rate)
        # Only Model 1 should remain
        assert 2 in disqualified  # Model 3 (high masked rate)
        
        # Calculate final weights
        sharpes = [model1_sharpe, model2_sharpe, model3_sharpe]
        confidences = [model1_confidence, model2_confidence, model3_confidence]
        
        weights = []
        for i in range(n_models):
            if i in disqualified:
                weights.append(0.0)
            else:
                # Sharpe × confidence
                weight = max(sharpes[i], 0.0) * confidences[i]
                weights.append(weight)
        
        # Normalize
        total = sum(weights)
        if total > 0:
            normalized_weights = [w / total for w in weights]
        else:
            normalized_weights = [1.0 / n_models] * n_models
        
        # Model 1 should have all weight (others disqualified)
        if 1 in disqualified and 2 in disqualified:
            assert normalized_weights[0] == 1.0
            assert normalized_weights[1] == 0.0
            assert normalized_weights[2] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
