"""Tests for inference robustness guards in decode module."""

import numpy as np
import pytest
import warnings

from ztb.inference.decode import decode_action, InferenceConfig


class TestLogitsClipping:
    """Test logits clipping guard."""
    
    def test_extreme_positive_logits(self):
        """Very large positive logits should be clipped."""
        logits = np.array([1000.0, 500.0, 100.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(logits_clip_value=20.0, deterministic=True)
        
        action, info = decode_action(logits, legal_mask, config)
        
        # Should not overflow
        # Note: After clipping to 20.0, all logits become equal (20.0, 20.0, 20.0)
        # So probabilities will be uniform, and deterministic mode will select first (argmax of tie)
        assert np.all(np.isfinite(info["probabilities"]))
        # Probabilities should be approximately uniform
        assert np.allclose(info["probabilities"], [1/3, 1/3, 1/3], atol=0.01)
    
    def test_extreme_negative_logits(self):
        """Very large negative logits should be clipped."""
        logits = np.array([-1000.0, -500.0, -100.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(logits_clip_value=20.0, deterministic=True)
        
        action, info = decode_action(logits, legal_mask, config)
        
        # Should not underflow, should select least negative action
        assert action == 2
        assert np.all(np.isfinite(info["probabilities"]))
    
    def test_mixed_extreme_logits(self):
        """Mix of extreme positive and negative logits."""
        logits = np.array([999.0, -999.0, 0.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(logits_clip_value=20.0, deterministic=True)
        
        action, info = decode_action(logits, legal_mask, config)
        
        # Should handle clipping correctly
        assert action == 0  # Largest after clipping
        assert np.all(np.isfinite(info["probabilities"]))


class TestTemperatureGuards:
    """Test temperature range guards."""
    
    def test_temperature_too_low(self):
        """Temperature below min_temperature should be clamped."""
        logits = np.array([1.0, 0.5, 0.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(
            temperature=0.1,  # Below min (0.5)
            min_temperature=0.5,
            max_temperature=1.5,
            deterministic=True
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action, info = decode_action(logits, legal_mask, config)
            
            # Should warn about clamping
            assert len(w) > 0
            assert "outside safe range" in str(w[0].message).lower()
        
        # Should still work
        assert action == 0
        assert np.all(np.isfinite(info["probabilities"]))
    
    def test_temperature_too_high(self):
        """Temperature above max_temperature should be clamped."""
        logits = np.array([1.0, 0.5, 0.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(
            temperature=2.0,  # Above max (1.5)
            min_temperature=0.5,
            max_temperature=1.5,
            deterministic=True
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action, info = decode_action(logits, legal_mask, config)
            
            # Should warn about clamping
            assert len(w) > 0
            assert "outside safe range" in str(w[0].message).lower()
        
        # Should still work
        assert action == 0
        assert np.all(np.isfinite(info["probabilities"]))


class TestAllIllegalFallback:
    """Test fallback when all actions are illegal."""
    
    def test_all_illegal_single_obs(self):
        """Single observation with all illegal actions."""
        logits = np.array([1.0, 0.5, 0.0])
        legal_mask = np.array([0, 0, 0])  # All illegal
        config = InferenceConfig(fallback_action=0, deterministic=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action, info = decode_action(logits, legal_mask, config)
            
            # Should warn about fallback
            assert len(w) > 0
            assert "no legal actions" in str(w[0].message).lower()
        
        # Should fall back to HOLD (action 0)
        assert action == 0
        assert np.all(np.isfinite(info["probabilities"]))
    
    def test_all_illegal_batch(self):
        """Batch with some all-illegal observations."""
        logits = np.array([
            [1.0, 0.5, 0.0],  # Normal
            [2.0, 1.0, 0.0],  # All illegal
            [0.5, 1.5, 0.0],  # Normal
        ])
        legal_mask = np.array([
            [1, 1, 1],  # All legal
            [0, 0, 0],  # All illegal
            [1, 0, 1],  # Partial
        ])
        config = InferenceConfig(fallback_action=0, deterministic=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actions, info = decode_action(logits, legal_mask, config)
            
            # Should warn about fallback for observation 1
            assert len(w) > 0
            assert "no legal actions" in str(w[0].message).lower()
        
        # Observation 0: normal selection (action 0)
        # Observation 1: fallback to HOLD (action 0)
        # Observation 2: normal selection (action 1)
        assert len(actions) == 3
        assert actions[1] == 0  # Fallback
        assert np.all(np.isfinite(info["probabilities"]))


class TestNaNHandling:
    """Test NaN/Inf detection and recovery."""
    
    def test_nan_logits(self):
        """Logits with NaN values."""
        logits = np.array([np.nan, 0.5, 0.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(deterministic=True)
        
        # After clipping, NaN becomes clipped value
        # Softmax should handle it
        action, info = decode_action(logits, legal_mask, config)
        
        # Should fall back to uniform over legal actions if needed
        assert np.all(np.isfinite(info["probabilities"]))
        assert info["probabilities"].sum() == pytest.approx(1.0)
    
    def test_inf_logits(self):
        """Logits with Inf values."""
        logits = np.array([np.inf, 0.5, 0.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(logits_clip_value=20.0, deterministic=True)
        
        # Should clip to max value
        action, info = decode_action(logits, legal_mask, config)
        
        # Should handle clipping correctly
        assert action == 0  # Inf clipped to +20
        assert np.all(np.isfinite(info["probabilities"]))


class TestRobustnessIntegration:
    """Test multiple guards working together."""
    
    def test_extreme_logits_with_all_illegal(self):
        """Extreme logits combined with all-illegal fallback."""
        logits = np.array([
            [1000.0, -1000.0, 0.0],  # Extreme but legal
            [500.0, -500.0, 0.0],    # Extreme and all illegal
        ])
        legal_mask = np.array([
            [1, 1, 1],  # All legal
            [0, 0, 0],  # All illegal
        ])
        config = InferenceConfig(
            logits_clip_value=20.0,
            fallback_action=0,
            deterministic=True
        )
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            actions, info = decode_action(logits, legal_mask, config)
        
        assert len(actions) == 2
        assert actions[0] == 0  # Normal (clipped, but still max)
        assert actions[1] == 0  # Fallback to HOLD
        assert np.all(np.isfinite(info["probabilities"]))
    
    def test_edge_temperature_with_extreme_logits(self):
        """Edge temperature combined with extreme logits."""
        logits = np.array([999.0, -999.0, 0.0])
        legal_mask = np.array([1, 1, 1])
        config = InferenceConfig(
            temperature=0.1,  # Too low
            min_temperature=0.5,
            logits_clip_value=20.0,
            deterministic=True
        )
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            action, info = decode_action(logits, legal_mask, config)
        
        # Both guards should activate, result should be valid
        assert action == 0
        assert np.all(np.isfinite(info["probabilities"]))
        assert info["probabilities"].sum() == pytest.approx(1.0)
