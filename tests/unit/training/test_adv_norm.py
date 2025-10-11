#!/usr/bin/env python3
"""
Unit tests for Per-Action Advantage Normalization (PAN).

Tests verify that:
1. Minority actions (SELL) are not crushed to zero
2. Gradients flow properly for all actions
3. Statistics are computed correctly
4. Edge cases are handled (empty actions, single sample, etc.)
"""

import numpy as np
import pytest

try:
    from ztb.training.adv_norm import (
        PerActionAdvantageNormalizer,
        normalize_advantages_per_action
    )
except ImportError:
    pytest.skip("ztb.training.adv_norm module not available", allow_module_level=True)


class TestPerActionAdvantageNormalizer:
    """Test suite for PerActionAdvantageNormalizer."""
    
    def test_basic_normalization(self):
        """Test basic three-action normalization."""
        # Synthetic data: HOLD dominant, SELL minority
        advantages = np.array([
            1.0, 1.2, 0.8, 1.1,  # HOLD (4 samples, mean≈1.0)
            0.5, 0.6,            # BUY (2 samples, mean≈0.55)
            -0.3, -0.2           # SELL (2 samples, mean≈-0.25)
        ])
        actions = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        
        normalizer = PerActionAdvantageNormalizer(n_actions=3)
        normalized = normalizer.normalize(advantages, actions)
        
        # Check normalization per action
        for action_idx in range(3):
            mask = (actions == action_idx)
            action_normalized = normalized[mask]
            
            # Mean should be close to 0
            assert abs(action_normalized.mean()) < 0.1, \
                f"Action {action_idx} mean not normalized"
            
            # Std should be close to 1 (if enough samples)
            if np.sum(mask) > 1:
                assert 0.8 < action_normalized.std() < 1.2, \
                    f"Action {action_idx} std not normalized"
    
    def test_minority_action_not_crushed(self):
        """Test that minority action (SELL) maintains scale."""
        # Extreme imbalance: 90% HOLD, 5% BUY, 5% SELL
        n_hold = 90
        n_buy = 5
        n_sell = 5
        
        advantages = np.concatenate([
            np.random.randn(n_hold) + 1.0,   # HOLD: positive bias
            np.random.randn(n_buy) + 0.5,    # BUY: neutral
            np.random.randn(n_sell) - 0.5    # SELL: negative (minority)
        ])
        
        actions = np.concatenate([
            np.zeros(n_hold, dtype=int),
            np.ones(n_buy, dtype=int),
            np.full(n_sell, 2, dtype=int)
        ])
        
        # Traditional normalization
        traditional = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PAN normalization
        normalizer = PerActionAdvantageNormalizer(n_actions=3)
        pan_normalized = normalizer.normalize(advantages, actions)
        
        # Extract SELL advantages
        sell_mask = (actions == 2)
        sell_traditional = traditional[sell_mask]
        sell_pan = pan_normalized[sell_mask]
        
        # PAN should maintain SELL variance better
        assert sell_pan.std() > sell_traditional.std(), \
            "PAN should restore SELL variance"
        
        # PAN SELL should have ~unit variance
        assert 0.8 < sell_pan.std() < 1.2, \
            f"SELL variance should be ~1.0, got {sell_pan.std():.4f}"
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        advantages = np.array([1.0, 1.5, 2.0, 0.5, 0.8, -0.2, -0.1])
        actions = np.array([0, 0, 0, 1, 1, 2, 2])
        
        normalizer = PerActionAdvantageNormalizer(n_actions=3)
        normalizer.normalize(advantages, actions)
        
        stats = normalizer.get_statistics()
        
        # Check counts
        assert stats["action_counts"] == [3, 2, 2], \
            "Action counts mismatch"
        
        # Check means (approximately)
        expected_means = [
            np.mean([1.0, 1.5, 2.0]),  # HOLD
            np.mean([0.5, 0.8]),        # BUY
            np.mean([-0.2, -0.1])       # SELL
        ]
        np.testing.assert_allclose(
            stats["action_means"], expected_means, rtol=1e-4
        )
        
        # Check total
        assert stats["total_samples"] == 7
    
    def test_single_action_sample(self):
        """Test behavior with single sample per action."""
        advantages = np.array([1.0, 0.5, -0.5])
        actions = np.array([0, 1, 2])
        
        normalizer = PerActionAdvantageNormalizer(
            n_actions=3,
            min_samples_per_action=1
        )
        
        # Should not crash
        normalized = normalizer.normalize(advantages, actions)
        
        # With single sample, normalization is trivial (mean centering only)
        assert normalized.shape == advantages.shape
    
    def test_missing_action(self):
        """Test behavior when an action has no samples."""
        advantages = np.array([1.0, 1.2, 0.5, 0.6])
        actions = np.array([0, 0, 1, 1])  # No SELL (action 2)
        
        normalizer = PerActionAdvantageNormalizer(n_actions=3)
        normalized = normalizer.normalize(advantages, actions)
        
        stats = normalizer.get_statistics()
        
        # SELL count should be 0
        assert stats["action_counts"][2] == 0
        
        # HOLD and BUY should still be normalized
        assert 0.8 < normalized[actions == 0].std() < 1.2
        assert 0.8 < normalized[actions == 1].std() < 1.2
    
    def test_convenience_function(self):
        """Test convenience function for one-shot normalization."""
        advantages = np.array([1.0, 1.2, 0.5, 0.6, -0.3, -0.2])
        actions = np.array([0, 0, 1, 1, 2, 2])
        
        normalized = normalize_advantages_per_action(advantages, actions)
        
        # Should produce valid normalization
        assert normalized.shape == advantages.shape
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
    
    def test_gradient_flow_simulation(self):
        """Simulate gradient flow to verify minority action learning."""
        # Synthetic policy gradients scenario
        n_samples = 100
        
        # 70% HOLD, 20% BUY, 10% SELL (imbalanced)
        actions = np.concatenate([
            np.zeros(70, dtype=int),
            np.ones(20, dtype=int),
            np.full(10, 2, dtype=int)
        ])
        
        # Advantages: SELL has strong signal (should learn)
        advantages = np.concatenate([
            np.random.randn(70) * 0.5,     # HOLD: weak signal
            np.random.randn(20) * 0.5,     # BUY: weak signal
            np.random.randn(10) * 2.0 + 3.0  # SELL: STRONG signal (should learn!)
        ])
        
        # Traditional normalization
        traditional = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PAN normalization
        normalizer = PerActionAdvantageNormalizer(n_actions=3)
        pan_normalized = normalizer.normalize(advantages, actions)
        
        # Simulate "effective gradient" (advantage magnitude)
        sell_mask = (actions == 2)
        
        traditional_sell_mag = np.abs(traditional[sell_mask]).mean()
        pan_sell_mag = np.abs(pan_normalized[sell_mask]).mean()
        
        # PAN should preserve SELL gradient magnitude better
        assert pan_sell_mag > traditional_sell_mag * 0.8, \
            "PAN should maintain SELL gradient magnitude"
        
        # PAN SELL should have reasonable magnitude (not crushed near 0)
        assert pan_sell_mag > 0.5, \
            f"SELL gradient too weak: {pan_sell_mag:.4f}"
    
    def test_epsilon_stability(self):
        """Test numerical stability with very small variances."""
        # All advantages nearly identical (very small variance)
        advantages = np.array([1.0, 1.0001, 0.9999, 1.0002])
        actions = np.array([0, 0, 0, 0])
        
        normalizer = PerActionAdvantageNormalizer(n_actions=3, epsilon=1e-8)
        
        # Should not crash or produce NaN/Inf
        normalized = normalizer.normalize(advantages, actions)
        
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
    
    def test_return_statistics_flag(self):
        """Test return_statistics parameter."""
        advantages = np.array([1.0, 1.2, 0.5, -0.3])
        actions = np.array([0, 0, 1, 2])
        
        normalizer = PerActionAdvantageNormalizer(n_actions=3)
        
        # Without statistics
        normalized = normalizer.normalize(advantages, actions, return_statistics=False)
        assert isinstance(normalized, np.ndarray)
        
        # With statistics
        normalized, stats = normalizer.normalize(
            advantages, actions, return_statistics=True
        )
        assert isinstance(normalized, np.ndarray)
        assert isinstance(stats, tuple)
        assert len(stats) == 3  # (means, stds, counts)


def test_comparison_with_traditional():
    """Visual comparison: Traditional vs PAN normalization."""
    print("\n=== Traditional vs PAN Comparison ===")
    
    # Extreme imbalance scenario
    advantages = np.concatenate([
        np.random.randn(80) + 1.0,   # HOLD: 80% (positive bias)
        np.random.randn(15) + 0.3,   # BUY: 15%
        np.random.randn(5) - 0.5     # SELL: 5% (minority, negative)
    ])
    
    actions = np.concatenate([
        np.zeros(80, dtype=int),
        np.ones(15, dtype=int),
        np.full(5, 2, dtype=int)
    ])
    
    # Traditional
    traditional = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PAN
    normalizer = PerActionAdvantageNormalizer(n_actions=3)
    pan_normalized = normalizer.normalize(advantages, actions)
    
    for action_idx, name in enumerate(["HOLD", "BUY", "SELL"]):
        mask = (actions == action_idx)
        
        trad_mean = traditional[mask].mean()
        trad_std = traditional[mask].std()
        
        pan_mean = pan_normalized[mask].mean()
        pan_std = pan_normalized[mask].std()
        
        print(f"\n{name} (n={np.sum(mask)}):")
        print(f"  Traditional: mean={trad_mean:+.4f}, std={trad_std:.4f}")
        print(f"  PAN:         mean={pan_mean:+.4f}, std={pan_std:.4f}")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    test_comparison_with_traditional()
