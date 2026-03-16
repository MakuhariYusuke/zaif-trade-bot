#!/usr/bin/env python3
"""
Unit tests for Stratified Sampler.

Tests verify:
1. Regime classification (up/down/sideways)
2. Bucketing by (regime × prev_action)
3. Balanced sampling across buckets
4. Minority action boosting
5. Edge cases (empty buckets, small batches)
"""

import numpy as np
import pytest

try:
    from ztb.training.stratified_sampler import StratifiedSampler
except ImportError:
    pytest.skip(
        "ztb.training.stratified_sampler module not available", allow_module_level=True
    )
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL


class TestStratifiedSampler:
    """Test suite for StratifiedSampler."""

    def test_initialization(self):
        """Test sampler initialization."""
        sampler = StratifiedSampler(n_actions=3, regime_window=20)

        assert sampler.n_actions == 3
        assert sampler.regime_window == 20
        assert sampler.bucket_counts.shape == (3, 3)  # 3 regimes × 3 actions
        assert sampler.total_batches_sampled == 0

    def test_regime_classification_uptrend(self):
        """Test regime classification for uptrend."""
        sampler = StratifiedSampler(regime_window=10, regime_threshold=0.01)

        # Uptrend: prices increasing
        prices = np.linspace(100, 120, 50)  # +20% over 50 steps
        indices = np.arange(20, 40)  # Sufficient history

        regimes = sampler.classify_regime(prices, indices)

        # Should be mostly "up" (0)
        assert (regimes == 0).sum() > len(regimes) * 0.8, "Most should be uptrend"

    def test_regime_classification_downtrend(self):
        """Test regime classification for downtrend."""
        sampler = StratifiedSampler(regime_window=10, regime_threshold=0.01)

        # Downtrend: prices decreasing
        prices = np.linspace(120, 100, 50)  # -20% over 50 steps
        indices = np.arange(20, 40)

        regimes = sampler.classify_regime(prices, indices)

        # Should be mostly "down" (1)
        assert (regimes == 1).sum() > len(regimes) * 0.8, "Most should be downtrend"

    def test_regime_classification_sideways(self):
        """Test regime classification for sideways."""
        sampler = StratifiedSampler(regime_window=10, regime_threshold=0.01)

        # Sideways: prices oscillating around mean
        prices = 100 + np.sin(np.linspace(0, 4 * np.pi, 50)) * 0.5  # ±0.5% oscillation
        indices = np.arange(20, 40)

        regimes = sampler.classify_regime(prices, indices)

        # Should be mostly "sideways" (2)
        assert (regimes == 2).sum() > len(regimes) * 0.5, "Many should be sideways"

    def test_bucket_indices(self):
        """Test bucketing by (regime, prev_action)."""
        sampler = StratifiedSampler(n_actions=3)

        # Create synthetic data
        regimes = np.array(
            [
                ACTION_HOLD,
                ACTION_HOLD,
                ACTION_BUY,
                ACTION_BUY,
                ACTION_SELL,
                ACTION_SELL,
                ACTION_HOLD,
                ACTION_HOLD,
                ACTION_SELL,
            ]
        )  # 3 of each regime
        prev_actions = np.array(
            [
                ACTION_HOLD,
                ACTION_BUY,
                ACTION_SELL,
                ACTION_HOLD,
                ACTION_BUY,
                ACTION_SELL,
                ACTION_HOLD,
                ACTION_BUY,
                ACTION_SELL,
            ]
        )  # Cyclic actions

        buckets = sampler.bucket_indices(regimes, prev_actions)

        # Check bucket structure
        assert len(buckets) == 9, "Should have 9 buckets (3×3)"

        # Check specific bucket
        assert 0 in buckets[(0, 0)], "Index 0 should be in (up, HOLD)"
        assert 1 in buckets[(0, 1)], "Index 1 should be in (up, BUY)"
        assert 2 in buckets[(1, 2)], "Index 2 should be in (down, SELL)"

    def test_sample_batch_basic(self):
        """Test basic batch sampling."""
        sampler = StratifiedSampler(n_actions=3, regime_window=10)

        # Create data
        n_samples = 1000
        prices = np.linspace(100, 120, n_samples)  # Uptrend
        prev_actions = np.random.randint(0, 3, size=n_samples)

        # Sample batch
        batch_size = 90
        batch_indices = sampler.sample_batch(prices, prev_actions, batch_size)

        # Check batch size
        assert (
            len(batch_indices) == batch_size
        ), f"Batch size mismatch: {len(batch_indices)} vs {batch_size}"

        # Check indices are valid
        assert np.all(batch_indices >= 0)
        assert np.all(batch_indices < n_samples)

    def test_minority_boosting(self):
        """Test that minority actions are boosted in sampled batch."""
        sampler = StratifiedSampler(n_actions=3, regime_window=10)

        # Create imbalanced data
        n_samples = 1000
        prices = np.linspace(100, 120, n_samples)
        prev_actions = np.zeros(n_samples, dtype=int)  # Majority HOLD

        # Add minority actions
        prev_actions[100:110] = ACTION_BUY  # 1% BUY
        prev_actions[200:205] = ACTION_SELL  # 0.5% SELL (extreme minority)

        # Sample batch
        batch_size = 90
        batch_indices = sampler.sample_batch(prices, prev_actions, batch_size)

        # Analyze batch
        sampled_actions = prev_actions[batch_indices]
        action_counts = np.bincount(sampled_actions, minlength=3)

        # Original ratios
        original_sell_ratio = (prev_actions == 2).sum() / len(prev_actions)
        batch_sell_ratio = action_counts[2] / len(batch_indices)

        # Minority should be boosted
        assert (
            batch_sell_ratio > original_sell_ratio
        ), f"SELL should be boosted: {batch_sell_ratio:.4f} vs {original_sell_ratio:.4f}"

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        sampler = StratifiedSampler(n_actions=3, regime_window=10)

        # Create data
        n_samples = 500
        prices = np.linspace(100, 110, n_samples)
        prev_actions = np.random.randint(0, 3, size=n_samples)

        # Sample multiple batches
        for _ in range(5):
            sampler.sample_batch(prices, prev_actions, batch_size=90)

        # Check statistics
        stats = sampler.get_statistics()

        assert stats["total_batches"] == 5
        assert stats["bucket_counts"].shape == (3, 3)
        assert stats["bucket_samples_drawn"].sum() > 0

    def test_empty_bucket_handling(self):
        """Test handling of empty buckets."""
        sampler = StratifiedSampler(n_actions=3, regime_window=10)

        # Create data with missing scenarios
        n_samples = 100
        prices = np.linspace(100, 120, n_samples)  # Only uptrend
        prev_actions = np.zeros(n_samples, dtype=int)  # Only HOLD

        # Sample batch (should not crash despite missing buckets)
        batch_indices = sampler.sample_batch(prices, prev_actions, batch_size=30)

        assert len(batch_indices) > 0, "Should sample despite empty buckets"

    def test_small_batch_size(self):
        """Test sampling with small batch size."""
        sampler = StratifiedSampler(n_actions=3, regime_window=10)

        n_samples = 100
        prices = np.linspace(100, 110, n_samples)
        prev_actions = np.random.randint(0, 3, size=n_samples)

        # Small batch
        batch_size = 5
        batch_indices = sampler.sample_batch(prices, prev_actions, batch_size)

        # Should still work
        assert len(batch_indices) <= batch_size
        assert len(batch_indices) > 0

    def test_reset_statistics(self):
        """Test statistics reset."""
        sampler = StratifiedSampler(n_actions=3, regime_window=10)

        # Sample batch
        n_samples = 100
        prices = np.linspace(100, 110, n_samples)
        prev_actions = np.random.randint(0, 3, size=n_samples)
        sampler.sample_batch(prices, prev_actions, batch_size=30)

        # Reset
        sampler.reset_statistics()

        # Check reset
        assert sampler.total_batches_sampled == 0
        assert sampler.bucket_counts.sum() == 0
        assert sampler.bucket_samples_drawn.sum() == 0


def test_stratified_sampler_visual():
    """Visual test comparing stratified vs uniform sampling."""
    print("\n=== Stratified vs Uniform Sampling Comparison ===\n")

    # Create imbalanced dataset
    n_samples = 1000
    prices = np.linspace(100, 150, n_samples)  # Uptrend
    prev_actions = np.zeros(n_samples, dtype=int)

    # Add minority actions
    prev_actions[100:150] = ACTION_BUY  # 5% BUY
    prev_actions[200:220] = ACTION_SELL  # 2% SELL (minority)

    original_counts = np.bincount(prev_actions, minlength=3)
    print("Original data distribution:")
    print(
        f"  HOLD: {original_counts[ACTION_HOLD]} ({original_counts[ACTION_HOLD]/n_samples*100:.1f}%)"
    )
    print(
        f"  BUY:  {original_counts[ACTION_BUY]} ({original_counts[ACTION_BUY]/n_samples*100:.1f}%)"
    )
    print(
        f"  SELL: {original_counts[ACTION_SELL]} ({original_counts[ACTION_SELL]/n_samples*100:.1f}%)"
    )

    # Uniform sampling
    batch_size = 90
    uniform_indices = np.random.choice(n_samples, size=batch_size, replace=False)
    uniform_actions = prev_actions[uniform_indices]
    uniform_counts = np.bincount(uniform_actions, minlength=3)

    print("\nUniform sampling:")
    print(
        f"  HOLD: {uniform_counts[ACTION_HOLD]} ({uniform_counts[ACTION_HOLD]/batch_size*100:.1f}%)"
    )
    print(
        f"  BUY:  {uniform_counts[ACTION_BUY]} ({uniform_counts[ACTION_BUY]/batch_size*100:.1f}%)"
    )
    print(
        f"  SELL: {uniform_counts[ACTION_SELL]} ({uniform_counts[ACTION_SELL]/batch_size*100:.1f}%)"
    )

    # Stratified sampling
    sampler = StratifiedSampler(n_actions=3, regime_window=20)
    stratified_indices = sampler.sample_batch(prices, prev_actions, batch_size)
    stratified_actions = prev_actions[stratified_indices]
    stratified_counts = np.bincount(stratified_actions, minlength=3)

    print("\nStratified sampling:")
    print(
        f"  HOLD: {stratified_counts[ACTION_HOLD]} ({stratified_counts[ACTION_HOLD]/batch_size*100:.1f}%)"
    )
    print(
        f"  BUY:  {stratified_counts[ACTION_BUY]} ({stratified_counts[ACTION_BUY]/batch_size*100:.1f}%)"
    )
    print(
        f"  SELL: {stratified_counts[ACTION_SELL]} ({stratified_counts[ACTION_SELL]/batch_size*100:.1f}%)"
    )

    # Check minority boosting
    sell_boost = stratified_counts[2] / max(1, uniform_counts[2])
    print(f"\nSELL boost factor: {sell_boost:.2f}x")

    if sell_boost > 1.5:
        print("✓ Strong minority boosting detected!")
    elif sell_boost > 1.0:
        print("✓ Moderate minority boosting detected")
    else:
        print("⚠ No significant boosting")

    print("\n✅ Visual comparison complete!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    test_stratified_sampler_visual()
