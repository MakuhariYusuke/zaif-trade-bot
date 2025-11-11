#!/usr/bin/env python3
"""
Stratified Mini-batch Sampler for Action Bias Mitigation.

The stratified sampler ensures balanced representation of minority scenarios
in training batches by:
1. Bucketing transitions by (regime × prev_action)
2. Sampling proportionally from each bucket
3. Ensuring minority scenarios (e.g., down regime + SELL-effective) get exposure

This prevents batch imbalance where majority scenarios dominate gradient updates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

logger = logging.getLogger(__name__)


class StratifiedSampler:
    """
    Stratified sampler that balances regime × prev_action buckets.

    Buckets:
    - Regime: up (price rising), down (falling), sideways (ranging)
    - Prev action: HOLD (0), BUY (1), SELL (2)
    - Total: 3 regimes × 3 actions = 9 buckets

    Sampling strategy:
    - Target: batch_size / n_buckets samples per bucket
    - If bucket has insufficient samples, take all and compensate from others
    - Ensures minority scenarios get proportional representation
    """

    def __init__(
        self,
        n_actions: int = 3,
        regime_window: int = 20,
        regime_threshold: float = 0.001,
        min_samples_per_bucket: int = 1,
    ):
        """
        Initialize stratified sampler.

        Args:
            n_actions: Number of actions (default 3: HOLD, BUY, SELL)
            regime_window: Window for regime classification (price change lookback)
            regime_threshold: Threshold for up/down classification (% change)
            min_samples_per_bucket: Minimum samples to maintain per bucket
        """
        self.n_actions = n_actions
        self.regime_window = regime_window
        self.regime_threshold = regime_threshold
        self.min_samples_per_bucket = min_samples_per_bucket

        # Statistics tracking
        self.bucket_counts = np.zeros((3, n_actions), dtype=int)  # [regime, action]
        self.bucket_samples_drawn = np.zeros((3, n_actions), dtype=int)
        self.total_batches_sampled = 0

    def classify_regime(
        self, prices: NDArray[np.float32], indices: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """
        Classify market regime for given indices.

        Args:
            prices: Price series (close prices)
            indices: Indices to classify

        Returns:
            regime_labels: Array of regime labels (0=up, 1=down, 2=sideways)
        """
        regime_labels = np.zeros(len(indices), dtype=int)

        for i, idx in enumerate(indices):
            if idx < self.regime_window:
                # Insufficient history → default to sideways
                regime_labels[i] = 2
                continue

            # Calculate price change over window
            start_price = prices[idx - self.regime_window]
            end_price = prices[idx]
            pct_change = (end_price - start_price) / start_price

            # Classify regime
            if pct_change > self.regime_threshold:
                regime_labels[i] = 0  # Up
            elif pct_change < -self.regime_threshold:
                regime_labels[i] = 1  # Down
            else:
                regime_labels[i] = 2  # Sideways

        return regime_labels

    def bucket_indices(
        self, regimes: NDArray[np.int64], prev_actions: NDArray[np.int64]
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Bucket transition indices by (regime, prev_action).

        Args:
            regimes: Regime labels (0/1/2 for up/down/sideways)
            prev_actions: Previous action labels (0/1/2 for HOLD/BUY/SELL)

        Returns:
            buckets: Dict[(regime_id, action_id)] -> [indices]
        """
        buckets: Dict[Tuple[int, int], List[int]] = {}

        # Initialize all buckets
        for regime in range(3):
            for action in range(self.n_actions):
                buckets[(regime, action)] = []

        # Assign indices to buckets
        for idx in range(len(regimes)):
            regime = int(regimes[idx])
            action = int(prev_actions[idx])
            buckets[(regime, action)].append(idx)

        # Update statistics
        for regime in range(3):
            for action in range(self.n_actions):
                self.bucket_counts[regime, action] = len(buckets[(regime, action)])

        return buckets

    def sample_batch(
        self,
        prices: NDArray[np.float32],
        prev_actions: NDArray[np.int64],
        batch_size: int,
        available_indices: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.int64]:
        """
        Sample a stratified batch.

        Args:
            prices: Price series for regime classification
            prev_actions: Previous actions for bucketing
            batch_size: Target batch size
            available_indices: If provided, only sample from these indices

        Returns:
            sampled_indices: Stratified batch indices
        """
        # Default to all indices if not specified
        if available_indices is None:
            available_indices = np.arange(len(prev_actions))

        # Classify regimes for available indices
        regimes = self.classify_regime(prices, available_indices)

        # Bucket transitions
        buckets = self.bucket_indices(regimes, prev_actions[available_indices])

        # Calculate target samples per bucket
        n_buckets = 3 * self.n_actions  # 9 buckets
        target_per_bucket = max(1, batch_size // n_buckets)

        sampled_indices = []

        # Sample from each bucket
        for regime in range(3):
            for action in range(self.n_actions):
                bucket_key = (regime, action)
                bucket_indices = buckets[bucket_key]

                if len(bucket_indices) == 0:
                    # Empty bucket → skip
                    continue

                # Sample from bucket (with replacement if insufficient samples)
                n_to_sample = min(target_per_bucket, len(bucket_indices))
                if len(bucket_indices) < target_per_bucket:
                    # Insufficient samples → take all
                    sampled = bucket_indices
                else:
                    # Sufficient samples → random sample
                    sampled = np.random.choice(
                        bucket_indices, size=n_to_sample, replace=False
                    ).tolist()

                sampled_indices.extend(sampled)
                self.bucket_samples_drawn[regime, action] += len(sampled)

        # If batch too small, pad with random samples
        if len(sampled_indices) < batch_size:
            remaining = batch_size - len(sampled_indices)
            # Sample from non-empty buckets
            all_bucket_indices = []
            for bucket_indices in buckets.values():
                all_bucket_indices.extend(bucket_indices)

            if len(all_bucket_indices) > 0:
                padding = np.random.choice(
                    all_bucket_indices,
                    size=min(remaining, len(all_bucket_indices)),
                    replace=True,
                ).tolist()
                sampled_indices.extend(padding)

        # If batch too large, truncate
        if len(sampled_indices) > batch_size:
            sampled_indices = np.random.choice(
                sampled_indices, size=batch_size, replace=False
            ).tolist()

        # Convert to array and map back to original indices
        result = np.array([available_indices[i] for i in sampled_indices])

        self.total_batches_sampled += 1

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about bucket distribution."""
        return {
            "bucket_counts": self.bucket_counts.copy(),
            "bucket_samples_drawn": self.bucket_samples_drawn.copy(),
            "total_batches": self.total_batches_sampled,
            "avg_samples_per_bucket": self.bucket_samples_drawn.sum()
            / max(1, self.total_batches_sampled),
        }

    def reset_statistics(self) -> None:
        """Reset statistics tracking."""
        self.bucket_counts.fill(0)
        self.bucket_samples_drawn.fill(0)
        self.total_batches_sampled = 0


def test_stratified_sampler() -> None:
    """Basic test for StratifiedSampler."""
    print("\n=== Stratified Sampler Basic Test ===\n")

    # Create synthetic data
    # Scenario: 1000 timesteps, price gradually increases (up regime)
    # Majority: HOLD actions, minority: SELL actions
    n_samples = 1000
    prices = np.linspace(100, 150, n_samples)  # Up trend
    prev_actions = np.zeros(n_samples, dtype=int)  # Mostly HOLD

    # Add some minority actions
    prev_actions[200:210] = ACTION_BUY  # Some BUY
    prev_actions[200:210] = ACTION_SELL  # Fewer SELL (minority)

    # Create sampler
    sampler = StratifiedSampler(n_actions=3, regime_window=20, regime_threshold=0.001)

    # Sample batch
    batch_size = 90  # 9 buckets × 10 samples
    batch_indices = sampler.sample_batch(
        prices=prices, prev_actions=prev_actions, batch_size=batch_size
    )

    print(f"Batch size: {len(batch_indices)}")
    print(f"Target batch size: {batch_size}")

    # Analyze sampled batch
    sampled_actions = prev_actions[batch_indices]
    action_counts = np.bincount(sampled_actions, minlength=3)

    print("\nAction distribution in sampled batch:")
    print(f"  HOLD: {action_counts[ACTION_HOLD]}")
    print(f"  BUY:  {action_counts[ACTION_BUY]}")
    print(f"  SELL: {action_counts[ACTION_SELL]}")

    # Get statistics
    stats = sampler.get_statistics()
    print("\nBucket counts (regime × action):")
    print(stats["bucket_counts"])
    print("\nBucket samples drawn:")
    print(stats["bucket_samples_drawn"])

    # Check minority representation
    sell_ratio_original = (prev_actions == ACTION_SELL).sum() / len(prev_actions)
    sell_ratio_batch = action_counts[ACTION_SELL] / len(batch_indices)

    print(f"\nSELL ratio in original data: {sell_ratio_original:.4f}")
    print(f"SELL ratio in sampled batch: {sell_ratio_batch:.4f}")

    if sell_ratio_batch > sell_ratio_original:
        print("✓ Minority action (SELL) is BOOSTED in batch (stratification working!)")
    else:
        print("⚠ Minority action not boosted (may need tuning)")

    print("\n✅ Stratified Sampler basic test complete!")


if __name__ == "__main__":
    test_stratified_sampler()
