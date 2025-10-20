#!/usr/bin/env python3
"""
Per-Action Advantage Normalization (PAN).

This module implements action-specific advantage normalization to prevent
minority actions (e.g., SELL) from having their gradients crushed by
the majority actions (e.g., HOLD/BUY).

Key concept:
- Traditional PPO: A' = (A - mean(A)) / std(A)  [全サンプル一括]
- PAN: For each action a ∈ {HOLD, BUY, SELL}:
       A_a' = (A_a - mean(A_a)) / std(A_a)  [アクション別]
       Then recombine all normalized advantages.

Expected effect: Restores the relative scale of minority actions,
allowing their gradients to flow properly during backpropagation.
"""

import logging
from typing import Any, Dict, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

logger = logging.getLogger(__name__)


class PerActionAdvantageNormalizer:
    """
    Normalizes advantages separately for each action.

    This prevents majority actions from dominating the normalization
    statistics and crushing minority action gradients.
    """

    def __init__(
        self, n_actions: int = 3, epsilon: float = 1e-8, min_samples_per_action: int = 1
    ):
        """
        Initialize normalizer.

        Args:
            n_actions: Number of discrete actions (default: 3 for HOLD/BUY/SELL)
            epsilon: Small constant for numerical stability
            min_samples_per_action: Minimum samples per action for normalization
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.min_samples_per_action = min_samples_per_action

        # Statistics tracking
        self.reset_statistics()

    def reset_statistics(self) -> None:
        """Reset statistics for new training iteration."""
        self.action_counts = np.zeros(self.n_actions, dtype=np.int32)
        self.action_means = np.zeros(self.n_actions, dtype=np.float32)
        self.action_stds = np.zeros(self.n_actions, dtype=np.float32)

    def normalize(
        self,
        advantages: NDArray[np.float32],
        actions: NDArray[np.int64],
        return_statistics: bool = False,
    ) -> Union[
        NDArray[np.floating[Any]],
        Tuple[
            NDArray[np.floating[Any]],
            Tuple[
                NDArray[np.floating[Any]],
                NDArray[np.floating[Any]],
                NDArray[np.integer[Any]],
            ],
        ],
    ]:
        """
        Normalize advantages per action.

        Args:
            advantages: Advantage values [batch_size]
            actions: Action indices [batch_size]
            return_statistics: If True, also return normalization statistics

        Returns:
            Normalized advantages [batch_size]
            If return_statistics=True, also returns (means, stds, counts)
        """
        advantages = np.asarray(advantages, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)

        if len(advantages) != len(actions):
            raise ValueError(
                f"Advantages and actions must have same length, "
                f"got {len(advantages)} and {len(actions)}"
            )

        # Initialize normalized advantages with original values
        normalized = advantages.copy()

        # Reset statistics for this batch
        self.reset_statistics()

        # Normalize each action separately
        for action_idx in range(self.n_actions):
            # Get mask for this action
            mask = actions == action_idx
            n_samples = np.sum(mask)

            self.action_counts[action_idx] = n_samples

            # Skip if insufficient samples
            if n_samples < self.min_samples_per_action:
                logger.warning(
                    f"Action {action_idx} has only {n_samples} samples "
                    f"(min: {self.min_samples_per_action}). Skipping normalization."
                )
                continue

            # Extract advantages for this action
            action_advantages = advantages[mask]

            # Compute statistics
            mean_val = np.mean(action_advantages)
            std_val = np.std(action_advantages)

            self.action_means[action_idx] = mean_val
            self.action_stds[action_idx] = std_val

            # Normalize (with epsilon for stability)
            normalized[mask] = (action_advantages - mean_val) / (std_val + self.epsilon)

            logger.debug(
                f"Action {action_idx}: n={n_samples}, "
                f"mean={mean_val:.4f}, std={std_val:.4f}"
            )

        if return_statistics:
            return normalized, (self.action_means, self.action_stds, self.action_counts)

        return normalized

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current normalization statistics.

        Returns:
            Dictionary with per-action statistics
        """
        return {
            "action_counts": self.action_counts.tolist(),
            "action_means": self.action_means.tolist(),
            "action_stds": self.action_stds.tolist(),
            "total_samples": int(np.sum(self.action_counts)),
        }


def normalize_advantages_per_action(
    advantages: NDArray[np.float32],
    actions: NDArray[np.int64],
    n_actions: int = 3,
    epsilon: float = 1e-8,
) -> NDArray[np.floating[Any]]:
    """
    Convenience function for one-shot advantage normalization.

    Args:
        advantages: Advantage values [batch_size]
        actions: Action indices [batch_size]
        n_actions: Number of discrete actions
        epsilon: Numerical stability constant

    Returns:
        Normalized advantages [batch_size]

    Example:
        >>> advantages = np.array([1.0, 2.0, -0.5, 0.3])
        >>> actions = np.array([0, 0, 2, 2])  # Two HOLD, two SELL
        >>> normalized = normalize_advantages_per_action(advantages, actions)
    """
    normalizer = PerActionAdvantageNormalizer(n_actions=n_actions, epsilon=epsilon)
    return cast(
        NDArray[np.floating[Any]],
        normalizer.normalize(advantages, actions, return_statistics=False),
    )


def test_pan_basic() -> None:
    """Test basic PAN functionality."""
    print("Testing Per-Action Advantage Normalization...")

    # Synthetic data: HOLD dominant, SELL minority
    advantages = np.array(
        [
            1.0,
            1.2,
            0.8,
            1.1,  # HOLD (4 samples)
            0.5,
            0.6,  # BUY (2 samples)
            -0.3,
            -0.2,  # SELL (2 samples) - minority with negative values
        ]
    )
    actions = np.array([0, 0, 0, 0, 1, 1, 2, 2])

    # Traditional normalization (would crush SELL)
    traditional = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PAN normalization
    normalizer = PerActionAdvantageNormalizer(n_actions=3)
    pan_normalized, stats = normalizer.normalize(
        advantages, actions, return_statistics=True
    )

    print("\n=== Original Advantages ===")
    print(f"HOLD: {advantages[actions == ACTION_HOLD]}")
    print(f"BUY:  {advantages[actions == ACTION_BUY]}")
    print(f"SELL: {advantages[actions == ACTION_SELL]}")

    print("\n=== Traditional Normalization ===")
    print(f"HOLD: {traditional[actions == ACTION_HOLD]}")
    print(f"BUY:  {traditional[actions == ACTION_BUY]}")
    print(f"SELL: {traditional[actions == ACTION_SELL]}")

    print("\n=== PAN Normalization ===")
    print(f"HOLD: {pan_normalized[actions == ACTION_HOLD]}")
    print(f"BUY:  {pan_normalized[actions == ACTION_BUY]}")
    print(f"SELL: {pan_normalized[actions == ACTION_SELL]}")

    print("\n=== PAN Statistics ===")
    means, stds, counts = stats
    for i in range(3):
        action_name = ["HOLD", "BUY", "SELL"][i]
        print(f"{action_name}: n={counts[i]}, mean={means[i]:.4f}, std={stds[i]:.4f}")

    # Verify SELL advantages are not crushed
    sell_pan_std = np.std(pan_normalized[actions == ACTION_SELL])
    print(f"\n✓ SELL std after PAN: {sell_pan_std:.4f}")
    print("  (Should be ~1.0 due to normalization)")

    assert sell_pan_std > 0.5, "SELL gradient should not be crushed!"

    print("\n✅ PAN basic test passed!")


if __name__ == "__main__":
    test_pan_basic()
