"""
Action Imbalance Weights for Training.

Computes inverse frequency weights to handle action imbalance,
with beta clipping, EMA smoothing, and safety guards.
"""

import numpy as np

from ztb.trading.environment.constants import EPSILON

class ActionWeightCalculator:
    """
    Calculate and manage action imbalance weights.

    Computes w_a = min(1/freq(a), beta) with:
    - Beta clipping to prevent extreme ratios
    - EMA smoothing for stability
    - Normalization to sum=3 (average=1.0)
    - Safety guards (entropy, KL divergence monitoring)
    """

    def __init__(
        self,
        beta: float = 3.0,
        ema_alpha: float = 0.1,
        epsilon: float = EPSILON,
        entropy_min: float = 0.05,
        target_kl_max: float = 0.03,
        kl_consecutive_max: int = 3,
    ):
        """
        Initialize weight calculator.

        Args:
            beta: Maximum weight (clips to prevent extreme ratios)
            ema_alpha: EMA coefficient for smoothing (0.1 = slow)
            epsilon: Small value to avoid division by zero
            entropy_min: Minimum entropy threshold (safety guard)
            target_kl_max: Maximum target_kl violation rate
            kl_consecutive_max: Maximum consecutive KL violations
        """
        self.beta = beta
        self.ema_alpha = ema_alpha
        self.epsilon = epsilon
        self.entropy_min = entropy_min
        self.target_kl_max = target_kl_max
        self.kl_consecutive_max = kl_consecutive_max

        # Internal state
        self._smoothed_counts: dict[str, float] | None = None
        self._kl_consecutive_violations = 0
        self._weights_active = True

    def compute_weights(
        self,
        action_counts: dict[str, int],
        apply_ema: bool = True,
    ) -> dict[str, float]:
        """
        Compute action weights from counts.

        Args:
            action_counts: Dictionary with action counts (HOLD, BUY, SELL)
            apply_ema: Whether to apply EMA smoothing

        Returns:
            Dictionary with normalized weights (sum=3, average=1.0)
        """
        # Extract counts
        total = sum(action_counts.values())

        if total == 0:
            return {"HOLD": 1.0, "BUY": 1.0, "SELL": 1.0}

        # Apply EMA smoothing if enabled
        if apply_ema and self._smoothed_counts is not None:
            smoothed = self._apply_ema(action_counts)
        else:
            smoothed = {k: float(v) for k, v in action_counts.items()}
            self._smoothed_counts = smoothed

        # Compute frequencies
        total_smoothed = sum(smoothed.values())
        frequencies = {
            action: max(count / total_smoothed, self.epsilon)
            for action, count in smoothed.items()
        }

        # Compute raw weights (inverse frequency)
        weights_raw = {action: 1.0 / freq for action, freq in frequencies.items()}

        # Clip to beta
        weights_clipped = {
            action: min(weight, self.beta) for action, weight in weights_raw.items()
        }

        # Normalize to sum=3 (average=1.0 for 3 actions)
        weight_sum = sum(weights_clipped.values())
        weights_normalized = {
            action: weight * 3.0 / weight_sum
            for action, weight in weights_clipped.items()
        }

        return weights_normalized

    def _apply_ema(self, current_counts: dict[str, int]) -> dict[str, float]:
        """
        Apply EMA smoothing to action counts.

        Args:
            current_counts: Current observation counts

        Returns:
            Smoothed counts
        """
        if self._smoothed_counts is None:
            self._smoothed_counts = {k: float(v) for k, v in current_counts.items()}
            return self._smoothed_counts

        smoothed = {}
        for action in current_counts:
            prev = self._smoothed_counts.get(action, 0.0)
            curr = float(current_counts[action])
            smoothed[action] = self.ema_alpha * curr + (1 - self.ema_alpha) * prev

        self._smoothed_counts = smoothed
        return smoothed

    def check_safety_guards(
        self,
        entropy: float,
        kl_violations_rate: float,
    ) -> tuple[bool, str]:
        """
        Check safety guards and determine if weights should be disabled.

        Args:
            entropy: Current moving entropy
            kl_violations_rate: Rate of target_kl violations (e.g., 0.02 = 2%)

        Returns:
            tuple of (should_revert, reason)
            - should_revert: True if weights should revert to 1.0
            - reason: Human-readable reason for revert
        """
        # Check entropy
        if entropy < self.entropy_min:
            self._weights_active = False
            return True, f"Entropy too low: {entropy:.4f} < {self.entropy_min}"

        # Check KL violations
        if kl_violations_rate > self.target_kl_max:
            self._kl_consecutive_violations += 1

            if self._kl_consecutive_violations >= self.kl_consecutive_max:
                self._weights_active = False
                return (
                    True,
                    f"KL violations too high: {kl_violations_rate:.1%} > {self.target_kl_max:.1%} for {self._kl_consecutive_violations} updates",
                )
        else:
            # Reset counter on success
            self._kl_consecutive_violations = 0

        return False, ""

    def reset_guards(self) -> None:
        """Reset safety guard state (e.g., after intervention)."""
        self._kl_consecutive_violations = 0
        self._weights_active = True

    def get_safe_weights(
        self,
        action_counts: dict[str, int],
        entropy: float,
        kl_violations_rate: float,
        apply_ema: bool = True,
    ) -> tuple[dict[str, float], bool, str]:
        """
        Get weights with safety guard checks.

        Args:
            action_counts: Action counts
            entropy: Current moving entropy
            kl_violations_rate: KL violation rate
            apply_ema: Whether to apply EMA smoothing

        Returns:
            tuple of (weights, guard_triggered, reason)
            - weights: Either computed weights or [1.0, 1.0, 1.0] if guard triggered
            - guard_triggered: True if safety guard was triggered
            - reason: Human-readable reason for guard trigger
        """
        # Check guards
        should_revert, reason = self.check_safety_guards(entropy, kl_violations_rate)

        if should_revert or not self._weights_active:
            # Revert to uniform weights
            return {"HOLD": 1.0, "BUY": 1.0, "SELL": 1.0}, True, reason

        # Compute normal weights
        weights = self.compute_weights(action_counts, apply_ema=apply_ema)
        return weights, False, ""

def compute_action_weights(
    action_counts: dict[str, int],
    beta: float = 3.0,
    epsilon: float = EPSILON,
) -> dict[str, float]:
    """
    Compute inverse frequency weights (stateless version).

    Args:
        action_counts: Dictionary with action counts (HOLD, BUY, SELL)
        beta: Maximum weight (clips to prevent extreme ratios)
        epsilon: Small value to avoid division by zero

    Returns:
        Dictionary with normalized weights (sum=3, average=1.0)
    """
    calculator = ActionWeightCalculator(beta=beta, epsilon=epsilon)
    return calculator.compute_weights(action_counts, apply_ema=False)

def cosine_warmup_schedule(
    current_step: int,
    warmup_start: int = 5000,
    warmup_end: int = 15000,
) -> float:
    """
    Cosine warmup schedule for weights.

    Args:
        current_step: Current training step
        warmup_start: Step to start applying weights (before this: weight=1.0)
        warmup_end: Step to reach full weights

    Returns:
        Weight multiplier in [0, 1]
    """
    if current_step < warmup_start:
        return 0.0  # No weighting (w=1.0)

    if current_step >= warmup_end:
        return 1.0  # Full weighting

    # Cosine interpolation
    progress = (current_step - warmup_start) / (warmup_end - warmup_start)
    return float(0.5 * (1.0 - np.cos(np.pi * progress)))
