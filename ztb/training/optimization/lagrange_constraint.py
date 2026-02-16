"""
Lagrange Constraint for Target Action Rate.

Adds explicit constraint to maintain target action rate during training:
L(θ) = L_PPO(θ) - λ * max(0, |r_target - r_actual(θ)| - tolerance)

Where:
- r_target: Target action rate (e.g., 0.33 = 33% for balanced)
- r_actual: Current batch action rate (legal steps only)
- λ: Lagrange multiplier (dual variable)
- tolerance: Allowed deviation before penalty

Dual update: λ ← clip(λ + η * (|r_target - r_actual| - tolerance), 0, λ_max)
"""

from collections import deque
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

ActionType = Literal["HOLD", "BUY", "SELL"]


class LagrangeConstraint:
    """
    Lagrange constraint manager for target action rate.

    Supports HOLD/BUY/SELL action balance constraints.
    """

    def __init__(
        self,
        target_action: ActionType = "SELL",
        r_target: float = 0.15,
        tolerance: float = 0.05,
        eta: float = 1e-3,
        lambda_max: float = 1.0,
        warmup_steps: int = 5000,
    ):
        """
        Initialize Lagrange constraint.

        Args:
            target_action: Action to balance ("HOLD", "BUY", or "SELL")
            r_target: Target action rate (default: 0.15 = 15%)
            tolerance: Allowed deviation before penalty (default: 0.05 = ±5%)
            eta: Dual learning rate (default: 1e-3)
            lambda_max: Maximum lambda value (default: 1.0)
            warmup_steps: Steps before activating constraint (default: 5000)
        """
        self.target_action = target_action
        self.r_target = r_target
        self.tolerance = tolerance
        self.eta = eta
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps

        # Action mapping
        # The project uses ACTION constants (HOLD=0, BUY=1, SELL=-1) for action
        # values while legal action masks use columns [HOLD, BUY, SELL] -> indices
        # [0,1,2]. Keep both representations to correctly count chosen+legal
        # occurrences.
        from ztb.trading.constants import ACTION_HOLD, ACTION_BUY, ACTION_SELL

        self.target_value = {
            "HOLD": ACTION_HOLD,
            "BUY": ACTION_BUY,
            "SELL": ACTION_SELL,
        }[target_action]
        # Backwards-compatible alias expected by some tests
        self.action_idx = self.target_value
        # column index in legal_masks corresponding to target action
        self.target_col_index = {"HOLD": 0, "BUY": 1, "SELL": 2}[target_action]

        # Dual variable
        self.lambda_dual = 0.0

        # Statistics
        self.step_count = 0
        self.action_rates: deque[float] = deque(maxlen=5000)
        self.penalties: deque[float] = deque(maxlen=5000)
        self.lambda_history: deque[float] = deque(maxlen=5000)

    def compute_penalty(
        self,
        actions: NDArray[np.int64],
        legal_masks: Union[NDArray[np.bool_], NDArray[np.floating[Any]]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Lagrange penalty term.

        Args:
            actions: Batch of actions [batch_size]
            legal_masks: Legal action masks [batch_size, n_actions]

        Returns:
            Tuple of (penalty, info_dict)
            - penalty: Lagrange penalty (to subtract from PPO loss)
            - info_dict: Statistics for logging
        """
        self.step_count += 1

        # Compute target action rate (legal steps only)
        # Actions: 0=HOLD, 1=BUY, 2=SELL
        # actions may use ACTION_* values (e.g., -1 for SELL) while legal_masks
        # index columns as [HOLD, BUY, SELL]. Build masks accordingly.
        action_mask = actions == self.target_value
        legal_action_mask = legal_masks[:, self.target_col_index] == 1

        # Count legal steps where target action was chosen AND legal
        legal_action_count = np.sum(action_mask & legal_action_mask)

        # Count total legal steps (where at least one action is legal)
        legal_steps_mask = np.any(legal_masks == 1, axis=1)
        total_legal_steps = np.sum(legal_steps_mask)

        if total_legal_steps > 0:
            r_actual = legal_action_count / total_legal_steps
        else:
            r_actual = 0.0

        # Compute penalty (only if past warmup)
        if self.step_count > self.warmup_steps:
            # penalty = -λ * max(0, |r_target - r_actual| - tolerance)
            deviation = abs(self.r_target - r_actual)
            constraint_violation = max(0.0, deviation - self.tolerance)
            penalty = -self.lambda_dual * constraint_violation

            # Update dual variable
            # λ ← clip(λ + η * (|r_target - r_actual| - tolerance), 0, λ_max)
            lambda_delta = self.eta * (deviation - self.tolerance)
            self.lambda_dual = np.clip(
                self.lambda_dual + lambda_delta,
                0.0,
                self.lambda_max,
            )
        else:
            penalty = 0.0
            constraint_violation = 0.0
            deviation = 0.0

        # Record statistics
        self.action_rates.append(r_actual)
        self.penalties.append(penalty)
        self.lambda_history.append(self.lambda_dual)

        info = {
            f"r_{self.target_action.lower()}": r_actual,
            "r_target": self.r_target,
            "deviation": deviation,
            "lambda_dual": self.lambda_dual,
            "constraint_violation": constraint_violation,
            "penalty": penalty,
            f"legal_{self.target_action.lower()}_count": int(legal_action_count),
            "total_legal_steps": int(total_legal_steps),
        }

        return penalty, info

    def get_statistics(self, window: int = 100) -> Dict[str, float]:
        """
        Get moving statistics.

        Args:
            window: Moving average window

        Returns:
            Dictionary with statistics
        """
        if len(self.action_rates) == 0:
            return {
                f"r_{self.target_action.lower()}_mean": 0.0,
                "lambda_dual": self.lambda_dual,
                "penalty_mean": 0.0,
            }

        recent_rates = self.action_rates[-window:]
        recent_penalties = self.penalties[-window:]

        return {
            f"r_{self.target_action.lower()}_mean": float(np.mean(recent_rates)),
            f"r_{self.target_action.lower()}_std": float(np.std(recent_rates)),
            "lambda_dual": self.lambda_dual,
            "penalty_mean": float(np.mean(recent_penalties)),
            "constraint_active": self.step_count > self.warmup_steps,
        }

    def reset(self) -> None:
        """Reset constraint state."""
        self.lambda_dual = 0.0
        self.step_count = 0
        self.action_rates = []
        self.penalties = []
        self.lambda_history = []


def apply_lagrange_to_loss(
    ppo_loss: torch.Tensor,
    actions: NDArray[np.int64],
    legal_masks: Union[NDArray[np.bool_], NDArray[np.floating[Any]]],
    lagrange: LagrangeConstraint,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Apply Lagrange constraint to PPO loss.

    Args:
        ppo_loss: Original PPO loss (scalar tensor)
        actions: Batch actions
        legal_masks: Legal action masks
        lagrange: Lagrange constraint instance

    Returns:
        Tuple of (constrained_loss, info_dict)
    """
    # Compute penalty
    penalty, info = lagrange.compute_penalty(actions, legal_masks)

    # Add penalty to loss (penalty is negative, so this increases loss if constraint violated)
    # L_constrained = L_PPO - λ * max(0, r_min - r_sell)
    constrained_loss = ppo_loss + penalty

    return constrained_loss, info


def test_lagrange_constraint() -> None:
    """Test Lagrange constraint with synthetic data."""
    print("Testing Lagrange Constraint...")

    # Create constraint for SELL action
    constraint = LagrangeConstraint(
        target_action="SELL",
        r_target=0.15,
        tolerance=0.05,
        eta=1e-3,
        lambda_max=1.0,
        warmup_steps=100,
    )

    # Simulate low SELL rate scenario
    print("\nScenario 1: Low SELL rate (should increase lambda)")
    for i in range(200):
        # Actions: mostly HOLD (0) with few SELL (2)
        actions = np.random.choice([0, 1, 2], size=100, p=[0.7, 0.25, 0.05])
        legal_masks = np.ones((100, 3))  # All legal

        penalty, info = constraint.compute_penalty(actions, legal_masks)

        if (i + 1) % 50 == 0:
            print(
                f"  Step {i+1}: r_sell={info['r_sell']:.3f}, λ={info['lambda_dual']:.4f}, penalty={penalty:.4f}"
            )

    # Simulate good SELL rate scenario
    print("\nScenario 2: Good SELL rate (should maintain/decrease lambda)")
    for i in range(100):
        # Actions: balanced SELL rate
        actions = np.random.choice([0, 1, 2], size=100, p=[0.5, 0.3, 0.2])
        legal_masks = np.ones((100, 3))

        penalty, info = constraint.compute_penalty(actions, legal_masks)

        if (i + 1) % 25 == 0:
            print(
                f"  Step {i+1}: r_sell={info['r_sell']:.3f}, λ={info['lambda_dual']:.4f}, penalty={penalty:.4f}"
            )

    # Get final statistics
    stats = constraint.get_statistics()
    print("\nFinal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Lagrange constraint test complete")


if __name__ == "__main__":
    test_lagrange_constraint()
