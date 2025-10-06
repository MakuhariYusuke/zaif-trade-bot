"""
Lagrange Constraint for Target SELL Rate.

Adds explicit constraint to maintain minimum SELL rate during training:
L(θ) = L_PPO(θ) - λ * max(0, r_min - r_sell(θ))

Where:
- r_min: Target minimum SELL rate (e.g., 0.15 = 15%)
- r_sell: Current batch SELL rate (legal steps only)
- λ: Lagrange multiplier (dual variable)

Dual update: λ ← clip(λ + η * (r_min - r_sell), 0, λ_max)
"""

from typing import Dict, Tuple
import numpy as np
import torch


class LagrangeConstraint:
    """
    Lagrange constraint manager for target SELL rate.
    """
    
    def __init__(
        self,
        r_min: float = 0.15,
        eta: float = 1e-3,
        lambda_max: float = 1.0,
        warmup_steps: int = 5000,
    ):
        """
        Initialize Lagrange constraint.
        
        Args:
            r_min: Target minimum SELL rate (default: 0.15 = 15%)
            eta: Dual learning rate (default: 1e-3)
            lambda_max: Maximum lambda value (default: 1.0)
            warmup_steps: Steps before activating constraint (default: 5000)
        """
        self.r_min = r_min
        self.eta = eta
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps
        
        # Dual variable
        self.lambda_dual = 0.0
        
        # Statistics
        self.step_count = 0
        self.sell_rates = []
        self.penalties = []
        self.lambda_history = []
    
    def compute_penalty(
        self,
        actions: np.ndarray,
        legal_masks: np.ndarray,
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
        
        # Compute SELL rate (legal steps only)
        # Actions: 0=HOLD, 1=BUY, 2=SELL
        sell_mask = actions == 2
        legal_sell_mask = legal_masks[:, 2] == 1  # SELL is legal
        
        # Count legal steps where SELL was chosen AND legal
        legal_sell_count = np.sum(sell_mask & legal_sell_mask)
        
        # Count total legal steps (where at least one action is legal)
        legal_steps_mask = np.any(legal_masks == 1, axis=1)
        total_legal_steps = np.sum(legal_steps_mask)
        
        if total_legal_steps > 0:
            r_sell = legal_sell_count / total_legal_steps
        else:
            r_sell = 0.0
        
        # Compute penalty (only if past warmup)
        if self.step_count > self.warmup_steps:
            # penalty = -λ * max(0, r_min - r_sell)
            constraint_violation = max(0.0, self.r_min - r_sell)
            penalty = -self.lambda_dual * constraint_violation
            
            # Update dual variable
            # λ ← clip(λ + η * (r_min - r_sell), 0, λ_max)
            lambda_delta = self.eta * (self.r_min - r_sell)
            self.lambda_dual = np.clip(
                self.lambda_dual + lambda_delta,
                0.0,
                self.lambda_max,
            )
        else:
            penalty = 0.0
            constraint_violation = 0.0
        
        # Record statistics
        self.sell_rates.append(r_sell)
        self.penalties.append(penalty)
        self.lambda_history.append(self.lambda_dual)
        
        info = {
            "r_sell": r_sell,
            "r_min": self.r_min,
            "lambda_dual": self.lambda_dual,
            "constraint_violation": constraint_violation,
            "penalty": penalty,
            "legal_sell_count": int(legal_sell_count),
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
        if len(self.sell_rates) == 0:
            return {
                "r_sell_mean": 0.0,
                "lambda_dual": self.lambda_dual,
                "penalty_mean": 0.0,
            }
        
        recent_rates = self.sell_rates[-window:]
        recent_penalties = self.penalties[-window:]
        
        return {
            "r_sell_mean": float(np.mean(recent_rates)),
            "r_sell_std": float(np.std(recent_rates)),
            "lambda_dual": self.lambda_dual,
            "penalty_mean": float(np.mean(recent_penalties)),
            "constraint_active": self.step_count > self.warmup_steps,
        }
    
    def reset(self):
        """Reset constraint state."""
        self.lambda_dual = 0.0
        self.step_count = 0
        self.sell_rates = []
        self.penalties = []
        self.lambda_history = []


def apply_lagrange_to_loss(
    ppo_loss: torch.Tensor,
    actions: np.ndarray,
    legal_masks: np.ndarray,
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


def test_lagrange_constraint():
    """Test Lagrange constraint with synthetic data."""
    print("Testing Lagrange Constraint...")
    
    # Create constraint
    constraint = LagrangeConstraint(
        r_min=0.15,
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
            print(f"  Step {i+1}: r_sell={info['r_sell']:.3f}, λ={info['lambda_dual']:.4f}, penalty={penalty:.4f}")
    
    # Simulate good SELL rate scenario
    print("\nScenario 2: Good SELL rate (should maintain/decrease lambda)")
    for i in range(100):
        # Actions: balanced SELL rate
        actions = np.random.choice([0, 1, 2], size=100, p=[0.5, 0.3, 0.2])
        legal_masks = np.ones((100, 3))
        
        penalty, info = constraint.compute_penalty(actions, legal_masks)
        
        if (i + 1) % 25 == 0:
            print(f"  Step {i+1}: r_sell={info['r_sell']:.3f}, λ={info['lambda_dual']:.4f}, penalty={penalty:.4f}")
    
    # Get final statistics
    stats = constraint.get_statistics()
    print("\nFinal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Lagrange constraint test complete")


if __name__ == "__main__":
    test_lagrange_constraint()
