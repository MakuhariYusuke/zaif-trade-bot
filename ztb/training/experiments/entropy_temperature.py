#!/usr/bin/env python3
"""
Target Entropy for automatic exploration control.

This module implements automatic entropy coefficient (ent_coef) learning,
similar to SAC's temperature parameter. It prevents premature exploration
collapse by maintaining a target entropy level.

Key concept:
- Fixed ent_coef: Exploration can dry up early (entropy → 0)
- Learned ent_coef (α): Automatically adjust to maintain H* = κ·log(|A|)
  where κ ∈ [0, 1] (typically 0.7) and |A| is the number of actions.

Implementation:
- L_temp = α · (H* - H_π)  [Minimize this loss]
- α ← α + lr_α · ∇_α L_temp

Expected effect: Exploration is sustained throughout training,
minority actions continue to be sampled.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TargetEntropyController:
    """
    Automatic entropy coefficient controller.

    Learns ent_coef (temperature) to maintain target entropy level,
    preventing both premature exploration collapse and excessive randomness.
    """

    def __init__(
        self,
        n_actions: int = 3,
        target_entropy_ratio: float = 0.7,
        initial_temperature: float = 0.01,
        lr_temperature: float = 3e-4,
        device: str = "cpu",
    ):
        """
        Initialize target entropy controller.

        Args:
            n_actions: Number of discrete actions
            target_entropy_ratio: Target entropy as ratio of max entropy (κ)
                                 H* = κ · log(n_actions)
            initial_temperature: Initial temperature (α_0)
            lr_temperature: Learning rate for temperature updates
            device: Device for torch tensors
        """
        self.n_actions = n_actions
        self.target_entropy_ratio = target_entropy_ratio
        self.device = device

        # Target entropy: H* = κ · log(|A|)
        self.target_entropy = target_entropy_ratio * np.log(n_actions)

        # Temperature parameter (log_alpha for numerical stability)
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(initial_temperature), dtype=torch.float32)
        )

        # Optimizer for temperature
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_temperature)

        # Initialize statistics
        self.reset_statistics()

        # Statistics
        self.reset_statistics()

        logger.info(
            f"TargetEntropyController initialized: "
            f"H*={self.target_entropy:.4f}, "
            f"α_0={initial_temperature:.6f}"
        )

    def reset_statistics(self) -> None:
        """Reset statistics tracking."""
        self.history: Dict[str, List[float]] = {"alpha": [], "entropy": [], "loss": []}

    @property
    def compute_entropy(
        self, action_logits: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute policy entropy from action logits.

        Args:
            action_logits: Raw logits [batch_size, n_actions]
            actions: Taken actions (optional, for verification)

        Returns:
            Mean entropy over batch
        """
        # Convert to probabilities
        probs = torch.softmax(action_logits, dim=-1)

        # Clip for numerical stability
        probs = torch.clamp(probs, min=1e-8, max=1.0)

        # H(π) = -Σ p(a) log p(a)
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Return mean entropy
        return entropy.mean()

    def update(self, current_entropy: torch.Tensor) -> Tuple[float, float]:
        """
        Update temperature parameter.

        Args:
            current_entropy: Current policy entropy (scalar tensor)

        Returns:
            Tuple of (temperature_loss, current_alpha)
        """
        # Temperature loss: L = α · (H* - H_π)
        # We want to minimize this, which encourages H_π → H*
        target_entropy_tensor = torch.tensor(
            self.target_entropy, dtype=torch.float32, device=self.device
        )

        alpha = torch.exp(self.log_alpha)
        temp_loss = alpha * (target_entropy_tensor - current_entropy)

        # Update temperature
        self.alpha_optimizer.zero_grad()
        temp_loss.backward()  # type: ignore[no-untyped-call]
        self.alpha_optimizer.step()

        # Track statistics
        current_alpha = alpha.item()
        loss_value = temp_loss.item()
        entropy_value = current_entropy.item()

        self.history["alpha"].append(current_alpha)
        self.history["entropy"].append(entropy_value)
        self.history["loss"].append(loss_value)

        logger.debug(
            f"Entropy: {entropy_value:.4f} (target: {self.target_entropy:.4f}), "
            f"α: {current_alpha:.6f}, loss: {loss_value:.6f}"
        )

        return loss_value, current_alpha

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.history["alpha"]:
            return {
                "current_alpha": self.alpha,
                "target_entropy": self.target_entropy,
                "num_updates": 0,
            }

        return {
            "current_alpha": self.alpha,
            "target_entropy": self.target_entropy,
            "num_updates": len(self.history["alpha"]),
            "mean_entropy": np.mean(self.history["entropy"]),
            "mean_alpha": np.mean(self.history["alpha"]),
            "alpha_std": np.std(self.history["alpha"]),
        }

    def should_update(self, step: int, update_frequency: int = 1) -> bool:
        """
        Check if temperature should be updated at this step.
        """
        return step % update_frequency == 0
