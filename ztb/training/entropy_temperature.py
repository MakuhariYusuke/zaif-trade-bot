#!/usr/bin/env python3
"""Target Entropy for automatic exploration control.

This module implements automatic entropy coefficient (``ent_coef``) learning,
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

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

logger = logging.getLogger(__name__)

HistoryDict = dict[str, list[float]]

class TargetEntropyController:
    """Automatic entropy coefficient controller.

    Learns ``ent_coef`` (temperature) to maintain target entropy level,
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
        self.target_entropy = float(target_entropy_ratio * np.log(n_actions))

        # Temperature parameter (log_alpha for numerical stability)
        self.log_alpha = nn.Parameter(
            torch.tensor(
                np.log(initial_temperature),
                dtype=torch.float32,
                device=self.device,
            )
        )

        # Optimizer for temperature is created lazily because many tests and
        # lightweight code paths only inspect configuration/statistics.
        self._lr_temperature = lr_temperature
        self._alpha_optimizer: Adam | None = None

        # Initialize statistics container
        self.history: HistoryDict = {"alpha": [], "entropy": [], "loss": []}
        self.reset_statistics()

        logger.info(
            f"TargetEntropyController initialized: "
            f"H*={self.target_entropy:.4f}, "
            f"α_0={initial_temperature:.6f}"
        )

    def reset_statistics(self) -> None:
        """Reset statistics tracking."""
        self.history = {"alpha": [], "entropy": [], "loss": []}

    @property
    def alpha(self) -> float:
        """Get current temperature (α = exp(log_alpha))."""
        return cast(float, torch.exp(self.log_alpha).item())

    @property
    def alpha_optimizer(self) -> Adam:
        """Create the optimizer only when temperature updates are requested."""
        if self._alpha_optimizer is None:
            self._alpha_optimizer = Adam([self.log_alpha], lr=self._lr_temperature)
        return self._alpha_optimizer

    def compute_entropy(
        self,
        action_logits: Tensor,
        actions: Tensor | None = None,
    ) -> Tensor:
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

    def update(self, current_entropy: Tensor) -> tuple[float, float]:
        """
        Update temperature parameter.

        Args:
            current_entropy: Current policy entropy (scalar tensor)

        Returns:
            tuple of (temperature_loss, current_alpha)
        """
        # Temperature updates should remain trainable even if an outer caller
        # accidentally leaves grad mode disabled. We only optimize α here, so
        # the entropy input is treated as a detached statistic.
        with torch.enable_grad():
            target_entropy_tensor = torch.tensor(
                self.target_entropy,
                dtype=torch.float32,
                device=self.device,
            )

            alpha = torch.exp(self.log_alpha)
            temp_loss = alpha * (target_entropy_tensor - current_entropy.detach())

            self.alpha_optimizer.zero_grad()
            temp_loss.backward()
            self.alpha_optimizer.step()

        # Track statistics
        current_alpha = alpha.detach().item()
        loss_value = temp_loss.detach().item()
        entropy_value = current_entropy.detach().item()

        self.history["alpha"].append(current_alpha)
        self.history["entropy"].append(entropy_value)
        self.history["loss"].append(loss_value)

        logger.debug(
            f"Entropy: {entropy_value:.4f} (target: {self.target_entropy:.4f}), "
            f"α: {current_alpha:.6f}, loss: {loss_value:.6f}"
        )

        return loss_value, current_alpha

    def get_statistics(self) -> dict[str, Any]:
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
            "mean_entropy": float(np.mean(self.history["entropy"])),
            "mean_alpha": float(np.mean(self.history["alpha"])),
            "alpha_std": float(np.std(self.history["alpha"])),
        }

    def should_update(self, step: int, update_frequency: int = 1) -> bool:
        """
        Check if temperature should be updated at this step.

        Args:
            step: Current training step
            update_frequency: How often to update (default: every step)

        Returns:
            True if should update
        """
        return (step % update_frequency) == 0
