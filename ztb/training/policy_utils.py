"""
Common policy utilities for training.

This module provides utility functions for policy manipulation
shared across different trainer implementations.
"""

from typing import Any, Optional

import torch
from sb3_contrib import MaskablePPO

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def neutralize_policy_bias(model: Optional[MaskablePPO]) -> None:
    """
    Neutralize policy head bias to prevent initial action preferences.
    
    This function zeros out the bias in the policy network to ensure
    the model doesn't have initial preferences for specific actions.
    
    Args:
        model: The MaskablePPO model to neutralize. If None, logs a warning.
    """
    if model is None:
        logger.warning("Model not initialized, cannot neutralize bias")
        return

    policy = model.policy

    # Try different policy structures
    bias_neutralized = False

    # Try policy_net (common in sb3_contrib)
    if hasattr(policy, "policy_net"):
        policy_head = (
            policy.policy_net[-1]
            if isinstance(policy.policy_net, list)
            else policy.policy_net
        )
        if hasattr(policy_head, "bias") and getattr(policy_head, "bias", None) is not None:
            bias = getattr(policy_head, "bias")
            with torch.no_grad():
                bias.zero_()
            logger.info("Neutralized policy bias in policy_net")
            bias_neutralized = True

    # Try action_net (alternative structure)
    if not bias_neutralized and hasattr(policy, "action_net"):
        action_net = policy.action_net
        if hasattr(action_net, "bias") and getattr(action_net, "bias", None) is not None:
            bias = getattr(action_net, "bias")
            with torch.no_grad():
                bias.zero_()
            logger.info("Neutralized policy bias in action_net")
            bias_neutralized = True

    # Try mlp_extractor + action_net combination
    if not bias_neutralized and hasattr(policy, "mlp_extractor"):
        if hasattr(policy.mlp_extractor, "policy_net"):
            policy_mlp = policy.mlp_extractor.policy_net
            last_layer = policy_mlp[-1] if hasattr(policy_mlp, "__getitem__") else None
            if last_layer and hasattr(last_layer, "bias") and getattr(last_layer, "bias", None) is not None:
                    bias = getattr(last_layer, "bias")
                    with torch.no_grad():
                        bias.zero_()
                    logger.info("Neutralized policy bias in mlp_extractor.policy_net")
                    bias_neutralized = True

    if not bias_neutralized:
        logger.warning(
            "Could not find policy bias to neutralize. "
            "Policy structure may be different than expected."
        )


def get_policy_entropy_coefficient(model: MaskablePPO) -> float:
    """
    Get the current entropy coefficient from the model.
    
    Args:
        model: The MaskablePPO model.
        
    Returns:
        The current entropy coefficient value.
    """
    if hasattr(model, "ent_coef"):
        ent_coef = model.ent_coef
        return float(ent_coef)
    return 0.0


def set_policy_entropy_coefficient(model: MaskablePPO, new_ent_coef: float) -> None:
    """
    Set the entropy coefficient for the model.
    
    Args:
        model: The MaskablePPO model.
        new_ent_coef: The new entropy coefficient value.
    """
    if hasattr(model, "ent_coef"):
        model.ent_coef = new_ent_coef
        logger.debug(f"Updated entropy coefficient to {new_ent_coef:.6f}")
    else:
        logger.warning("Model does not have ent_coef attribute")


def apply_cosine_decay_entropy(
    model: MaskablePPO,
    current_step: int,
    total_steps: int,
    initial_ent_coef: float,
    final_ent_coef: float,
) -> None:
    """
    Apply cosine decay schedule to entropy coefficient.
    
    This implements a cosine annealing schedule for the entropy coefficient,
    gradually reducing exploration as training progresses.
    
    Args:
        model: The MaskablePPO model.
        current_step: Current training step.
        total_steps: Total number of training steps.
        initial_ent_coef: Initial entropy coefficient value.
        final_ent_coef: Final entropy coefficient value.
    """
    import math
    
    if current_step >= total_steps:
        new_ent_coef = final_ent_coef
    else:
        # Cosine decay formula
        progress = current_step / total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        new_ent_coef = final_ent_coef + (initial_ent_coef - final_ent_coef) * cosine_decay
    
    set_policy_entropy_coefficient(model, new_ent_coef)
