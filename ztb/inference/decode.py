"""
Unified action decoding with strict mask enforcement and tiebreaker.

This module implements the standardized inference pipeline:
1. Apply action mask (illegal actions → logits = -1e9)
2. Apply temperature scaling
3. Softmax normalization
4. Tiebreaker logic (if enabled)
5. Action selection (argmax or sample)

Critical Requirements:
- Mask BEFORE softmax (illegal actions get ~0 probability)
- Temperature scaling for exploration control
- Tiebreaker: top1==HOLD AND (p1-p2)<tau AND legal(top2) → select top2
- Numerical stability (logsumexp for softmax)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class InferenceConfig:
    """Configuration for action decoding."""

    temperature: float = 0.7  # Temperature for softmax scaling (lower=more greedy)
    tiebreaker_tau: float = 0.05  # Margin threshold for tiebreaker
    enable_tiebreaker: bool = True  # Enable tiebreaker logic
    deterministic: bool = True  # Use argmax (True) or sample (False)
    
    # Robustness guards
    min_temperature: float = 0.5  # Minimum safe temperature
    max_temperature: float = 1.5  # Maximum safe temperature
    logits_clip_value: float = 20.0  # Clip logits to [-clip, +clip]
    fallback_action: int = 0  # Fallback action when all actions illegal (HOLD)


def decode_action(
    logits: np.ndarray | torch.Tensor,
    legal_mask: np.ndarray | torch.Tensor,
    config: Optional[InferenceConfig] = None,
) -> Tuple[int | np.ndarray, dict]:
    """
    Decode action from logits with strict mask enforcement.

    Pipeline:
    1. Mask illegal actions (logits → -1e9)
    2. Temperature scaling (logits / T)
    3. Softmax normalization
    4. Tiebreaker (if enabled and conditions met)
    5. Action selection (argmax or sample)

    Args:
        logits: Raw action logits [batch_size, n_actions] or [n_actions]
        legal_mask: Legal action mask [batch_size, n_actions] or [n_actions]
                   (1=legal, 0=illegal)
        config: Inference configuration (uses defaults if None)

    Returns:
        Tuple of:
        - action: Selected action(s) (int if single, ndarray if batch)
        - info: Dictionary with:
            - probabilities: Post-softmax probabilities
            - top2_actions: Top 2 actions by probability
            - top2_probs: Top 2 probabilities
            - margin: Probability margin (p1 - p2)
            - tiebreaker_activated: Whether tiebreaker was used

    Raises:
        ValueError: If all actions are illegal
    """
    if config is None:
        config = InferenceConfig()

    # Guard: Validate temperature range
    if not (config.min_temperature <= config.temperature <= config.max_temperature):
        import warnings
        warnings.warn(
            f"Temperature {config.temperature} outside safe range "
            f"[{config.min_temperature}, {config.max_temperature}]. "
            f"Clamping to range."
        )
        config.temperature = np.clip(
            config.temperature, config.min_temperature, config.max_temperature
        )

    # Convert to numpy if torch tensor
    is_torch = isinstance(logits, torch.Tensor)
    if is_torch:
        logits_np = logits.detach().cpu().numpy()
        mask_np = legal_mask.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits)
        mask_np = np.asarray(legal_mask)

    # Guard: Clip logits to safe range (prevent overflow in exp)
    logits_np = np.clip(logits_np, -config.logits_clip_value, config.logits_clip_value)

    # Handle single observation (add batch dimension)
    single_obs = logits_np.ndim == 1
    if single_obs:
        logits_np = logits_np[np.newaxis, :]
        mask_np = mask_np[np.newaxis, :]

    batch_size, n_actions = logits_np.shape

    # Guard: Handle all-illegal-actions case
    # If all actions are illegal, fall back to fallback_action (HOLD)
    mask_sums = mask_np.sum(axis=1)
    if not np.all(mask_sums > 0):
        import warnings
        all_illegal_mask = mask_sums == 0
        warnings.warn(
            f"{all_illegal_mask.sum()} observations have no legal actions. "
            f"Falling back to action {config.fallback_action} (HOLD)."
        )
        # Set fallback action as legal for these observations
        mask_np[all_illegal_mask, config.fallback_action] = 1

    # Step 1: Apply mask (illegal actions → -1e9)
    masked_logits = np.where(mask_np, logits_np, -1e9)

    # Step 2: Temperature scaling
    scaled_logits = masked_logits / config.temperature

    # Step 3: Softmax normalization (numerically stable)
    # Use logsumexp trick: softmax(x) = exp(x - logsumexp(x))
    max_logits = np.max(scaled_logits, axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logits)
    sum_exp = exp_logits.sum(axis=1, keepdims=True)
    
    # Guard: Detect NaN/Inf in softmax and retry with higher temperature
    if np.any(~np.isfinite(sum_exp)) or np.any(sum_exp == 0):
        import warnings
        warnings.warn(
            "NaN or Inf detected in softmax. Retrying with temperature=1.0."
        )
        # Retry with temperature = 1.0
        scaled_logits = masked_logits / 1.0
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        
        # If still broken, fall back to uniform distribution over legal actions
        if np.any(~np.isfinite(sum_exp)) or np.any(sum_exp == 0):
            warnings.warn(
                "Softmax still invalid after retry. Falling back to uniform over legal actions."
            )
            probabilities = mask_np.astype(np.float32)
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        else:
            probabilities = exp_logits / sum_exp
    else:
        probabilities = exp_logits / sum_exp

    # Step 4 & 5: Action selection with tiebreaker
    actions = np.zeros(batch_size, dtype=np.int32)
    tiebreaker_activated = np.zeros(batch_size, dtype=bool)
    top2_actions = np.zeros((batch_size, 2), dtype=np.int32)
    top2_probs = np.zeros((batch_size, 2), dtype=np.float32)
    margins = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        probs = probabilities[i]
        mask = mask_np[i]

        # Get top 2 actions by probability
        sorted_indices = np.argsort(probs)[::-1]
        top2_actions[i] = sorted_indices[:2]
        top2_probs[i] = probs[sorted_indices[:2]]
        margins[i] = top2_probs[i, 0] - top2_probs[i, 1]

        top1_action = top2_actions[i, 0]
        top2_action = top2_actions[i, 1]

        # Tiebreaker logic
        if (
            config.enable_tiebreaker
            and config.deterministic
            and top1_action == 0  # HOLD
            and margins[i] < config.tiebreaker_tau
            and mask[top2_action] == 1  # top2 is legal (explicit check)
        ):
            # Activate tiebreaker: select top2 instead of top1
            actions[i] = top2_action
            tiebreaker_activated[i] = True
        elif config.deterministic:
            # Standard deterministic: argmax
            actions[i] = top1_action
        else:
            # Stochastic: sample from distribution
            actions[i] = np.random.choice(n_actions, p=probs)

    # Prepare info dict
    info = {
        "probabilities": probabilities,
        "top2_actions": top2_actions,
        "top2_probs": top2_probs,
        "margin": margins,
        "tiebreaker_activated": tiebreaker_activated,
    }

    # Remove batch dimension if single observation
    if single_obs:
        action = int(actions[0])
        info["probabilities"] = probabilities[0]
        info["top2_actions"] = top2_actions[0]
        info["top2_probs"] = top2_probs[0]
        info["margin"] = float(margins[0])
        info["tiebreaker_activated"] = bool(tiebreaker_activated[0])
    else:
        action = actions

    return action, info


def compute_legal_sell_rate(
    actions: np.ndarray, legal_masks: np.ndarray
) -> dict[str, float]:
    """
    Compute legal SELL rate statistics.

    Args:
        actions: Selected actions [n_steps]
        legal_masks: Legal action masks [n_steps, n_actions]

    Returns:
        Dictionary with:
        - total_steps: Total number of steps
        - legal_sell_steps: Steps where SELL was legal
        - sell_actions: Steps where SELL was chosen
        - legal_sell_rate: SELL rate among legal SELL steps (target: ≥15%)
        - overall_sell_rate: SELL rate overall
    """
    SELL_ACTION = 2

    total_steps = len(actions)
    legal_sell_steps = np.sum(legal_masks[:, SELL_ACTION])
    sell_actions = np.sum(actions == SELL_ACTION)

    # Legal SELL rate: among steps where SELL was legal, how often was it chosen?
    legal_sell_rate = (
        sell_actions / legal_sell_steps if legal_sell_steps > 0 else 0.0
    )
    overall_sell_rate = sell_actions / total_steps if total_steps > 0 else 0.0

    return {
        "total_steps": total_steps,
        "legal_sell_steps": int(legal_sell_steps),
        "sell_actions": int(sell_actions),
        "legal_sell_rate": legal_sell_rate,
        "overall_sell_rate": overall_sell_rate,
    }
