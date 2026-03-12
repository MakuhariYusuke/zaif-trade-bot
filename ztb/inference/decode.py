from __future__ import annotations

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
from typing import Any

import numpy as np
try:
    import torch
except Exception:
    torch = None
from numpy.typing import NDArray

from ztb.trading.constants import ACTION_HOLD
from ztb.trading.environment.constants import EPSILON


def _torch_tensor_type() -> type[Any] | None:
    """Return a valid torch.Tensor type when the torch module is well-formed."""
    if torch is None:
        return None
    tensor_type = getattr(torch, "Tensor", None)
    return tensor_type if isinstance(tensor_type, type) else None


def _to_numpy_array(value: Any) -> NDArray[np.generic]:
    """Convert torch/stub tensors and array-likes into numpy arrays."""
    torch_tensor_type = _torch_tensor_type()
    if torch_tensor_type is not None and isinstance(value, torch_tensor_type):
        tensor = value.detach() if hasattr(value, "detach") else value
        tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
        if hasattr(tensor, "numpy"):
            return np.asarray(tensor.numpy())
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    if hasattr(value, "_arr"):
        return np.asarray(value._arr)
    array = np.asarray(value)
    if array.dtype == object and array.ndim == 0:
        scalar = array.item()
        if scalar is not value:
            return _to_numpy_array(scalar)
    if array.size == 0 and hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return np.asarray([value[i] for i in range(len(value))])
        except Exception:
            pass
    return array

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

    # Advantage-aware tiebreaker
    enable_advantage_tiebreaker: bool = True  # Use advantage sign for tiebreaker
    advantage_epsilon: float = EPSILON  # Threshold for advantage comparison

    # Cost-aware decode gate
    enable_cost_gate: bool = True  # Enable cost-based filtering
    cost_gate_lambda: float = 1.2  # Cost multiplier (λ): require ΔA ≥ λ * cost
    transaction_cost: float = 0.001  # Default transaction cost (0.1%)
    slippage: float = 0.0005  # Default slippage (0.05%)

def decode_action(
    logits: NDArray[np.float32] | torch.Tensor,
    legal_mask: NDArray[np.bool_] | torch.Tensor,
    config: InferenceConfig | None = None,
    advantages: NDArray[np.float32] | torch.Tensor | None = None,
    current_position: int | None = None,
) -> tuple[int | NDArray[np.integer[Any]], dict[str, Any]]:
    """
    Decode action from logits with strict mask enforcement.

    Pipeline:
    1. Mask illegal actions (logits → -1e9)
    2. Temperature scaling (logits / T)
    3. Softmax normalization
    4. Advantage-aware tiebreaker (if enabled and advantages provided)
    5. Probability-margin tiebreaker (if enabled and conditions met)
    6. Cost-aware gate (if enabled and position change detected)
    7. Action selection (argmax or sample)

    Args:
        logits: Raw action logits [batch_size, n_actions] or [n_actions]
        legal_mask: Legal action mask [batch_size, n_actions] or [n_actions]
                   (1=legal, 0=illegal)
        config: Inference configuration (uses defaults if None)
        advantages: Advantage values [batch_size, n_actions] or [n_actions]
                   (optional, for advantage-aware tiebreaker)
        current_position: Current position/action (optional, for cost gate)

    Returns:
        tuple of:
        - action: Selected action(s) (int if single, ndarray if batch)
        - info: Dictionary with:
            - probabilities: Post-softmax probabilities
            - top2_actions: Top 2 actions by probability
            - top2_probs: Top 2 probabilities
            - margin: Probability margin (p1 - p2)
            - tiebreaker_activated: Whether tiebreaker was used
            - tiebreaker_reason: Reason for tiebreaker ('prob_margin', 'advantage_sign', or None)
            - cost_gate_triggered: Whether cost gate prevented action
            - estimated_cost: Estimated transaction cost (if cost gate evaluated)

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

    # Convert to numpy if torch tensor or lightweight torch stub tensor
    torch_tensor_type = _torch_tensor_type()
    logits_np = _to_numpy_array(logits)
    mask_np = _to_numpy_array(legal_mask)

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
        # set fallback action as legal for these observations
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

        warnings.warn("NaN or Inf detected in softmax. Retrying with temperature=1.0.")
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

    # Convert advantages to numpy if provided
    advantages_np = None
    if advantages is not None:
        advantages_np = _to_numpy_array(advantages)

        # Handle single observation case
        if advantages_np.ndim == 1:
            advantages_np = advantages_np[np.newaxis, :]

    # Step 4 & 5 & 6: Action selection with advantage-aware tiebreaker and cost gate
    actions = np.zeros(batch_size, dtype=np.int32)
    tiebreaker_activated = np.zeros(batch_size, dtype=bool)
    tiebreaker_reasons: list[str | None] = [None] * batch_size
    cost_gate_triggered = np.zeros(batch_size, dtype=bool)
    estimated_costs = np.zeros(batch_size, dtype=np.float32)
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

        # Initialize selected action with top1 (default)
        selected_action = top1_action
        tiebreaker_reason: str | None = None

        # Advantage-aware tiebreaker (Priority 1: strongest signal)
        if (
            config.enable_advantage_tiebreaker
            and config.deterministic
            and advantages_np is not None
            and mask[top2_action] == 1  # top2 must be legal
        ):
            adv_top1 = advantages_np[i, top1_action]
            adv_top2 = advantages_np[i, top2_action]

            # If top2 has positive advantage and top1 has non-positive advantage
            if (
                adv_top2 > config.advantage_epsilon
                and adv_top1 <= config.advantage_epsilon
            ):
                selected_action = top2_action
                tiebreaker_activated[i] = True
                tiebreaker_reason = "advantage_sign"

        # Probability-margin tiebreaker (Priority 2: if advantage-tiebreaker did not trigger)
        if (
            not tiebreaker_activated[i]
            and config.enable_tiebreaker
            and config.deterministic
            and top1_action == ACTION_HOLD
            and margins[i] < config.tiebreaker_tau
            and mask[top2_action] == 1  # top2 is legal
        ):
            # Activate tiebreaker: select top2 instead of top1
            selected_action = top2_action
            tiebreaker_activated[i] = True
            tiebreaker_reason = "prob_margin"

        # Stochastic sampling (if not deterministic)
        if not config.deterministic and not tiebreaker_activated[i]:
            selected_action = np.random.choice(n_actions, p=probs)

        # Cost-aware gate (Priority 3: filter out unprofitable actions)
        if (
            config.enable_cost_gate
            and advantages_np is not None
            and current_position is not None
            and selected_action != current_position  # Position change
            and selected_action != 0  # Not HOLD
        ):
            # Estimate cost of position change
            total_cost = config.transaction_cost + config.slippage
            estimated_costs[i] = total_cost

            # Calculate advantage delta
            adv_selected = advantages_np[i, selected_action]
            adv_current = (
                advantages_np[i, current_position]
                if current_position < len(advantages_np[i])
                else 0.0
            )
            advantage_delta = adv_selected - adv_current

            # Apply cost gate: require delta_A >= lambda * cost
            cost_threshold = config.cost_gate_lambda * total_cost

            if advantage_delta < cost_threshold:
                # Cost gate triggered: fall back to HOLD
                selected_action = 0  # HOLD
                cost_gate_triggered[i] = True
                # Cancel tiebreaker if cost gate overrides
                tiebreaker_activated[i] = False
                tiebreaker_reason = None

        actions[i] = selected_action
        tiebreaker_reasons[i] = tiebreaker_reason

    # Prepare info dict
    info = {
        "probabilities": probabilities,
        "top2_actions": top2_actions,
        "top2_probs": top2_probs,
        "margin": margins,
        "tiebreaker_activated": tiebreaker_activated,
        "tiebreaker_reason": tiebreaker_reasons,
        "cost_gate_triggered": cost_gate_triggered,
        "estimated_cost": estimated_costs,
    }

    # Remove batch dimension if single observation
    if single_obs:
        action: int | NDArray[np.integer[Any]] = int(actions[0])
        info["probabilities"] = probabilities[0]
        info["top2_actions"] = top2_actions[0]
        info["top2_probs"] = top2_probs[0]
        info["margin"] = float(margins[0])
        info["tiebreaker_activated"] = bool(tiebreaker_activated[0])
        info["tiebreaker_reason"] = tiebreaker_reasons[0]
        info["cost_gate_triggered"] = bool(cost_gate_triggered[0])
        info["estimated_cost"] = float(estimated_costs[0])
    else:
        action = actions

    return action, info

def compute_legal_sell_rate(
    actions: NDArray[np.int64], legal_masks: NDArray[np.bool_]
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
    legal_sell_rate = sell_actions / legal_sell_steps if legal_sell_steps > 0 else 0.0
    overall_sell_rate = sell_actions / total_steps if total_steps > 0 else 0.0

    return {
        "total_steps": total_steps,
        "legal_sell_steps": int(legal_sell_steps),
        "sell_actions": int(sell_actions),
        "legal_sell_rate": legal_sell_rate,
        "overall_sell_rate": overall_sell_rate,
    }
