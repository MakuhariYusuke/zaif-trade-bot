"""Pure PPO policy helpers shared by sidecar runtime and tests."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray


class PPOPolicyLike(Protocol):
    @property
    def policy(self) -> object: ...

    def save(self, path: str) -> None: ...

    def predict(
        self,
        observation: object,
        deterministic: bool = True,
    ) -> tuple[object, object | None]: ...


def coerce_action_index(action: object) -> int:
    """Convert PPO predict output into the discrete trading action index."""

    if isinstance(action, np.ndarray):
        if action.size == 0:
            return 0
        return int(action.reshape(-1)[0])
    if isinstance(action, (int, np.integer)):
        return int(action)
    return 0


def one_hot_ppo_probabilities(action_index: int) -> dict[str, float]:
    """Fallback probability representation when policy logits are unavailable."""

    clamped_action = 0 if action_index < 0 or action_index > 2 else action_index
    return {
        "skip": 1.0 if clamped_action == 0 else 0.0,
        "buy": 1.0 if clamped_action == 1 else 0.0,
        "sell": 1.0 if clamped_action == 2 else 0.0,
    }


def extract_action_probabilities(
    model: PPOPolicyLike,
    observation: object,
    *,
    action_masks: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Extract current buy/sell/skip probabilities from a PPO policy."""

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "obs_to_tensor") or not hasattr(
        policy, "get_distribution"
    ):
        action, _ = model.predict(observation, deterministic=True)
        return one_hot_ppo_probabilities(coerce_action_index(action))

    obs_to_tensor = cast(Any, getattr(policy, "obs_to_tensor"))
    get_distribution = cast(Any, getattr(policy, "get_distribution"))
    obs_tensor, _ = obs_to_tensor(observation)
    distribution = get_distribution(obs_tensor)
    raw_distribution = getattr(distribution, "distribution", distribution)

    probs_like = getattr(raw_distribution, "probs", None)
    if probs_like is None:
        logits = getattr(raw_distribution, "logits", None)
        if logits is not None:
            probs_like = torch.softmax(logits, dim=-1)

    if probs_like is None:
        action, _ = model.predict(observation, deterministic=True)
        return one_hot_ppo_probabilities(coerce_action_index(action))

    if hasattr(probs_like, "detach"):
        probabilities = np.asarray(probs_like.detach().cpu().numpy(), dtype=float)
    else:
        probabilities = np.asarray(probs_like, dtype=float)
    flat_probabilities = probabilities.reshape(-1)
    if flat_probabilities.size < 3:
        action, _ = model.predict(observation, deterministic=True)
        return one_hot_ppo_probabilities(coerce_action_index(action))

    clipped = flat_probabilities[:3]
    if action_masks is not None and action_masks.shape[0] >= 3:
        masked = clipped * action_masks[:3].astype(float)
        if masked.sum() > 0.0:
            clipped = masked / masked.sum()

    return {
        "skip": float(clipped[0]),
        "buy": float(clipped[1]),
        "sell": float(clipped[2]),
    }


__all__ = [
    "PPOPolicyLike",
    "coerce_action_index",
    "extract_action_probabilities",
    "one_hot_ppo_probabilities",
]
