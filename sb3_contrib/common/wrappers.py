"""Wrapper shims compatible with the local PPO compatibility layer."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

MaskFn = Callable[[gym.Env], object]


def _default_mask_fn(env: gym.Env) -> object:
    """Best-effort default mask getter compatible with current env contracts."""
    for attr_name in ("get_action_masks", "action_mask", "get_legal_actions"):
        attr = getattr(env, attr_name, None)
        if callable(attr):
            return attr()
    raise AttributeError(
        "ActionMasker requires env to expose get_action_masks(), action_mask(), "
        "or get_legal_actions(), or an explicit mask function must be provided."
    )


class ActionMasker(gym.Wrapper):
    """Minimal Gymnasium-compatible ActionMasker shim.

    The local repo does not depend on full ``sb3_contrib`` at runtime, but PPO
    compatibility still expects an env wrapper object that behaves like a normal
    Gymnasium environment and exposes an action mask accessor.
    """

    def __init__(
        self,
        env: gym.Env,
        mask_fn: MaskFn | None = None,
        *,
        action_mask_fn: MaskFn | None = None,
    ) -> None:
        super().__init__(env)
        self.mask_fn = mask_fn or action_mask_fn or _default_mask_fn

    def action_masks(self) -> NDArray[np.bool_]:
        """Return the current legal-action mask in sb3-contrib-compatible shape."""
        mask = self.mask_fn(self.env)
        return np.asarray(mask, dtype=np.bool_)

    def get_action_masks(self) -> NDArray[np.bool_]:
        """Alias used by existing trainer/helpers in this repo."""
        return self.action_masks()
