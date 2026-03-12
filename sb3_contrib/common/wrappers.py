"""Wrapper shims."""
from __future__ import annotations

from collections.abc import Callable


class ActionMasker:
    def __init__(self, env: object, mask_fn: Callable[..., object]) -> None:
        self.env = env
        self.mask_fn = mask_fn
