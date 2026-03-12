"""Minimal callback shims."""
from __future__ import annotations

from abc import ABC
from collections.abc import Iterable


class BaseCallback(ABC):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.n_calls = 0


class CallbackList(BaseCallback):
    def __init__(self, callbacks: Iterable[BaseCallback] | None = None) -> None:
        super().__init__()
        self.callbacks = list(callbacks or [])


class EvalCallback(BaseCallback):
    pass


class CheckpointCallback(BaseCallback):
    pass
