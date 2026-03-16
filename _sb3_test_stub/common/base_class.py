"""Minimal BaseAlgorithm shim."""
from __future__ import annotations

class BaseAlgorithm:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def learn(self, total_timesteps: int, **kwargs: object) -> "BaseAlgorithm":
        return self

    def predict(self, observation: object, deterministic: bool = False) -> tuple[int, None]:
        return 0, None

    def save(self, path: str) -> None:
        return None

    @classmethod
    def load(cls, path: str, *args: object, **kwargs: object) -> "BaseAlgorithm":
        return cls(*args, **kwargs)
