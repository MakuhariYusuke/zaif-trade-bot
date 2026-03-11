"""Minimal Monitor shim."""
from __future__ import annotations

class Monitor:
    def __init__(self, env: object, filename: str | None = None) -> None:
        self.env = env
        self.filename = filename

    def reset(self) -> object:
        if hasattr(self.env, "reset"):
            return self.env.reset()  # type: ignore[no-any-return, misc]
        return None

    def step(self, action: object) -> object:
        if hasattr(self.env, "step"):
            return self.env.step(action)  # type: ignore[no-any-return, misc]
        return None, 0.0, False, {}
