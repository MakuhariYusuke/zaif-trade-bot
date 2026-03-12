"""Minimal vectorized environment shims."""
from __future__ import annotations

from collections.abc import Callable


class DummyVecEnv:
    def __init__(self, env_fns: list[Callable[[], object]]) -> None:
        self.env_fns = env_fns
        self.envs = [fn() for fn in env_fns] if env_fns else []

    def reset(self) -> object:
        if self.envs and hasattr(self.envs[0], "reset"):
            return self.envs[0].reset()  # type: ignore[no-any-return, misc]
        return None

    def step(self, action: object) -> tuple[object, float, bool, dict[str, object]]:
        if self.envs and hasattr(self.envs[0], "step"):
            result = self.envs[0].step(action)  # type: ignore[misc]
            if isinstance(result, tuple) and len(result) == 4:
                obs, reward, done, info = result
                info_obj = info if isinstance(info, dict) else {}
                return obs, float(reward), bool(done), dict(info_obj)
        return None, 0.0, False, {}


class VecEnv(DummyVecEnv):
    pass


class VecFrameStack:
    def __init__(self, env: object, n_stack: int = 1) -> None:
        self.env = env
        self.n_stack = n_stack


class VecNormalize:
    def __init__(self, env: object, **kwargs: object) -> None:
        self.env = env
        self.kwargs = kwargs
