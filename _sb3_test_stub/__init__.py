"""Lightweight stable_baselines3 compatibility package for test environments."""
from __future__ import annotations

from stable_baselines3.common.base_class import BaseAlgorithm


class _DummyAlgo(BaseAlgorithm):
    """Minimal algorithm shim used for import-time compatibility."""

    def __init__(
        self, policy: object | None = None, env: object | None = None, **kwargs: object
    ) -> None:
        super().__init__(policy=policy, env=env, **kwargs)
        self.policy = policy
        self.env = env
        self.kwargs = kwargs

    def learn(self, total_timesteps: int, **kwargs: object) -> "_DummyAlgo":
        return self

    @classmethod
    def load(
        cls, path: str, env: object | None = None, **kwargs: object
    ) -> "_DummyAlgo":
        return cls(env=env, **kwargs)


class SAC(_DummyAlgo):
    pass


class PPO(_DummyAlgo):
    pass


class A2C(_DummyAlgo):
    pass


class DQN(_DummyAlgo):
    pass


class TD3(_DummyAlgo):
    pass


__all__ = ["SAC", "PPO", "A2C", "DQN", "TD3"]
