"""Common submodule shims for stable_baselines3 compatibility."""

from .base_class import BaseAlgorithm
from .callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from .monitor import Monitor
from .policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from .type_aliases import GymEnv, Schedule, TensorDict
from .vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecNormalize

__all__ = [
    "BaseAlgorithm",
    "BaseCallback",
    "CallbackList",
    "CheckpointCallback",
    "EvalCallback",
    "Monitor",
    "ActorCriticPolicy",
    "MultiInputActorCriticPolicy",
    "GymEnv",
    "Schedule",
    "TensorDict",
    "DummyVecEnv",
    "VecEnv",
    "VecFrameStack",
    "VecNormalize",
]
