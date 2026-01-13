"""Minimal vec_env implementations used in tests."""

class DummyVecEnv:
    def __init__(self, env_fns):
        self.envs = env_fns

    def reset(self):
        return None

    def step(self, action):
        return None, 0, False, {}


class VecFrameStack:
    pass


class VecNormalize:
    def __init__(self, *args, **kwargs):
        pass


__all__ = ["DummyVecEnv", "VecFrameStack", "VecNormalize"]

# Backwards-compatible alias sometimes expected by older code
class VecEnv(DummyVecEnv):
    """Compatibility alias for code that imports VecEnv from SB3."""
    pass

__all__.append("VecEnv")
