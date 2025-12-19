"""Minimal wrappers shim used by tests (ActionMasker).

This implements a tiny `ActionMasker` class compatible with the code under
test. It is not a full implementation, but suffices for unit tests and allows
patching/mocking in tests.
"""

class ActionMasker:
    def __init__(self, env, mask_fn=None):
        self.env = env
        self.mask_fn = mask_fn

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, item):
        return getattr(self.env, item)

__all__ = ["ActionMasker"]
