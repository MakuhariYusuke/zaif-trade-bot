"""Minimal Monitor implementation for tests."""

class Monitor:

    def reset(self):
        return None

    def step(self, action):
        return None, 0, False, {}

__all__ = ["Monitor"]
