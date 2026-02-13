"""Minimal callbacks module for tests."""

class BaseCallback:
    def __init__(self, *args, **kwargs):
        self.n_calls = 0




class EvalCallback(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__()


class CheckpointCallback(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__()


__all__ = ["BaseCallback", "CallbackList", "EvalCallback", "CheckpointCallback"]
