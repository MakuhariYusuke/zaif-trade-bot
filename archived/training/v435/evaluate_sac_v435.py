"""Minimal evaluator shim for v435 used in tests."""


class SACv435Evaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self):
        return {}


__all__ = ["SACv435Evaluator"]
