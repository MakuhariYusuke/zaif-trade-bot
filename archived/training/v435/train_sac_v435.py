"""Minimal SAC v435 trainer shim used by v435 tests."""


class SACv435Trainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        return {"status": "trained"}


__all__ = ["SACv435Trainer"]
