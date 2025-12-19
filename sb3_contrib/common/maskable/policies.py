"""Minimal Maskable policy shim used in tests."""

class MaskableActorCriticPolicy:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return None

__all__ = ["MaskableActorCriticPolicy"]
