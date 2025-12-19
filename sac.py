# Minimal SAC test shim used by legacy tests that import `sac`.
class SACSuite:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        return True

__all__ = ["SACSuite"]
