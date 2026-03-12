"""Minimal logger shim."""


class Logger:
    def record(self, *args, **kwargs):
        return None

    def dump(self, *args, **kwargs):
        return None
