"""Minimal prometheus_client compatibility shim."""
from __future__ import annotations

class _Metric:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.value = 0.0

    def labels(self, *args: object, **kwargs: object) -> "_Metric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount

    def set(self, value: float) -> None:
        self.value = float(value)

    def observe(self, value: float) -> None:
        self.value = float(value)


class Counter(_Metric):
    pass


class Gauge(_Metric):
    pass


class Histogram(_Metric):
    pass


class Summary(_Metric):
    pass


def start_http_server(port: int, *args: object, **kwargs: object) -> None:
    return None
