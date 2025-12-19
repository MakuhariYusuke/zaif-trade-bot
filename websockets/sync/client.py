"""Tiny sync client shim providing `connect` to satisfy `yfinance`.

This is intentionally minimal: it returns a context manager with a no-op
`recv` method and closes cleanly. It is only for import-time compatibility
during tests and should not be used for any real networking.
"""
from contextlib import contextmanager


class _NoopWS:
    def __init__(self, *args, **kwargs):
        pass

    async def recv(self):
        return None

    def close(self):
        return None


@contextmanager
def connect(*args, **kwargs):
    ws = _NoopWS()
    try:
        yield ws
    finally:
        ws.close()
from contextlib import contextmanager


@contextmanager
def connect(*args, **kwargs):
    """Sync context manager stub that yields a dummy connection object."""
    class _Conn:
        def send(self, *a, **k):
            return None

        def recv(self):
            return None

        def close(self):
            return None

    conn = _Conn()
    try:
        yield conn
    finally:
        conn.close()
"""Minimal sync client for websockets used by yfinance during imports.

This provides a `connect` function that can be used as a context manager and
yields a stub connection object with `send` and `recv` methods. It is NOT a
replacement for the real library; it's only for test-time import safety.
"""
from contextlib import contextmanager


class _DummyConnection:
    def send(self, message):
        return None

    def recv(self):
        return None


@contextmanager
def connect(*args, **kwargs):
    conn = _DummyConnection()
    yield conn
