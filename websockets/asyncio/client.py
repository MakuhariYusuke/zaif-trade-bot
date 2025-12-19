from contextlib import asynccontextmanager


@asynccontextmanager
async def connect(*args, **kwargs):
    """Async context manager stub that yields a dummy connection object.

    The object does not implement real websocket functionality — it's only
    present to allow third-party modules to import and call `connect` during
    test collection.
    """
    class _Conn:
        async def send(self, *a, **k):
            return None

        async def recv(self):
            return None

        async def close(self):
            return None

    conn = _Conn()
    try:
        yield conn
    finally:
        await conn.close()
"""Minimal asyncio client for websockets used by yfinance during imports.

Provides an async `connect` coroutine that can be used with `async with`.
"""
import asyncio
from contextlib import asynccontextmanager


class _DummyAsyncConnection:
    async def send(self, message):
        return None

    async def recv(self):
        await asyncio.sleep(0)
        return None


@asynccontextmanager
async def connect(*args, **kwargs):
    conn = _DummyAsyncConnection()
    try:
        yield conn
    finally:
        return
