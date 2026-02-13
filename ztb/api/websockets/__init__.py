"""Lightweight websockets package stub used to satisfy `yfinance` imports
during test collection. Only `sync.client.connect` is provided as a tiny
no-op callable.
"""
__all__ = ["sync"]
"""Minimal shim of the `websockets` package used by yfinance during imports.

This only implements the small surface the tests need (exceptions and
sync/async client connect functions). It is intentionally tiny and safe for
test-collection time imports.
"""
from . import exceptions

__all__ = ["exceptions"]
"""Minimal websockets shim used to satisfy optional imports (e.g., yfinance).

This provides a very small subset of the real websockets package used during
test collection so tests that import yfinance don't fail when websockets isn't
installed in the environment.
"""
from importlib import util

# Mark this as a package
__all__ = ["sync", "exceptions"]
