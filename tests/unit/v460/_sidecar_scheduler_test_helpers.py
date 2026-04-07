from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


def make_shutdown_wait(*, set_after: int = 2, shutdown_event: object) -> object:
    """N 回目の wait で shutdown する side effect."""
    counter = {"calls": 0}

    def _wait(*_args: object, **_kwargs: object) -> bool:
        counter["calls"] += 1
        if counter["calls"] >= set_after:
            getattr(shutdown_event, "set")()
            return True
        return False

    return _wait


@contextmanager
def patch_noop_paths(*paths: str) -> Iterator[None]:
    """指定パスを `return_value=None` の patch にまとめる."""
    with ExitStack() as stack:
        for path in paths:
            stack.enter_context(patch(path, return_value=None))
        yield


@contextmanager
def patch_module_noop_suffixes(module_prefix: str, *suffixes: str) -> Iterator[None]:
    """共通 prefix + suffix 群を no-op patch する."""
    with patch_noop_paths(*(f"{module_prefix}.{suffix}" for suffix in suffixes)):
        yield
