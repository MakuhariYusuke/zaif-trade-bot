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
