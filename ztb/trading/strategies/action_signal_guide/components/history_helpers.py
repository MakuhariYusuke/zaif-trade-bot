"""
Shared history helpers for Action Signal Guide components.

Provides reusable bounded-append behavior with compaction semantics.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def append_with_compaction(
    history: list[T],
    value: T,
    *,
    high_water: int,
    retain: int,
) -> None:
    """Append into history and compact once high-water threshold is exceeded."""
    if retain <= 0:
        raise ValueError("retain must be greater than 0")
    if high_water < retain:
        high_water = retain

    history.append(value)
    if len(history) > high_water:
        del history[:-retain]
