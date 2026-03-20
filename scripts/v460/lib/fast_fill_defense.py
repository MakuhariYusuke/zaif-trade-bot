"""Compatibility shim for canonical fast-fill defense helpers.

Canonical implementations now live in `ztb.trading.risk.fast_fill_defense`.
This shim keeps the familiar import path for `run_fill_test`, fixtures,
and broad v460 tests while we migrate call sites gradually.
"""

import time

from ztb.trading.risk.fast_fill_defense import (  # noqa: F401
    FastFillDefense,
    FastFillDefenseConfig,
    _SideState,
)
