"""Compatibility shim for canonical parameter adaptation helpers.

Canonical implementations now live in `ztb.trading.sizing.param_adapter`.
This shim intentionally keeps the theory references visible because some
v460 review/tests still inspect this module docstring for:

- Avellaneda-Stoikov
- Glosten-Milgrom

Keep the runtime surface thin while preserving that compatibility.
"""

from ztb.trading.sizing.param_adapter import *  # noqa: F401,F403
