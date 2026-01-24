"""
Deprecated schema module.
"""

from __future__ import annotations

import warnings

from ztb.config.schemas import zaif as _zaif

warnings.warn(
    "ztb.config.schema is deprecated; use ztb.config.schemas.zaif",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_zaif.__all__)
globals().update({name: getattr(_zaif, name) for name in __all__})
