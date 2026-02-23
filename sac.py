"""Compatibility entrypoint for legacy `from sac import SACSuite` imports."""

from ztb.training.sac import SACSuite

__all__ = ["SACSuite"]
