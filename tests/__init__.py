"""Top-level tests package to support intra-test relative imports during
py.test collection. Adding __init__ avoids import errors when tests use
package-relative imports (e.g., from ..performance import foo).
"""
"""Unit tests package."""

__all__ = []
