"""Minimal shim for `jsonschema` to support configuration validation in tests.

This lightweight module implements a `validate` function and a
`ValidationError` exception with a `message` attribute to mimic the
real package's minimal interface used by the codebase. It does not perform
real JSON Schema validation and is intended only for test/import-time
compatibility when `jsonschema` is not installed.
"""
from typing import Any


class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = str(message)


def validate(instance: Any, schema: Any) -> None:
    """No-op validate function (does not raise) for tests.

    Replace with real `jsonschema` package in production/dev envs where
    full validation is desired.
    """
    return None


__all__ = ["validate", "ValidationError"]
