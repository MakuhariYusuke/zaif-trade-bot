"""Lightweight confirmation helper used by trainers for long-running operations.

This module is intentionally small and strongly typed so mypy can import it
without requiring interactive prompts or platform-specific dependencies.
"""

from typing import Optional, Any


def confirm_long_running_operation(message: str, default: bool = False, **kwargs: Any) -> bool:
    """Return user's confirmation for a long-running operation.

    In non-interactive contexts this function returns the provided default.
    It exists primarily to be type-friendly for static analysis and tests.
    """

    # Accept and ignore extra kwargs used by callers (operation_name,
    # estimated_time, risk_description, etc.). In CI or when running
    # non-interactively, prefer the default.
    return default
