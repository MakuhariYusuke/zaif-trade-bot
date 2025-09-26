"""
Unit tests for ErrorHandler
"""
import pytest
from unittest.mock import patch
from ztb.utils.error_handler import catch_and_notify


def test_catch_and_notify_no_exception():
    """Test decorator when no exception occurs"""
    @catch_and_notify
    def safe_function():
        return "success"

    result = safe_function()
    assert result == "success"


def test_catch_and_notify_with_exception():
    """Test decorator catches exceptions"""
    @catch_and_notify
    def risky_function():
        raise ValueError("test error")

    # Should not raise exception, handled internally
    risky_function()