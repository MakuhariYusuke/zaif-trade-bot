#!/usr/bin/env python3
"""
test_test_utils.py
Unit tests for test utility functions
"""

import time
from unittest.mock import patch

import pytest

from ztb.tests.utils.test_utils import timed_test


class TestTimedTestDecorator:
    """Test @timed_test decorator functionality"""

    def test_timed_test_fast_success(self):
        """Test decorator with fast successful test"""

        @timed_test
        def fast_test():
            return "success"

        with patch("builtins.print") as mock_print:
            result = fast_test()
            assert result == "success"
            # Should not print slow test warning for fast tests
            assert not any(
                "SLOW TEST" in str(call) for call in mock_print.call_args_list
            )

    def test_timed_test_slow_success(self):
        """Test decorator with slow successful test"""

        @timed_test
        def slow_test():
            time.sleep(6)  # Exceed 5 second threshold
            return "slow_success"

        with patch("builtins.print") as mock_print:
            result = slow_test()
            assert result == "slow_success"
            # Should print slow test warning
            assert len(mock_print.call_args_list) == 1
            call_args = str(mock_print.call_args_list[0])
            assert "SLOW TEST: slow_test took" in call_args
            assert "6.0" in call_args

    def test_timed_test_failure(self):
        """Test decorator with failing test"""

        @timed_test
        def failing_test():
            time.sleep(2)
            raise AssertionError("Test failed")

        with patch("builtins.print") as mock_print:
            with pytest.raises(AssertionError, match="Test failed"):
                failing_test()

            # Should print failed test message
            assert len(mock_print.call_args_list) == 1
            call_args = str(mock_print.call_args_list[0])
            assert "FAILED TEST: failing_test failed after" in call_args
            assert "2.0" in call_args

    def test_timed_test_exception_timing(self):
        """Test that timing works correctly even when exception occurs"""

        @timed_test
        def exception_test():
            time.sleep(1.5)
            raise ValueError("Something went wrong")

        with patch("builtins.print") as mock_print:
            with pytest.raises(ValueError, match="Something went wrong"):
                exception_test()

            call_args = str(mock_print.call_args_list[0])
            assert "FAILED TEST: exception_test failed after" in call_args
            assert "1.5" in call_args

    def test_timed_test_boundary_slow(self):
        """Test boundary case - exactly at slow threshold"""

        @timed_test
        def boundary_test():
            time.sleep(5.1)  # Just over 5 seconds
            return "boundary"

        with patch("builtins.print") as mock_print:
            result = boundary_test()
            assert result == "boundary"
            # Should print slow test warning
            call_args = str(mock_print.call_args_list[0])
            assert "SLOW TEST: boundary_test took" in call_args

    def test_timed_test_boundary_fast(self):
        """Test boundary case - just under slow threshold"""

        @timed_test
        def fast_boundary_test():
            time.sleep(4.9)  # Just under 5 seconds
            return "fast_boundary"

        with patch("builtins.print") as mock_print:
            result = fast_boundary_test()
            assert result == "fast_boundary"
            # Should not print slow test warning
            assert not any(
                "SLOW TEST" in str(call) for call in mock_print.call_args_list
            )

    def test_timed_test_with_arguments(self):
        """Test decorator preserves function arguments and return values"""

        @timed_test
        def test_with_args(a, b, c=None):
            time.sleep(0.1)
            return a + b + (c or 0)

        with patch("builtins.print") as mock_print:
            result = test_with_args(1, 2, c=3)
            assert result == 6
            # Should not print slow test warning
            assert not any(
                "SLOW TEST" in str(call) for call in mock_print.call_args_list
            )

    def test_timed_test_function_name_preservation(self):
        """Test that function name is correctly reported in messages"""

        @timed_test
        def unique_test_name_123():
            time.sleep(6)
            return "done"

        with patch("builtins.print") as mock_print:
            unique_test_name_123()
            call_args = str(mock_print.call_args_list[0])
            assert "unique_test_name_123" in call_args

    def test_timed_test_multiple_calls(self):
        """Test decorator works correctly across multiple calls"""
        call_count = 0

        @timed_test
        def multi_call_test():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.1)  # Fast
            else:
                time.sleep(6)  # Slow
            return f"call_{call_count}"

        with patch("builtins.print") as mock_print:
            # First call - fast
            result1 = multi_call_test()
            assert result1 == "call_1"
            assert len(mock_print.call_args_list) == 0

            # Second call - slow
            result2 = multi_call_test()
            assert result2 == "call_2"
            assert len(mock_print.call_args_list) == 1
            call_args = str(mock_print.call_args_list[0])
            assert "SLOW TEST: multi_call_test took" in call_args
