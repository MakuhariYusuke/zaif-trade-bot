#!/usr/bin/env python3
"""
test_config.py
Unit tests for ZTBConfig class
"""

import os
from unittest.mock import patch

from ztb.utils.config import ZTBConfig


class TestZTBConfig:
    """Test ZTBConfig configuration management"""

    def setup_method(self):
        """Clean up environment before each test"""
        test_keys = [
            "ZTB_TEST_BOOL",
            "ZTB_TEST_INT",
            "ZTB_TEST_FLOAT",
            "ZTB_TEST_STR",
            "ZTB_INVALID_FLOAT",
        ]
        for key in test_keys:
            if key in os.environ:
                del os.environ[key]

    def teardown_method(self):
        """Clean up environment after each test"""
        self.setup_method()

    def test_get_existing_value(self):
        """Test getting existing environment variable"""
        with patch.dict(os.environ, {"ZTB_TEST_STR": "test_value"}):
            config = ZTBConfig()
            assert config.get("ZTB_TEST_STR") == "test_value"

    def test_get_default_value(self):
        """Test getting default value when env var not set"""
        config = ZTBConfig()
        assert config.get("ZTB_NON_EXISTENT", "default") == "default"

    def test_get_bool_true_values(self):
        """Test boolean parsing for true values"""
        config = ZTBConfig()
        true_values = ["true", "1", "yes", "on", "TRUE", "YES", "ON"]

        for value in true_values:
            with patch.dict(os.environ, {"ZTB_TEST_BOOL": value}):
                assert config.get_bool("ZTB_TEST_BOOL") is True

    def test_get_bool_false_values(self):
        """Test boolean parsing for false values"""
        config = ZTBConfig()
        false_values = ["false", "0", "no", "off", "anything_else", "FALSE"]

        for value in false_values:
            with patch.dict(os.environ, {"ZTB_TEST_BOOL": value}):
                assert config.get_bool("ZTB_TEST_BOOL") is False

    def test_get_bool_default(self):
        """Test boolean default when env var not set"""
        config = ZTBConfig()
        assert config.get_bool("ZTB_NON_EXISTENT") is False
        assert config.get_bool("ZTB_NON_EXISTENT", True) is True

    def test_get_int_valid_values(self):
        """Test integer parsing for valid values"""
        config = ZTBConfig()

        with patch.dict(os.environ, {"ZTB_TEST_INT": "42"}):
            assert config.get_int("ZTB_TEST_INT") == 42

        with patch.dict(os.environ, {"ZTB_TEST_INT": "-10"}):
            assert config.get_int("ZTB_TEST_INT") == -10

    def test_get_int_invalid_values_fallback(self):
        """Test integer parsing fallback for invalid values"""
        config = ZTBConfig()

        with patch.dict(os.environ, {"ZTB_TEST_INT": "not_a_number"}):
            with patch("builtins.print") as mock_print:
                result = config.get_int("ZTB_TEST_INT", 99)
                assert result == 99
                mock_print.assert_called_once_with(
                    "Warning: Invalid integer value for ZTB_TEST_INT: not_a_number, using default 99"
                )

    def test_get_int_default(self):
        """Test integer default when env var not set"""
        config = ZTBConfig()
        assert config.get_int("ZTB_NON_EXISTENT") == 0
        assert config.get_int("ZTB_NON_EXISTENT", 123) == 123

    def test_get_float_valid_values(self):
        """Test float parsing for valid values"""
        config = ZTBConfig()

        test_cases = [
            ("3.14", 3.14),
            ("-2.5", -2.5),
            ("0.0", 0.0),
            ("1e-5", 1e-5),
            ("42", 42.0),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ZTB_TEST_FLOAT": env_value}):
                assert config.get_float("ZTB_TEST_FLOAT") == expected

    def test_get_float_invalid_values_fallback(self):
        """Test float parsing fallback for invalid values"""
        config = ZTBConfig()

        invalid_values = [
            "not_a_number",
            "3.14.15",
            "abc",
            "",
            "null",
        ]

        for invalid_value in invalid_values:
            with patch.dict(os.environ, {"ZTB_INVALID_FLOAT": invalid_value}):
                with patch("builtins.print") as mock_print:
                    result = config.get_float("ZTB_INVALID_FLOAT", 99.5)
                    assert result == 99.5
                    mock_print.assert_called_once_with(
                        f"Warning: Invalid float value for ZTB_INVALID_FLOAT: {invalid_value}, using default 99.5"
                    )

    def test_get_float_default(self):
        """Test float default when env var not set"""
        config = ZTBConfig()
        assert config.get_float("ZTB_NON_EXISTENT") == 0.0
        assert config.get_float("ZTB_NON_EXISTENT", 123.45) == 123.45

    def test_get_edge_cases_invalid_values(self):
        """Test edge cases with invalid environment variable values"""
        config = ZTBConfig()

        # Test empty string
        with patch.dict(os.environ, {"ZTB_EMPTY_STR": ""}):
            with patch("builtins.print") as mock_print:
                result = config.get_int("ZTB_EMPTY_STR", 42)
                assert result == 42
                mock_print.assert_called_once()

        # Test whitespace-only string
        with patch.dict(os.environ, {"ZTB_WHITESPACE": "   "}):
            with patch("builtins.print") as mock_print:
                result = config.get_float("ZTB_WHITESPACE", 3.14)
                assert result == 3.14
                mock_print.assert_called_once()

        # Test very large numbers
        with patch.dict(os.environ, {"ZTB_HUGE_INT": "999999999999999999999999999999"}):
            result = config.get_int("ZTB_HUGE_INT", 0)
            # Should handle large integers if valid
            assert isinstance(result, int)

        # Test scientific notation for float
        with patch.dict(os.environ, {"ZTB_SCI_FLOAT": "1.23e-4"}):
            result = config.get_float("ZTB_SCI_FLOAT")
            assert abs(result - 0.000123) < 1e-6

    def test_get_mixed_case_boolean_values(self):
        """Test boolean parsing with mixed case values"""
        config = ZTBConfig()

        mixed_case_true = ["True", "TRUE", "True", "YES", "Yes", "ON", "On"]
        mixed_case_false = ["False", "FALSE", "False", "NO", "No", "OFF", "Off"]

        for value in mixed_case_true:
            with patch.dict(os.environ, {"ZTB_MIXED_BOOL": value}):
                assert config.get_bool("ZTB_MIXED_BOOL") is True

        for value in mixed_case_false:
            with patch.dict(os.environ, {"ZTB_MIXED_BOOL": value}):
                assert config.get_bool("ZTB_MIXED_BOOL") is False

    def test_get_numeric_edge_cases(self):
        """Test numeric parsing edge cases"""
        config = ZTBConfig()

        # Test float that looks like int
        with patch.dict(os.environ, {"ZTB_FLOAT_INT": "42.0"}):
            assert config.get_float("ZTB_FLOAT_INT") == 42.0
            assert config.get_int("ZTB_FLOAT_INT", 0) == 42  # Should truncate

        # Test negative zero
        with patch.dict(os.environ, {"ZTB_NEG_ZERO": "-0"}):
            assert config.get_int("ZTB_NEG_ZERO") == 0
            assert config.get_float("ZTB_NEG_ZERO") == 0.0

        # Test leading/trailing whitespace
        with patch.dict(os.environ, {"ZTB_SPACED_NUM": "  123  "}):
            assert config.get_int("ZTB_SPACED_NUM", 0) == 123

        with patch.dict(os.environ, {"ZTB_SPACED_FLOAT": "  45.67  "}):
            assert config.get_float("ZTB_SPACED_FLOAT", 0.0) == 45.67

    def test_log_config(self):
        """Test configuration logging"""
        config = ZTBConfig()

        test_config = {
            "ZTB_MEM_PROFILE": "1",
            "ZTB_CUDA_WARN_GB": "8.0",
            "ZTB_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, test_config):
            with patch("builtins.print") as mock_print:
                config.log_config()
                # Verify that print was called (exact output may vary)
                assert mock_print.called
                call_args = str(mock_print.call_args_list)
                assert "ZTB_MEM_PROFILE" in call_args
                assert "ZTB_CUDA_WARN_GB" in call_args
                assert "ZTB_LOG_LEVEL" in call_args
