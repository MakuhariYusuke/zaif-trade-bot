import os
from unittest.mock import patch

from ztb.utils.config import ZTBConfig


class TestZTBConfig:
    """Test ZTBConfig functionality."""

    def test_get_existing_env_var(self):
        """Test getting existing environment variable."""
        config = ZTBConfig()
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            assert config.get("TEST_KEY") == "test_value"

    def test_get_missing_env_var_with_default(self):
        """Test getting missing environment variable with default."""
        config = ZTBConfig()
        assert config.get("MISSING_KEY", "default_value") == "default_value"

    def test_get_missing_env_var_without_default(self):
        """Test getting missing environment variable without default."""
        config = ZTBConfig()
        assert config.get("MISSING_KEY") is None

    def test_get_bool_true_values(self):
        """Test boolean parsing with true values."""
        config = ZTBConfig()
        true_values = ["true", "TRUE", "1", "yes", "YES", "on", "ON"]

        for value in true_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                assert config.get_bool("TEST_BOOL") == True

    def test_get_bool_false_values(self):
        """Test boolean parsing with false values."""
        config = ZTBConfig()
        false_values = [
            "false",
            "FALSE",
            "0",
            "no",
            "NO",
            "off",
            "OFF",
            "anything_else",
        ]

        for value in false_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                assert config.get_bool("TEST_BOOL") == False

    def test_get_bool_missing_with_default(self):
        """Test boolean parsing with missing value and default."""
        config = ZTBConfig()
        assert config.get_bool("MISSING_BOOL", True) == True
        assert config.get_bool("MISSING_BOOL", False) == False

    def test_get_int_valid(self):
        """Test integer parsing with valid value."""
        config = ZTBConfig()
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert config.get_int("TEST_INT") == 42

    def test_get_int_invalid(self):
        """Test integer parsing with invalid value."""
        config = ZTBConfig()
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert config.get_int("TEST_INT", 99) == 99

    def test_get_int_missing_with_default(self):
        """Test integer parsing with missing value and default."""
        config = ZTBConfig()
        assert config.get_int("MISSING_INT", 123) == 123

    def test_get_float_valid(self):
        """Test float parsing with valid value."""
        config = ZTBConfig()
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert config.get_float("TEST_FLOAT") == 3.14

    def test_get_float_invalid(self):
        """Test float parsing with invalid value."""
        config = ZTBConfig()
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            assert config.get_float("TEST_FLOAT", 2.71) == 2.71

    def test_get_float_missing_with_default(self):
        """Test float parsing with missing value and default."""
        config = ZTBConfig()
        assert config.get_float("MISSING_FLOAT", 1.23) == 1.23
