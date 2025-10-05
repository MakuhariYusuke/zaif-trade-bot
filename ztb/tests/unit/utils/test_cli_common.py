"""
Unit tests for CLI common utilities
CLI共通ユーティリティの単体テスト
"""

import pytest
from ztb.utils.cli_common import CLIValidator


class TestCLIValidator:
    """Test CLIValidator"""

    def test_validate_positive_int_valid(self):
        """Test validating positive integers"""
        result = CLIValidator.validate_positive_int("5", "test_value")
        assert result == 5

    def test_validate_positive_int_zero(self):
        """Test validating zero (should fail)"""
        with pytest.raises(ValueError, match="test_value must be positive"):
            CLIValidator.validate_positive_int("0", "test_value")

    def test_validate_positive_int_negative(self):
        """Test validating negative integers (should fail)"""
        with pytest.raises(ValueError, match="test_value must be positive"):
            CLIValidator.validate_positive_int("-1", "test_value")

    def test_validate_positive_int_invalid_string(self):
        """Test validating invalid string (should fail)"""
        with pytest.raises(ValueError, match="test_value must be a positive integer"):
            CLIValidator.validate_positive_int("abc", "test_value")

    def test_validate_positive_float_valid(self):
        """Test validating positive floats"""
        result = CLIValidator.validate_positive_float("5.5", "test_value")
        assert result == 5.5

    def test_validate_positive_float_zero(self):
        """Test validating zero float (should fail)"""
        with pytest.raises(ValueError, match="test_value must be positive"):
            CLIValidator.validate_positive_float("0.0", "test_value")

    def test_validate_positive_float_negative(self):
        """Test validating negative floats (should fail)"""
        with pytest.raises(ValueError, match="test_value must be positive"):
            CLIValidator.validate_positive_float("-1.5", "test_value")

    def test_validate_positive_float_invalid_string(self):
        """Test validating invalid string (should fail)"""
        with pytest.raises(ValueError, match="test_value must be a positive float"):
            CLIValidator.validate_positive_float("abc", "test_value")

    def test_validate_path_exists_valid(self, tmp_path):
        """Test validating existing path"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        result = CLIValidator.validate_path_exists(str(test_file), "test_path")
        assert result == test_file

    def test_validate_path_exists_nonexistent(self):
        """Test validating nonexistent path (should fail)"""
        with pytest.raises(ValueError, match="test_path path does not exist"):
            CLIValidator.validate_path_exists("/nonexistent/path", "test_path")

    def test_validate_venue_coincheck(self):
        """Test validating coincheck venue"""
        result = CLIValidator.validate_venue("coincheck")
        assert result == "coincheck"

    def test_validate_venue_bitflyer(self):
        """Test validating bitflyer venue"""
        result = CLIValidator.validate_venue("bitflyer")
        assert result == "bitflyer"

    def test_validate_venue_binance(self):
        """Test validating binance venue"""
        result = CLIValidator.validate_venue("binance")
        assert result == "binance"

    def test_validate_venue_uppercase(self):
        """Test validating venue with uppercase (should convert to lowercase)"""
        result = CLIValidator.validate_venue("BITFLYER")
        assert result == "bitflyer"

    def test_validate_venue_mixed_case(self):
        """Test validating venue with mixed case"""
        result = CLIValidator.validate_venue("BitFlyer")
        assert result == "bitflyer"

    def test_validate_venue_invalid(self):
        """Test validating invalid venue (should fail)"""
        with pytest.raises(ValueError, match="Unsupported venue: invalid_venue"):
            CLIValidator.validate_venue("invalid_venue")

    def test_validate_venue_empty_string(self):
        """Test validating empty string (should fail)"""
        with pytest.raises(ValueError, match="Unsupported venue: "):
            CLIValidator.validate_venue("")

    def test_validate_venue_none_type(self):
        """Test validating None type (should fail with AttributeError)"""
        with pytest.raises(AttributeError):
            CLIValidator.validate_venue(None)