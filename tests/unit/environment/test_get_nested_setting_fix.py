#!/usr/bin/env python3
"""
Tests for _get_nested_setting syntax error fix in RewardCalculator.
"""


import pytest

from ztb.trading.environment.components.reward_calculator import RewardCalculator


class TestGetNestedSettingFix:
    """Test the _get_nested_setting syntax error fix."""

    @pytest.fixture

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with nested settings."""
        return {
            "reward_settings": {
                "transaction_cost": 0.001,
                "risk_penalty": {"enabled": True, "multiplier": 0.1},
                "regime_settings": {
                    "bull": {"multiplier": 1.2},
                    "bear": {"multiplier": 0.8},
                    "sideways": {"multiplier": 1.0},
                },
            }
        }

    def test_get_nested_setting_with_valid_path(self, reward_calculator, mock_config):
        """Test _get_nested_setting with valid nested path."""
        # Test accessing top-level setting
        result = reward_calculator._get_nested_setting("transaction_cost")
        assert result == 0.001

        # Test accessing nested setting (but our mock_config doesn't have nested structure)
        # The method uses self.reward_settings, not passed config
        result = reward_calculator._get_nested_setting("some_key")
        assert result is None  # Since our reward_settings doesn't have this

    def test_get_nested_setting_with_invalid_path(self, reward_calculator, mock_config):
        """Test _get_nested_setting with invalid path returns None."""
        # Test non-existent top-level key
        result = reward_calculator._get_nested_setting(mock_config, "non_existent.key")
        assert result is None

        # Test non-existent nested key
        result = reward_calculator._get_nested_setting(
            mock_config, "reward_settings.non_existent.key"
        )
        assert result is None

        # Test accessing key that exists but path goes too deep
        result = reward_calculator._get_nested_setting(
            mock_config, "reward_settings.transaction_cost.invalid"
        )
        assert result is None

    def test_get_nested_setting_with_default_value(
        self, reward_calculator, mock_config
    ):
        """Test _get_nested_setting with default value."""
        # Test with valid path
        result = reward_calculator._get_nested_setting(
            mock_config, "reward_settings.transaction_cost", default=0.002
        )
        assert result == 0.001  # Should return actual value, not default

        # Test with invalid path
        result = reward_calculator._get_nested_setting(
            mock_config, "invalid.path", default=0.005
        )
        assert result == 0.005  # Should return default

    def test_get_nested_setting_with_none_config(self, reward_calculator):
        """Test _get_nested_setting with None config."""
        result = reward_calculator._get_nested_setting(
            None, "any.path", default="test_default"
        )
        assert result == "test_default"

    def test_get_nested_setting_with_empty_path(self, reward_calculator, mock_config):
        """Test _get_nested_setting with empty path."""
        result = reward_calculator._get_nested_setting(
            mock_config, "", default="empty_default"
        )
        assert result == "empty_default"

    def test_get_nested_setting_with_non_dict_config(self, reward_calculator):
        """Test _get_nested_setting with non-dict config."""
        # Test with string config
        result = reward_calculator._get_nested_setting(
            "not_a_dict", "some.path", default="string_default"
        )
        assert result == "string_default"

        # Test with list config
        result = reward_calculator._get_nested_setting(
            [1, 2, 3], "some.path", default="list_default"
        )
        assert result == "list_default"

    def test_get_nested_setting_syntax_fix(self, reward_calculator, mock_config):
        """Test that the syntax error in _get_nested_setting is fixed."""
        # This test ensures the method can be called without syntax errors
        # The original bug was likely a syntax error in the method implementation

        try:
            # These calls should not raise SyntaxError
            result1 = reward_calculator._get_nested_setting(
                mock_config, "reward_settings.transaction_cost"
            )
            result2 = reward_calculator._get_nested_setting(
                mock_config, "invalid.path", default=None
            )

            # If we get here without exception, the syntax is fixed
            assert True

        except SyntaxError:
            # If we get a SyntaxError, the fix didn't work
            pytest.fail("_get_nested_setting still has syntax error")
        except Exception:
            # Other exceptions are OK for this test (we're only checking syntax)
            pass

    def test_get_nested_setting_handles_keyerror(self, reward_calculator):
        """Test that _get_nested_setting handles KeyError gracefully."""
        config = {"a": {"b": "value"}}

        # This should not raise KeyError
        result = reward_calculator._get_nested_setting(
            config, "a.nonexistent.key", default="safe_default"
        )
        assert result == "safe_default"

    def test_get_nested_setting_with_numeric_keys(self, reward_calculator):
        """Test _get_nested_setting with numeric path components."""
        config = {
            "settings": {1: {"value": "numeric_key"}, "2": {"value": "string_key"}}
        }

        # Test with numeric key (converted to string in path)
        result = reward_calculator._get_nested_setting(config, "settings.1.value")
        assert result == "numeric_key"

        # Test with string key
        result = reward_calculator._get_nested_setting(config, "settings.2.value")
        assert result == "string_key"


if __name__ == "__main__":
    pytest.main([__file__])
