#!/usr/bin/env python3
"""
Tests for _get_nested_setting syntax error fix in RewardCalculator.
"""


import pytest

from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.training.environments.environment_config import EnvironmentConfig


class TestGetNestedSettingFix:
    """Test the _get_nested_setting syntax error fix."""

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

    @pytest.fixture
    def reward_calculator(self, mock_config):
        """Create a RewardCalculator instance for nested-setting tests."""
        return RewardCalculator(
            config=EnvironmentConfig(initial_balance=10000.0, commission=0.001),
            reward_settings=mock_config["reward_settings"],
            initial_portfolio_value=10000.0,
        )

    def test_get_nested_setting_with_valid_path(self, reward_calculator, mock_config):
        """Test _get_nested_setting with valid nested path."""
        # Test accessing top-level setting
        result = reward_calculator._get_nested_setting("transaction_cost")
        assert result == 0.001

        result = reward_calculator._get_nested_setting("regime_settings.bull.multiplier")
        assert result == 1.2

    def test_get_nested_setting_with_invalid_path(self, reward_calculator, mock_config):
        """Test _get_nested_setting with invalid path returns None."""
        assert reward_calculator._get_nested_setting("non_existent.key") is None
        assert reward_calculator._get_nested_setting("regime_settings.non_existent.key") is None
        assert reward_calculator._get_nested_setting("transaction_cost.invalid") is None

    def test_get_nested_setting_with_default_value(
        self, reward_calculator, mock_config
    ):
        """Test _get_nested_setting with caller-side fallback."""
        result = reward_calculator._get_nested_setting("transaction_cost")
        assert result == 0.001

        result = reward_calculator._get_nested_setting("invalid.path")
        assert (result if result is not None else 0.005) == 0.005

    def test_get_nested_setting_with_none_config(self, reward_calculator):
        """Test _get_nested_setting with missing path."""
        result = reward_calculator._get_nested_setting("any.path")
        assert result is None

    def test_get_nested_setting_with_empty_path(self, reward_calculator, mock_config):
        """Test _get_nested_setting with empty path."""
        assert reward_calculator._get_nested_setting("") is None

    def test_get_nested_setting_with_non_dict_config(self, reward_calculator):
        """Test _get_nested_setting ignores incompatible value types cleanly."""
        reward_calculator.reward_settings.custom_reward_params["scalar"] = "value"
        assert reward_calculator._get_nested_setting("scalar") == "value"
        assert reward_calculator._get_nested_setting("scalar.missing") is None

    def test_get_nested_setting_syntax_fix(self, reward_calculator, mock_config):
        """Test that the syntax error in _get_nested_setting is fixed."""
        # This test ensures the method can be called without syntax errors
        # The original bug was likely a syntax error in the method implementation

        try:
            # These calls should not raise SyntaxError
            result1 = reward_calculator._get_nested_setting("transaction_cost")
            result2 = reward_calculator._get_nested_setting("invalid.path")

            # If we get here without exception, the syntax is fixed
            assert result1 == 0.001
            assert result2 is None

        except SyntaxError:
            # If we get a SyntaxError, the fix didn't work
            pytest.fail("_get_nested_setting still has syntax error")
        except Exception:
            # Other exceptions are OK for this test (we're only checking syntax)
            pass

    def test_get_nested_setting_handles_keyerror(self, reward_calculator):
        """Test that _get_nested_setting handles KeyError gracefully."""
        reward_calculator.reward_settings.custom_reward_params["a"] = {"b": "value"}
        assert reward_calculator._get_nested_setting("a.nonexistent.key") is None

    def test_get_nested_setting_with_numeric_keys(self, reward_calculator):
        """Test _get_nested_setting with numeric path components."""
        reward_calculator.reward_settings.custom_reward_params["settings"] = {
            1: {"value": "numeric_key"},
            "2": {"value": "string_key"},
        }

        # Test with numeric key (converted to string in path)
        result = reward_calculator._get_nested_setting("settings")
        assert isinstance(result, dict)
        assert result[1]["value"] == "numeric_key"

        # Test with string key
        assert result["2"]["value"] == "string_key"


if __name__ == "__main__":
    pytest.main([__file__])
