"""
Unit tests for backtest configuration functions.

Tests the configuration setup for pattern backtesting.
"""


from backtest.config import get_backtest_config_for_pattern, validate_pattern_name
from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuideConfig,
)


class TestBacktestConfig:
    """Test backtest configuration functions."""

    def test_validate_pattern_name_valid(self):
        """Test validate_pattern_name with valid patterns."""
        valid_patterns = ["harmonic", "dow_theory", "fibonacci", "oscillator"]
        for pattern in valid_patterns:
            assert validate_pattern_name(pattern) == True

    def test_validate_pattern_name_invalid(self):
        """Test validate_pattern_name with invalid pattern."""
        assert validate_pattern_name("invalid_pattern") == False

    def test_get_backtest_config_for_pattern_harmonic(self):
        """Test get_backtest_config_for_pattern with harmonic."""
        config = get_backtest_config_for_pattern("harmonic")

        assert isinstance(config, ActionSignalGuideConfig)
        assert config.enable_harmonic_patterns == True
        # Other patterns should be disabled
        assert config.enable_fibonacci_patterns == False
        assert config.enable_dow_theory_patterns == False

    def test_get_backtest_config_for_pattern_dow_theory(self):
        """Test get_backtest_config_for_pattern with dow_theory."""
        config = get_backtest_config_for_pattern("dow_theory")

        assert isinstance(config, ActionSignalGuideConfig)
        assert config.enable_dow_theory_patterns == True
        # Other patterns should be disabled
        assert config.enable_harmonic_patterns == False
        assert config.enable_fibonacci_patterns == False

    def test_get_backtest_config_for_pattern_none(self):
        """Test get_backtest_config_for_pattern with None."""
        config = get_backtest_config_for_pattern(None)

        assert isinstance(config, ActionSignalGuideConfig)
        # All patterns should be disabled
        assert config.enable_harmonic_patterns == False
        assert config.enable_dow_theory_patterns == False
        assert config.enable_fibonacci_patterns == False
