"""
Unit tests for ActionSignalGuideAdapter.

Tests the adapter initialization and configuration passing.
"""


from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuideConfig,
)


class TestActionSignalGuideAdapter:
    """Test ActionSignalGuideAdapter."""

    def test_adapter_with_harmonic_config(self):
        """Test adapter initializes with harmonic config."""
        config = ActionSignalGuideConfig(
            enable_harmonic_patterns=True,
            enable_dow_theory_patterns=False,
            debug_short_mode=False,
        )

        adapter = ActionSignalGuideAdapter(config=config)

        # Check that guide has the config
        assert adapter.config.enable_harmonic_patterns == True
        assert adapter.config.enable_dow_theory_patterns == False

        # Adapter initialized successfully with config

    def test_adapter_with_dow_theory_config(self):
        """Test adapter initializes with dow_theory config."""
        config = ActionSignalGuideConfig(
            enable_harmonic_patterns=False,
            enable_dow_theory_patterns=True,
            debug_short_mode=False,
        )

        adapter = ActionSignalGuideAdapter(config=config)

        # Check that guide has the config
        assert adapter.config.enable_harmonic_patterns == False
        assert adapter.config.enable_dow_theory_patterns == True

        # Adapter initialized successfully with config
