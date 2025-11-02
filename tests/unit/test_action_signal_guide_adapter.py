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
        assert adapter.guide.config.enable_harmonic_patterns == True
        assert adapter.guide.config.enable_dow_theory_patterns == False

        # Check that signal generator has harmonic recognizers
        harmonic_names = [
            "GartleyRecognizer",
            "BatRecognizer",
            "ButterflyRecognizer",
            "CrabRecognizer",
        ]
        recognizer_names = [
            r.name for r in adapter.guide.signal_generator.all_recognizers
        ]

        for name in harmonic_names:
            assert name in recognizer_names, f"{name} should be in all_recognizers"

    def test_adapter_with_dow_theory_config(self):
        """Test adapter initializes with dow_theory config."""
        config = ActionSignalGuideConfig(
            enable_harmonic_patterns=False,
            enable_dow_theory_patterns=True,
            debug_short_mode=False,
        )

        adapter = ActionSignalGuideAdapter(config=config)

        # Check that guide has the config
        assert adapter.guide.config.enable_harmonic_patterns == False
        assert adapter.guide.config.enable_dow_theory_patterns == True

        # Check that signal generator has dow theory recognizer
        assert "DowTheoryRecognizer" in [
            r.name for r in adapter.guide.signal_generator.all_recognizers
        ]

        # Check that harmonic recognizers are NOT included
        harmonic_names = [
            "GartleyRecognizer",
            "BatRecognizer",
            "ButterflyRecognizer",
            "CrabRecognizer",
        ]
        recognizer_names = [
            r.name for r in adapter.guide.signal_generator.all_recognizers
        ]

        for name in harmonic_names:
            assert (
                name not in recognizer_names
            ), f"{name} should NOT be in all_recognizers"
