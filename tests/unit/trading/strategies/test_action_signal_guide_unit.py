"""
Unit tests for ActionSignalGuide.

Tests the ActionSignalGuide initialization and configuration passing.
"""


from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
)


class TestActionSignalGuide:
    """Test ActionSignalGuide."""

    def test_action_signal_guide_with_harmonic_config(self):
        """Test ActionSignalGuide initializes with harmonic config."""
        config = ActionSignalGuideConfig(
            enable_harmonic_patterns=True,
            enable_dow_theory_patterns=False,
            debug_short_mode=False,
        )

        guide = ActionSignalGuide(config=config)

        # Check that config is set
        assert guide.config.enable_harmonic_patterns == True
        assert guide.config.enable_dow_theory_patterns == False

        # Check that signal generator has harmonic recognizers
        harmonic_names = [
            "GartleyRecognizer",
            "BatRecognizer",
            "ButterflyRecognizer",
            "CrabRecognizer",
        ]
        recognizer_names = [r.name for r in guide.signal_generator.all_recognizers]

        for name in harmonic_names:
            assert name in recognizer_names, f"{name} should be in all_recognizers"

    def test_action_signal_guide_with_dow_theory_config(self):
        """Test ActionSignalGuide initializes with dow_theory config."""
        config = ActionSignalGuideConfig(
            enable_harmonic_patterns=False,
            enable_dow_theory_patterns=True,
            debug_short_mode=False,
        )

        guide = ActionSignalGuide(config=config)

        # Check that config is set
        assert guide.config.enable_harmonic_patterns == False
        assert guide.config.enable_dow_theory_patterns == True

        # Check that signal generator has dow theory recognizer
        assert "DowTheoryRecognizer" in [
            r.name for r in guide.signal_generator.all_recognizers
        ]

        # Check that harmonic recognizers are NOT included
        harmonic_names = [
            "GartleyRecognizer",
            "BatRecognizer",
            "ButterflyRecognizer",
            "CrabRecognizer",
        ]
        recognizer_names = [r.name for r in guide.signal_generator.all_recognizers]

        for name in harmonic_names:
            assert (
                name not in recognizer_names
            ), f"{name} should NOT be in all_recognizers"
