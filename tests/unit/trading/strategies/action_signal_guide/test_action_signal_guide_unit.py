"""
Unit tests for ActionSignalGuide.

This file contains pure unit-level tests (config, recognizer inclusion, and basic initialization checks).
"""

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
)


def test_action_signal_guide_with_harmonic_config():
    config = ActionSignalGuideConfig(
        enable_harmonic_patterns=True,
        enable_dow_theory_patterns=False,
        debug_short_mode=False,
    )

    guide = ActionSignalGuide(config=config)

    assert guide.config.enable_harmonic_patterns is True
    assert guide.config.enable_dow_theory_patterns is False

    harmonic_names = [
        "GartleyRecognizer",
        "BatRecognizer",
        "ButterflyRecognizer",
        "CrabRecognizer",
    ]
    recognizer_names = [r.name for r in guide.signal_generator.all_recognizers]

    for name in harmonic_names:
        assert name in recognizer_names


def test_action_signal_guide_with_dow_theory_config():
    config = ActionSignalGuideConfig(
        enable_harmonic_patterns=False,
        enable_dow_theory_patterns=True,
        debug_short_mode=False,
    )

    guide = ActionSignalGuide(config=config)

    assert guide.config.enable_harmonic_patterns is False
    assert guide.config.enable_dow_theory_patterns is True

    assert "DowTheoryRecognizer" in [
        r.name for r in guide.signal_generator.all_recognizers
    ]

    harmonic_names = [
        "GartleyRecognizer",
        "BatRecognizer",
        "ButterflyRecognizer",
        "CrabRecognizer",
    ]
    recognizer_names = [r.name for r in guide.signal_generator.all_recognizers]
    for name in harmonic_names:
        assert name not in recognizer_names
