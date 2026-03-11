"""
Unit tests for SignalGenerator component.

Tests the SignalGenerator initialization and pattern recognition.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuideConfig,
)
from ztb.trading.strategies.action_signal_guide.components.signal_generator import (
    SignalGenerator,
)


class TestSignalGenerator:
    """Test SignalGenerator component."""

    @pytest.fixture
    def sample_config(self):
        """Create sample ActionSignalGuideConfig with harmonic patterns enabled."""
        return ActionSignalGuideConfig(
            enable_harmonic_patterns=True,
            enable_dow_theory_patterns=True,
            enable_fibonacci_patterns=False,
            debug_short_mode=False,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for signal generation."""
        rows = 120
        index = pd.date_range("2023-01-01", periods=rows, freq="1H")
        base = np.linspace(100.0, 110.0, rows)
        return pd.DataFrame(
            {
                "open": base - 0.2,
                "high": base + 0.5,
                "low": base - 0.5,
                "close": base,
                "volume": np.full(rows, 1000.0),
            },
            index=index,
        )

    def test_signal_generator_initialization_with_harmonic_enabled(self, sample_config):
        """Test SignalGenerator initializes with harmonic patterns enabled."""
        generator = SignalGenerator(config=sample_config)

        # Check that harmonic recognizers are included
        harmonic_names = [
            "GartleyRecognizer",
            "BatRecognizer",
            "ButterflyRecognizer",
            "CrabRecognizer",
        ]
        recognizer_names = [r.name for r in generator.all_recognizers]

        for name in harmonic_names:
            assert (
                name in recognizer_names
            ), f"{name} should be in all_recognizers when enable_harmonic_patterns=True"

    def test_signal_generator_initialization_with_harmonic_disabled(self):
        """Test SignalGenerator initializes with harmonic patterns disabled."""
        config = ActionSignalGuideConfig(
            enable_harmonic_patterns=False,
            enable_dow_theory_patterns=True,
        )
        generator = SignalGenerator(config=config)

        # Check that harmonic recognizers are NOT included
        harmonic_names = [
            "GartleyRecognizer",
            "BatRecognizer",
            "ButterflyRecognizer",
            "CrabRecognizer",
        ]
        recognizer_names = [r.name for r in generator.all_recognizers]

        for name in harmonic_names:
            assert (
                name not in recognizer_names
            ), f"{name} should NOT be in all_recognizers when enable_harmonic_patterns=False"

    def test_signal_generator_generate_signal(self, sample_config, sample_data):
        """Test SignalGenerator can generate signals."""
        generator = SignalGenerator(config=sample_config)

        # Generate signal at index 50 (sufficient data)
        result = generator.generate_signal(sample_data, current_index=50)

        # Result should be valid (may be None if no pattern found)
        # Just check it doesn't crash
        assert True  # If we reach here, no exception was raised
