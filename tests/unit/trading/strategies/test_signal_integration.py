"""
Test Signal Integration - Unit tests for signal integration functionality.

Tests the integration between technical signals and reward functions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ztb.trading.constants import ACTION_BUY, ACTION_SELL
from ztb.trading.strategies import (
    ActionSignalGuide,
    GuidanceMode,
    SignalDefinitions,
    SignalIntegration,
    SignalType,
)


class TestSignalIntegration:
    """Test cases for signal integration functionality."""

    @pytest.fixture
    def mock_base_reward_function(self):
        """Mock base reward function that returns a fixed value."""
        return Mock(return_value=1.0)

    @pytest.fixture
    def signal_guide(self):
        """Create a signal guide for testing."""
        guide = ActionSignalGuide(mode=GuidanceMode.FULL_GUIDANCE)
        # Set dummy feature names
        guide.set_feature_names(["close", "rsi", "macd", "bb_upper", "bb_lower"])
        return guide

    @pytest.fixture
    def signal_integration(self, signal_guide, mock_base_reward_function):
        """Create signal integration for testing."""
        return SignalIntegration(
            signal_guide=signal_guide,
            base_reward_function=mock_base_reward_function,
            signal_bonus_weight=0.1,
            signal_penalty_weight=0.05,
        )

    def test_signal_integration_creation(self, signal_integration):
        """Test that signal integration is created properly."""
        assert signal_integration.signal_guide is not None
        assert signal_integration.base_reward_function is not None
        assert signal_integration.signal_bonus_weight == 0.1
        assert signal_integration.signal_penalty_weight == 0.05

    def test_signal_integration_with_buy_signal(self, signal_integration):
        """Test signal integration when BUY signal aligns with action."""
        # Create observation with strong BUY signal (RSI oversold: < 30, MACD bullish: > 0)
        observation = np.array(
            [100.0, 25.0, 0.5, 105.0, 95.0]
        )  # close, rsi, macd, bb_upper, bb_lower

        reward = signal_integration.integrated_reward_function(
            observation=observation,
            action=ACTION_BUY,  # BUY action
            reward=1.0,
            next_observation=observation,
            done=False,
            info={},
            step=0,
        )

        # Should get bonus for aligning with BUY signal
        assert reward > 1.0

    def test_signal_integration_with_sell_signal(self, signal_integration):
        """Test signal integration when SELL signal aligns with action."""
        # Create observation with strong SELL signal (RSI overbought: > 70, MACD bearish: < 0)
        observation = np.array(
            [110.0, 75.0, -0.5, 105.0, 95.0]
        )  # close, rsi, macd, bb_upper, bb_lower

        reward = signal_integration.integrated_reward_function(
            observation=observation,
            action=ACTION_SELL,  # SELL action
            reward=1.0,
            next_observation=observation,
            done=False,
            info={},
            step=0,
        )

        # Should get bonus for aligning with SELL signal
        assert reward > 1.0

    def test_signal_integration_penalty(self, signal_integration):
        """Test signal integration penalty when action contradicts signal."""
        # Create observation with strong SELL signal (RSI overbought, MACD bearish)
        observation = np.array([110.0, 75.0, -0.5, 105.0, 95.0])

        reward = signal_integration.integrated_reward_function(
            observation=observation,
            action=ACTION_BUY,  # BUY action (contradicts SELL signal)
            reward=1.0,
            next_observation=observation,
            done=False,
            info={},
            step=0,
        )

        # Should get penalty for contradicting SELL signal
        assert reward < 1.0

    def test_signal_integration_neutral_signal(self, signal_integration):
        """Test signal integration with neutral signals."""
        # Create observation with neutral signals
        observation = np.array([100.0, 50.0, 0.0, 105.0, 95.0])  # neutral signals

        reward = signal_integration.integrated_reward_function(
            observation=observation,
            action=1,  # BUY action
            reward=1.0,
            next_observation=observation,
            done=False,
            info={},
            step=0,
        )

        # Should be close to base reward (no strong signal alignment)
        assert abs(reward - 1.0) < 0.1

    def test_guidance_mode_changes(self, signal_guide):
        """Test that guidance modes change behavior."""
        # Test FULL_GUIDANCE
        signal_guide.update_guidance_mode(GuidanceMode.FULL_GUIDANCE)
        assert signal_guide.mode == GuidanceMode.FULL_GUIDANCE
        assert signal_guide.signal_threshold == 0.3

        # Test MINIMAL_GUIDANCE
        signal_guide.update_guidance_mode(GuidanceMode.MINIMAL_GUIDANCE)
        assert signal_guide.mode == GuidanceMode.MINIMAL_GUIDANCE
        assert signal_guide.signal_threshold == 0.7

        # Test NO_GUIDANCE
        signal_guide.update_guidance_mode(GuidanceMode.NO_GUIDANCE)
        assert signal_guide.mode == GuidanceMode.NO_GUIDANCE
        assert signal_guide.max_signal_strength == 0.0

    def test_signal_guide_stats(self, signal_guide):
        """Test signal guide statistics."""
        stats = signal_guide.get_guidance_stats()
        assert "mode" in stats
        assert "signal_weight" in stats
        assert "signal_threshold" in stats
        assert "max_signal_strength" in stats
        assert "guidance_decay" in stats
        assert "num_features" in stats
        assert "available_signals" in stats

    def test_integration_stats(self, signal_integration):
        """Test integration statistics tracking."""
        # Initially should be zero
        stats = signal_integration.get_integration_stats()
        assert stats["total_steps"] == 0
        assert stats["signal_bonuses_applied"] == 0
        assert stats["signal_penalties_applied"] == 0

        # After some operations
        observation = np.array([100.0, 70.0, -0.5, 105.0, 95.0])
        signal_integration.integrated_reward_function(
            observation=observation,
            action=1,
            reward=1.0,
            next_observation=observation,
            done=False,
            info={},
            step=1,
        )

        stats = signal_integration.get_integration_stats()
        assert stats["total_steps"] == 1
        assert (
            stats["signal_bonuses_applied"] >= 0
        )  # May be 0 or 1 depending on signal strength

    def test_signal_definitions_creation(self):
        """Test that SignalDefinitions is created properly."""
        signals = SignalDefinitions()
        signal_names = signals.get_signal_names()
        assert len(signal_names) > 0

        # Test signal types
        buy_signals = signals.get_signals_by_type(SignalType.BUY)
        sell_signals = signals.get_signals_by_type(SignalType.SELL)
        neutral_signals = signals.get_signals_by_type(SignalType.NEUTRAL)

        assert len(buy_signals) > 0
        assert len(sell_signals) > 0
        assert len(neutral_signals) >= 0  # Neutral might be empty

        # Total should equal all signals
        assert len(buy_signals) + len(sell_signals) + len(neutral_signals) == len(
            signal_names
        )

    def test_signal_evaluation(self):
        """Test individual signal evaluation."""
        signals = SignalDefinitions()
        feature_names = ["close", "rsi_14", "macd", "bb_upper", "bb_lower"]

        # Test with BUY signal data (RSI oversold)
        buy_observation = np.array([100.0, 25.0, -0.5, 105.0, 95.0])
        sig_type, strength = signals.evaluate_signal(
            "rsi_oversold", buy_observation, feature_names
        )

        assert sig_type == SignalType.BUY
        assert strength > 0

        # Test with SELL signal data (RSI overbought)
        sell_observation = np.array([110.0, 75.0, 0.5, 105.0, 95.0])
        sig_type, strength = signals.evaluate_signal(
            "rsi_overbought", sell_observation, feature_names
        )

        assert sig_type == SignalType.SELL
        assert strength > 0

    @patch(
        "ztb.trading.strategies.signal_definitions.SignalDefinitions.evaluate_signal"
    )
    def test_signal_strength_calculation(self, mock_evaluate, signal_guide):
        """Test signal strength calculation with mocked signals."""
        # Mock all signals to avoid StopIteration
        mock_evaluate.return_value = (SignalType.BUY, 0.8)

        observation = np.array([100.0, 50.0, 0.0, 105.0, 95.0])
        strength = signal_guide.get_signal_strength(
            observation, ACTION_BUY, step=0
        )  # BUY action

        # Should get positive strength for BUY action with strong BUY signal
        assert strength > 0

    def test_feature_names_setting(self, signal_guide):
        """Test setting feature names."""
        feature_names = ["price", "volume", "rsi", "macd"]
        signal_guide.set_feature_names(feature_names)

        assert signal_guide.feature_names == feature_names

        # Test with None (should not crash)
        signal_guide.set_feature_names(None)
        assert signal_guide.feature_names is None
