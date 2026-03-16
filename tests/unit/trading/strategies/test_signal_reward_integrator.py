#!/usr/bin/env python3
"""
Comprehensive unit tests for SignalRewardIntegrator.
Tests integration of technical signals with reward functions, including new Bollinger Bands and ADX support.
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import ActionSignalGuide
from ztb.trading.strategies.signal_reward_integrator import SignalRewardIntegrator
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL


class TestSignalRewardIntegrator:
    """Test cases for SignalRewardIntegrator."""

    @pytest.fixture
    def signal_guide(self):
        """Create a real ActionSignalGuide instance for testing."""
        return ActionSignalGuide()

    @pytest.fixture
    def sample_observation(self):
        """Create sample observation data."""
        return np.array([
            105.0,  # open
            107.5,  # high
            104.0,  # low
            106.2,  # close
            1500.0  # volume
        ])

    def test_initialization_default_weights(self, signal_guide):
        """Test initialization with default weights."""
        integrator = SignalRewardIntegrator(signal_guide=signal_guide)

        assert integrator.signal_guide == signal_guide
        assert integrator.signal_bonus_weight == 0.1
        assert integrator.signal_penalty_weight == 0.05
        assert integrator.bollinger_weight == 1.3
        assert integrator.adx_weight == 1.4
        assert integrator.enable_advanced_integration is True

    def test_initialization_custom_weights(self, signal_guide):
        """Test initialization with custom weights."""
        integrator = SignalRewardIntegrator(
            signal_guide=signal_guide,
            signal_bonus_weight=0.2,
            signal_penalty_weight=0.1,
            bollinger_weight=1.5,
            adx_weight=1.6,
            enable_advanced_integration=False
        )

        assert integrator.signal_bonus_weight == 0.2
        assert integrator.signal_penalty_weight == 0.1
        assert integrator.bollinger_weight == 1.5
        assert integrator.adx_weight == 1.6
        assert integrator.enable_advanced_integration is False

    def test_integrate_signal_reward_basic(self, signal_guide, sample_observation):
        """Test basic reward integration."""
        integrator = SignalRewardIntegrator(
            signal_guide=signal_guide,
            enable_advanced_integration=False
        )

        base_reward = 1.0
        modified_reward = integrator.integrate_signal_reward(
            reward=base_reward,
            observation=sample_observation,
            action=ACTION_HOLD,
            step=1
        )

        # Should return a valid float
        assert isinstance(modified_reward, float)

    def test_integrate_signal_reward_advanced(self, signal_guide, sample_observation):
        """Test advanced reward integration."""
        integrator = SignalRewardIntegrator(
            signal_guide=signal_guide,
            enable_advanced_integration=True
        )

        base_reward = 1.0
        modified_reward = integrator.integrate_signal_reward(
            reward=base_reward,
            observation=sample_observation,
            action=ACTION_BUY,
            step=1
        )

        # Should return a valid float
        assert isinstance(modified_reward, float)

    def test_integrate_signal_reward_no_observation(self, signal_guide):
        """Test reward integration with no observation."""
        integrator = SignalRewardIntegrator(signal_guide=signal_guide)

        base_reward = 1.0
        modified_reward = integrator.integrate_signal_reward(
            reward=base_reward,
            observation=None,
            action=ACTION_HOLD,
            step=1
        )

        # Should return base reward when no observation
        assert modified_reward == base_reward

    def test_get_integration_stats(self, signal_guide, sample_observation):
        """Test getting integration statistics."""
        integrator = SignalRewardIntegrator(signal_guide=signal_guide)

        # Perform some integrations
        integrator.integrate_signal_reward(1.0, sample_observation, ACTION_BUY, 1)
        integrator.integrate_signal_reward(1.0, sample_observation, ACTION_SELL, 2)

        stats = integrator.get_integration_stats()

        assert 'total_steps' in stats
        assert 'signal_bonuses_applied' in stats
        assert 'signal_penalties_applied' in stats
        assert 'bollinger_signals_used' in stats
        assert 'adx_signals_used' in stats
        assert 'pattern_weights' in stats

        assert stats['total_steps'] == 2
        assert stats['bollinger_signals_used'] >= 0
        assert stats['adx_signals_used'] >= 0

    def test_reset_stats(self, signal_guide, sample_observation):
        """Test statistics reset functionality."""
        integrator = SignalRewardIntegrator(signal_guide=signal_guide)

        # Perform integration to generate stats
        integrator.integrate_signal_reward(1.0, sample_observation, ACTION_BUY, 1)

        # Reset stats
        integrator.reset_stats()

        stats = integrator.get_integration_stats()
        assert stats['total_steps'] == 0
        assert stats['signal_bonuses_applied'] == 0
        assert stats['signal_penalties_applied'] == 0
        assert stats['bollinger_signals_used'] == 0
        assert stats['adx_signals_used'] == 0

    def test_different_actions(self, signal_guide, sample_observation):
        """Test integration with different actions."""
        integrator = SignalRewardIntegrator(signal_guide=signal_guide)

        base_reward = 1.0

        # Test all actions
        for action in [ACTION_BUY, ACTION_HOLD, ACTION_SELL]:
            modified_reward = integrator.integrate_signal_reward(
                reward=base_reward,
                observation=sample_observation,
                action=action,
                step=1
            )
            assert isinstance(modified_reward, float)

    def test_advanced_integration_enabled(self, signal_guide, sample_observation):
        """Test with advanced integration enabled."""
        integrator = SignalRewardIntegrator(
            signal_guide=signal_guide,
            enable_advanced_integration=True
        )

        modified_reward = integrator.integrate_signal_reward(
            reward=1.0,
            observation=sample_observation,
            action=ACTION_BUY,
            step=1
        )

        assert isinstance(modified_reward, float)

    def test_advanced_integration_disabled(self, signal_guide, sample_observation):
        """Test with advanced integration disabled."""
        integrator = SignalRewardIntegrator(
            signal_guide=signal_guide,
            enable_advanced_integration=False
        )

        modified_reward = integrator.integrate_signal_reward(
            reward=1.0,
            observation=sample_observation,
            action=ACTION_BUY,
            step=1
        )

        assert isinstance(modified_reward, float)

    @pytest.mark.parametrize("bollinger_weight,adx_weight", [
        (1.0, 1.0),
        (1.5, 1.2),
        (2.0, 1.8),
    ])
    def test_different_weights(self, signal_guide, sample_observation, bollinger_weight, adx_weight):
        """Test with different pattern weights."""
        integrator = SignalRewardIntegrator(
            signal_guide=signal_guide,
            bollinger_weight=bollinger_weight,
            adx_weight=adx_weight
        )

        modified_reward = integrator.integrate_signal_reward(
            reward=1.0,
            observation=sample_observation,
            action=ACTION_BUY,
            step=1
        )

        assert isinstance(modified_reward, float)

        # Check that weights are correctly set
        stats = integrator.get_integration_stats()
        pattern_weights = stats['pattern_weights']
        assert pattern_weights['bollinger'] == bollinger_weight
        assert pattern_weights['adx'] == adx_weight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])