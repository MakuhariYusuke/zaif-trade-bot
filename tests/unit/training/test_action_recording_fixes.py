#!/usr/bin/env python3
"""
Tests for PPO/SAC action recording fixes in callbacks.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback


class TestActionRecordingFixes:
    """Test action recording fixes for PPO/SAC algorithms."""

    @pytest.fixture
    def mock_trainer(self):
        """Mock trainer for testing."""
        trainer = Mock()
        trainer.policy = Mock()
        trainer.policy.action_space = Mock()
        return trainer

    @pytest.fixture
    def callback(self, mock_trainer):
        """TrainingProgressCallback instance for testing."""
        callback = TrainingProgressCallback(
            check_freq=100, verbose=1, trainer_ref=mock_trainer
        )
        callback.model = mock_trainer  # Required for BaseCallback.on_step()
        return callback

    def test_ppo_discrete_action_recording(self, callback, mock_trainer):
        """Test that PPO discrete actions are recorded correctly."""
        # Mock PPO trainer (discrete actions)
        mock_trainer.policy.action_space.n = 3  # Discrete space with 3 actions

        # Set up callback with trainer reference
        callback.trainer = mock_trainer
        callback.locals = {"actions": np.array([1])}  # BUY action

        # Call the action recording logic
        callback._on_step()

        # Check that discrete action is recorded correctly
        assert len(callback.discrete_actions) == 1
        assert callback.discrete_actions[0] == 1  # BUY

        # Check that continuous equivalent is mapped correctly
        assert len(callback.continuous_actions) == 1
        assert callback.continuous_actions[0] == 1.0  # BUY maps to 1.0
        assert callback.continuous_actions[0] == 1.0  # BUY maps to 1.0

    def test_sac_continuous_action_recording(self, callback, mock_trainer):
        """Test that SAC continuous actions are recorded correctly."""
        # Mock SAC trainer (continuous actions)
        mock_trainer.policy.action_space.n = None  # No n attribute for continuous

        # Set up callback with trainer reference
        callback.trainer = mock_trainer
        callback.locals = {"actions": np.array([0.5])}  # Continuous action

        # Call the action recording logic
        callback._on_step()

        # Check that continuous action is recorded correctly
        assert len(callback.continuous_actions) == 1
        assert callback.continuous_actions[0] == 0.5

        # Check that discrete equivalent is calculated correctly
        assert len(callback.discrete_actions) == 1
        # 0.5 should map to BUY (1) based on continuous_to_discrete_action logic
        assert callback.discrete_actions[0] == 1

    def test_ppo_sell_action_not_counted_as_buy(self, callback, mock_trainer):
        """Test that PPO SELL actions are not miscounted as BUY."""
        # Mock PPO trainer
        mock_trainer.policy.action_space.n = 3

        # Test different PPO actions
        test_cases = [
            (0, 0.0, "HOLD"),  # HOLD -> 0.0
            (1, 1.0, "BUY"),  # BUY -> 1.0
            (2, -1.0, "SELL"),  # SELL -> -1.0
        ]

        for discrete_action, expected_continuous, action_name in test_cases:
            callback.discrete_actions.clear()
            callback.continuous_actions.clear()

            actions = np.array([discrete_action])
            logs = {}

            # Set up callback locals for on_step
            callback.locals = {"actions": actions}
            callback.on_step()

            assert callback.discrete_actions[0] == discrete_action
            assert (
                callback.continuous_actions[0] == expected_continuous
            ), f"{action_name} should map to {expected_continuous}"

    def test_sac_action_conversion_boundaries(self, callback, mock_trainer):
        """Test SAC action conversion at boundary values."""
        # Mock SAC trainer
        mock_trainer.policy.action_space.n = None

        # Test boundary conversions
        test_cases = [
            (-1.0, 2, "SELL"),  # -1.0 -> SELL (2)
            (-0.5, 2, "SELL"),  # -0.5 -> SELL (2)
            (0.0, 0, "HOLD"),  # 0.0 -> HOLD (0)
            (0.5, 1, "BUY"),  # 0.5 -> BUY (1)
            (1.0, 1, "BUY"),  # 1.0 -> BUY (1)
        ]

        for continuous_action, expected_discrete, action_name in test_cases:
            callback.discrete_actions.clear()
            callback.continuous_actions.clear()

            actions = np.array([continuous_action])
            logs = {}

            # Set up callback locals for on_step
            callback.locals = {"actions": actions}
            callback.on_step()

            assert callback.continuous_actions[0] == continuous_action
            assert (
                callback.discrete_actions[0] == expected_discrete
            ), f"{continuous_action} should convert to {action_name} ({expected_discrete})"

    def test_regime_action_counts_with_ppo(self, callback, mock_trainer):
        """Test regime-specific action counting with PPO."""
        # Mock PPO trainer
        mock_trainer.policy.action_space.n = 3

        # Mock regime detector
        callback.regime_action_counts = {
            "bull": [0, 0, 0],
            "bear": [0, 0, 0],
        }

        # Simulate regime detection
        def mock_detect_regime(price, step):
            return "bull" if step % 2 == 0 else "bear"

        # Mock the regime detection (we can't easily mock the reward_calculator here)
        # Instead, test the logic structure
        actions = np.array([1])  # BUY
        logs = {}

        # This should not crash
        callback.locals = {"actions": actions}
        callback.on_step()

        # The regime counting would happen in the reward calculator
        # but the action recording should work
        assert len(callback.discrete_actions) == 1
        assert callback.discrete_actions[0] == 1

    def test_action_recording_with_none_actions(self, callback, mock_trainer):
        """Test that action recording handles None/missing actions gracefully."""
        # Mock SAC trainer
        mock_trainer.policy.action_space.n = None

        # Test with None actions (should not crash)
        actions = None
        logs = {}

        # This should not raise an exception
        callback.locals = {"actions": actions}
        callback.on_step()

        # No actions should be recorded
        assert len(callback.discrete_actions) == 0
        assert len(callback.continuous_actions) == 0

    def test_mixed_algorithm_detection(self, callback):
        """Test that algorithm detection works for both PPO and SAC."""
        # Test PPO detection
        ppo_trainer = Mock()
        ppo_trainer.policy = Mock()
        ppo_trainer.policy.action_space = Mock()
        ppo_trainer.policy.action_space.n = 3

        callback.trainer_ref = ppo_trainer
        actions = np.array([1])
        logs = {}

        callback.locals = {"actions": actions}
        callback.on_step()

        assert callback.discrete_actions[-1] == 1
        assert callback.continuous_actions[-1] == 1.0

        # Test SAC detection
        sac_trainer = Mock()
        sac_trainer.policy = Mock()
        sac_trainer.policy.action_space = Mock()
        # SAC doesn't have n attribute
        del sac_trainer.policy.action_space.n

        callback.trainer_ref = sac_trainer
        callback.discrete_actions.clear()
        callback.continuous_actions.clear()

        actions = np.array([0.5])
        callback.locals = {"actions": actions}
        callback.on_step()

        assert callback.continuous_actions[-1] == 0.5
        assert callback.discrete_actions[-1] == 1  # 0.5 -> BUY


if __name__ == "__main__":
    pytest.main([__file__])
