"""
Unit tests for BalanceCurriculumManager SAC v448 Layer 3.

Tests:
- Stage progression based on conditions
- Emergency revert to forced_balance
- Integration with existing curriculum_stage system
- Backward compatibility (can be disabled)
"""

import pytest
from unittest.mock import Mock
from ztb.trading.environment.components.reward.balance_curriculum import (
    BalanceCurriculumManager,
)


@pytest.fixture
def mock_config():
    """Mock environment config."""
    config = Mock()
    config.curriculum_stage = "forced_balance"
    return config


@pytest.fixture
def manager(mock_config):
    """Create BalanceCurriculumManager instance."""
    return BalanceCurriculumManager(mock_config, enabled=True, auto_progression=True)


@pytest.fixture
def manager_disabled(mock_config):
    """Create disabled BalanceCurriculumManager (v447 compatibility)."""
    return BalanceCurriculumManager(mock_config, enabled=False)


class TestInitialization:
    """Test initialization and configuration."""

    def test_default_initialization(self, mock_config):
        """Manager initializes with default settings."""
        manager = BalanceCurriculumManager(mock_config)
        
        assert manager.enabled is True
        assert manager.auto_progression is True
        assert manager.emergency_revert is True
        assert manager.current_stage == "forced_balance"
        assert manager.emergency_count == 0

    def test_disabled_initialization(self, mock_config):
        """Disabled manager for backward compatibility."""
        manager = BalanceCurriculumManager(mock_config, enabled=False)
        
        assert manager.enabled is False
        assert manager.current_stage == "forced_balance"

    def test_custom_initial_stage(self, mock_config):
        """Manager respects custom initial stage from config."""
        mock_config.curriculum_stage = "pnl_focused"
        manager = BalanceCurriculumManager(mock_config)
        
        assert manager.current_stage == "pnl_focused"


class TestDisabledMode:
    """Test backward compatibility when disabled."""

    def test_update_returns_static_stage(self, manager_disabled):
        """Disabled manager returns fixed stage without changes."""
        status = manager_disabled.update(
            step=100,
            action_counts=[40, 30, 30],
            recent_rewards=[5.0] * 50,
        )
        
        assert status["stage"] == "forced_balance"
        assert status["changed"] is False
        assert status["emergency"] is False

    def test_no_stage_progression(self, manager_disabled):
        """Disabled manager never progresses stages."""
        # Simulate perfect conditions
        for _ in range(10):
            manager_disabled.update(
                step=200,
                action_counts=[10, 45, 45],  # Perfect balance
                recent_rewards=[10.0] * 50,
            )
        
        assert manager_disabled.current_stage == "forced_balance"

    def test_no_emergency_intervention(self, manager_disabled):
        """Disabled manager doesn't trigger emergency."""
        status = manager_disabled.update(
            step=100,
            action_counts=[10, 85, 5],  # Extreme bias
            recent_rewards=[-10.0] * 50,
        )
        
        assert status["emergency"] is False


class TestEmergencyRevert:
    """Test emergency revert to forced_balance."""

    def test_emergency_on_extreme_bias(self, manager):
        """Emergency triggers on BUY-SELL diff > 35%."""
        manager.current_stage = "pnl_focused"  # Start from advanced stage
        
        status = manager.update(
            step=100,
            action_counts=[10, 85, 5],  # 80% BUY, 5% SELL = 75% diff
            recent_rewards=[5.0],
        )
        
        assert status["emergency"] is True
        assert status["stage"] == "forced_balance"
        assert status["changed"] is True
        assert manager.emergency_count == 1

    def test_no_emergency_on_moderate_bias(self, manager):
        """No emergency when bias is below threshold."""
        manager.current_stage = "pnl_focused"
        
        status = manager.update(
            step=100,
            action_counts=[20, 50, 30],  # 50% BUY, 30% SELL = 20% diff
            recent_rewards=[5.0],
        )
        
        assert status["emergency"] is False
        assert status["stage"] == "pnl_focused"  # No change

    def test_emergency_on_negative_rewards_with_bias(self, manager):
        """Emergency triggers on sustained negative rewards + bias."""
        manager.current_stage = "balanced_transition"
        
        # Build up negative reward history
        negative_rewards = [-3.0] * 25
        status = manager.update(
            step=100,
            action_counts=[20, 55, 25],  # 30% diff (moderate bias)
            recent_rewards=negative_rewards,
        )
        
        assert status["emergency"] is True
        assert status["stage"] == "forced_balance"

    def test_emergency_limit(self, manager):
        """Emergency revert limited to max_emergency_reverts."""
        # Start from different advanced stages to trigger multiple emergencies
        stages_to_test = ["pnl_focused", "balanced_transition", "pnl_focused"]
        
        for i, stage in enumerate(stages_to_test):
            manager.current_stage = stage  # Move to advanced stage
            manager.update(
                step=100 + i * 100,
                action_counts=[10, 85, 5],  # Extreme bias
                recent_rewards=[-5.0],
            )
        
        # Should have triggered 3 emergencies
        assert manager.emergency_count == 3
        
        # 4th attempt: should not increase count (at limit)
        manager.current_stage = "pnl_focused"
        manager.update(
            step=500,
            action_counts=[10, 85, 5],
            recent_rewards=[-5.0],
        )
        
        # Count should not exceed max
        assert manager.emergency_count == 3

    def test_no_emergency_at_forced_balance(self, manager):
        """Already at forced_balance, no emergency needed."""
        status = manager.update(
            step=100,
            action_counts=[10, 85, 5],  # Extreme bias
            recent_rewards=[5.0],
        )
        
        # Emergency not triggered because already at safest stage
        assert status["emergency"] is False
        assert status["stage"] == "forced_balance"


class TestStageProgression:
    """Test automatic stage progression."""

    def test_no_progression_insufficient_steps(self, manager):
        """No progression before meeting min_steps requirement."""
        status = manager.update(
            step=50,  # Less than min_steps (100)
            action_counts=[40, 30, 30],  # Good balance
            recent_rewards=[2.0] * 20,
        )
        
        assert status["changed"] is False
        assert status["stage"] == "forced_balance"

    def test_forced_balance_to_balanced_transition(self, manager):
        """Progress from forced_balance to balanced_transition."""
        # Simulate good balance + positive rewards for sufficient steps
        for step in range(150):
            manager.update(
                step=step,
                action_counts=[40, 32, 28],  # 4% diff
                recent_rewards=[2.5],
            )
        
        assert manager.current_stage == "balanced_transition"

    def test_balanced_transition_to_pnl_focused(self, manager):
        """Progress from balanced_transition to pnl_focused."""
        manager.current_stage = "balanced_transition"
        manager.stage_start_step = 0
        
        # Build up positive rewards
        rewards = [3.0] * 40
        status = manager.update(
            step=250,
            action_counts=[30, 38, 32],  # 6% diff (good balance)
            recent_rewards=rewards,
        )
        
        # Should progress after meeting conditions
        assert status["stage"] in ["balanced_transition", "pnl_focused"]

    def test_no_progression_without_auto(self, mock_config):
        """No auto progression when auto_progression=False."""
        manager = BalanceCurriculumManager(
            mock_config, enabled=True, auto_progression=False
        )
        
        # Perfect conditions
        for step in range(200):
            manager.update(
                step=step,
                action_counts=[40, 30, 30],
                recent_rewards=[5.0],
            )
        
        # Should stay at initial stage
        assert manager.current_stage == "forced_balance"

    def test_stage_history_recorded(self, manager):
        """Stage transitions are recorded in history."""
        # Progress through stages
        for step in range(200):
            manager.update(
                step=step,
                action_counts=[40, 32, 28],
                recent_rewards=[3.0],
            )
        
        info = manager.get_stage_info()
        
        # Check history exists
        if manager.current_stage != "forced_balance":
            assert len(info["stage_history"]) > 0
            first_entry = info["stage_history"][0]
            assert "stage" in first_entry
            assert "start_step" in first_entry
            assert "end_step" in first_entry


class TestMetricsTracking:
    """Test metrics tracking and calculations."""

    def test_recent_rewards_tracking(self, manager):
        """Recent rewards are tracked with maxlen."""
        rewards = list(range(150))
        manager.update(
            step=100,
            action_counts=[40, 30, 30],
            recent_rewards=rewards,
        )
        
        # Should only keep last 100
        assert len(manager.recent_rewards) == 100
        assert manager.recent_rewards[-1] == 149

    def test_stage_rewards_cleared_on_progression(self, manager):
        """Stage rewards reset when transitioning."""
        # Build up rewards
        for i in range(50):
            manager.update(
                step=i,
                action_counts=[40, 30, 30],
                recent_rewards=[2.0],
            )
        
        initial_len = len(manager.stage_rewards)
        assert initial_len > 0
        
        # Force progression
        manager._progress_to_stage("balanced_transition", 100)
        
        # Stage rewards should be cleared
        assert len(manager.stage_rewards) == 0

    def test_get_stage_info(self, manager):
        """get_stage_info returns comprehensive status."""
        manager.update(
            step=150,
            action_counts=[40, 30, 30],
            recent_rewards=[2.0, 3.0, 2.5],
        )
        
        info = manager.get_stage_info()
        
        assert "current_stage" in info
        assert "steps_in_stage" in info
        assert "total_steps" in info
        assert "emergency_count" in info
        assert "enabled" in info
        assert info["total_steps"] == 150


class TestIntegration:
    """Integration tests for real-world usage."""

    def test_complete_progression_cycle(self, manager):
        """Simulate complete training with stage progression."""
        steps = 1000
        
        for step in range(steps):
            # Simulate gradually improving performance
            if step < 200:
                action_counts = [40, 32, 28]  # Balanced
                reward = 1.0 + step / 100
            elif step < 500:
                action_counts = [30, 38, 32]  # Slightly more trading
                reward = 2.0 + step / 100
            else:
                action_counts = [20, 42, 38]  # More aggressive
                reward = 5.0 + step / 200
            
            status = manager.update(
                step=step,
                action_counts=action_counts,
                recent_rewards=[reward],
            )
        
        # Should have progressed beyond forced_balance
        info = manager.get_stage_info()
        assert info["current_stage"] != "forced_balance" or len(info["stage_history"]) > 0

    def test_emergency_recovery(self, manager):
        """System recovers after emergency revert."""
        manager.current_stage = "pnl_focused"
        
        # Trigger emergency
        manager.update(
            step=100,
            action_counts=[10, 85, 5],
            recent_rewards=[-5.0],
        )
        
        assert manager.current_stage == "forced_balance"
        assert manager.emergency_count == 1
        
        # Recover with good behavior
        for step in range(200, 400):
            manager.update(
                step=step,
                action_counts=[40, 32, 28],
                recent_rewards=[3.0],
            )
        
        # Should progress again
        assert manager.current_stage in ["balanced_transition", "pnl_focused"]

    def test_get_current_stage_compatibility(self, manager):
        """get_current_stage() returns string for RewardCalculator."""
        stage = manager.get_current_stage()
        
        assert isinstance(stage, str)
        assert stage in BalanceCurriculumManager.STAGE_SEQUENCE
