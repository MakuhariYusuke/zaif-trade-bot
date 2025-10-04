"""
Unit tests for ztb.training.ppo_trainer module.
"""

import tempfile
from pathlib import Path

from ztb.training.eval_gates import EvalGates
from ztb.training.ppo_trainer import PPOTrainer


class TestPPOTrainer:
    """Test PPOTrainer functionality."""

    def test_initialization(self):
        """Test PPOTrainer initialization."""
        trainer = PPOTrainer()
        assert trainer is not None
        assert trainer.eval_gates is not None
        assert trainer.halt_callback is None
        assert trainer.checkpoint_interval == 10000
        assert trainer.current_step == 0
        assert not trainer.is_training
        assert trainer.halt_reason is None

    def test_initialization_with_params(self):
        """Test PPOTrainer initialization with custom parameters."""

        def mock_callback(reason: str) -> None:
            pass

        custom_gates = EvalGates(enabled=False)
        trainer = PPOTrainer(
            eval_gates=custom_gates,
            halt_callback=mock_callback,
            checkpoint_interval=5000,
        )

        assert trainer.eval_gates == custom_gates
        assert trainer.halt_callback == mock_callback
        assert trainer.checkpoint_interval == 5000

    def test_start_training(self):
        """Test starting training."""
        trainer = PPOTrainer()

        trainer.start_training()
        assert trainer.is_training
        assert trainer.halt_reason is None
        assert trainer.consecutive_failures == 0
        assert trainer.last_gate_check_step == 0

    def test_stop_training(self):
        """Test stopping training."""
        trainer = PPOTrainer()
        trainer.start_training()

        trainer.stop_training("Test stop")
        assert not trainer.is_training
        assert trainer.halt_reason == "Test stop"

    def test_stop_training_with_callback(self):
        """Test stopping training with callback."""
        callback_called = False
        captured_reason = None

        def mock_callback(reason: str) -> None:
            nonlocal callback_called, captured_reason
            callback_called = True
            captured_reason = reason

        trainer = PPOTrainer(halt_callback=mock_callback)
        trainer.start_training()

        trainer.stop_training("Test callback")
        assert callback_called
        assert captured_reason == "Test callback"

    def test_update_progress(self):
        """Test updating training progress."""
        trainer = PPOTrainer()
        trainer.start_training()

        # Update progress
        trainer.update_progress(100, 1.5)
        assert trainer.current_step == 100
        assert len(trainer.rewards_history) == 1
        assert len(trainer.steps_history) == 1
        assert trainer.rewards_history[0] == 1.5
        assert trainer.steps_history[0] == 100

        # Update again
        trainer.update_progress(200, 2.5)
        assert trainer.current_step == 200
        assert len(trainer.rewards_history) == 2
        assert trainer.rewards_history[1] == 2.5

    def test_reward_statistics(self):
        """Test reward statistics calculation."""
        trainer = PPOTrainer()
        trainer.start_training()

        # Add some rewards
        trainer.update_progress(100, 1.0)
        trainer.update_progress(200, 2.0)
        trainer.update_progress(300, 3.0)

        stats = trainer.get_reward_stats()
        assert stats["count"] == 3
        assert abs(stats["mean"] - 2.0) < 1e-6
        assert stats["variance"] > 0  # Should have some variance
        assert stats["std"] > 0

    def test_empty_reward_statistics(self):
        """Test reward statistics with no data."""
        trainer = PPOTrainer()

        stats = trainer.get_reward_stats()
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["variance"] == 0.0
        assert stats["std"] == 0.0

    def test_training_status(self):
        """Test getting training status."""
        trainer = PPOTrainer()
        trainer.start_training()
        trainer.update_progress(100, 1.5)

        status = trainer.get_training_status()
        assert status["is_training"] is True
        assert status["current_step"] == 100
        assert status["halt_reason"] is None
        assert status["consecutive_failures"] == 0
        assert "reward_stats" in status
        assert status["reward_stats"]["count"] == 1

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load functionality."""
        trainer = PPOTrainer()
        trainer.start_training()
        trainer.update_progress(100, 1.5)
        trainer.consecutive_failures = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.json"

            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Create new trainer and load checkpoint
            new_trainer = PPOTrainer()
            new_trainer.load_checkpoint(str(checkpoint_path))

            assert new_trainer.current_step == 100
            assert len(new_trainer.rewards_history) == 1
            assert (
                list(new_trainer.rewards_history)[0] == 1.5
            )  # Convert deque to list for comparison
            assert new_trainer.consecutive_failures == 2

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        trainer = PPOTrainer()

        # Should not raise error, just log warning
        trainer.load_checkpoint("nonexistent_checkpoint.json")
        assert trainer.current_step == 0  # Should remain unchanged

    def test_update_progress_not_training(self):
        """Test update_progress when not training."""
        trainer = PPOTrainer()

        # Should not update when not training
        trainer.update_progress(100, 1.5)
        assert trainer.current_step == 0
        assert len(trainer.rewards_history) == 0

    def test_reward_history_size_limit(self):
        """Test that reward history respects size limits."""
        trainer = PPOTrainer()
        trainer.start_training()

        # Add more rewards than maxlen (50000)
        for i in range(60000):
            trainer.update_progress(i, float(i % 10))

        # History should be limited
        assert len(trainer.rewards_history) <= 50000
        assert len(trainer.steps_history) <= 50000
