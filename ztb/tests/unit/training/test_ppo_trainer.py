"""
Unit tests for PPO Trainer with auto-halt functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.training.eval_gates import EvalGates, GateResult, GateStatus
from ztb.training.ppo_trainer import PPOTrainer


class TestPPOTrainer:
    """Test cases for PPOTrainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.eval_gates = EvalGates(enabled=True)
        self.halt_callback = MagicMock()
        self.trainer = PPOTrainer(
            eval_gates=self.eval_gates,
            halt_callback=self.halt_callback,
            checkpoint_interval=1000,
        )

    def test_initialization(self):
        """Test PPOTrainer initialization."""
        assert not self.trainer.is_training
        assert self.trainer.current_step == 0
        assert self.trainer.halt_reason is None
        assert self.trainer.consecutive_failures == 0
        assert len(self.trainer.rewards_history) == 0
        assert len(self.trainer.steps_history) == 0

    def test_start_training(self):
        """Test starting training."""
        self.trainer.start_training()

        assert self.trainer.is_training
        assert self.trainer.halt_reason is None
        assert self.trainer.consecutive_failures == 0

    def test_stop_training(self):
        """Test stopping training."""
        self.trainer.start_training()
        self.trainer.stop_training("Test reason")

        assert not self.trainer.is_training
        assert self.trainer.halt_reason == "Test reason"
        self.halt_callback.assert_called_once_with("Test reason")

    def test_update_progress_no_gates_check(self):
        """Test update_progress before gate check interval."""
        self.trainer.start_training()

        # Update with step < checkpoint_interval
        self.trainer.update_progress(500, 0.1)

        assert self.trainer.current_step == 500
        assert self.trainer.rewards_history == [0.1]
        assert self.trainer.steps_history == [500]
        # Should not have checked gates yet
        assert self.trainer.last_gate_check_step == 0

    @patch("ztb.training.eval_gates.EvalGates.evaluate_all")
    def test_update_progress_with_gates_check_success(self, mock_evaluate):
        """Test update_progress triggers successful gate check."""
        # Mock successful gate evaluation
        mock_evaluate.return_value = {
            "memory_rss": GateResult("memory_rss", GateStatus.PASS, "OK", 1e9, 2e9),
            "reward_trend_300k": GateResult(
                "reward_trend_300k", GateStatus.PASS, "Positive", 0.1, 0.0
            ),
        }

        self.trainer.start_training()

        # Update at checkpoint interval
        self.trainer.update_progress(1000, 0.1)

        assert self.trainer.last_gate_check_step == 1000
        assert self.trainer.consecutive_failures == 0  # Reset on success
        assert self.trainer.is_training  # Should continue training

    @patch("ztb.training.eval_gates.EvalGates.evaluate_all")
    def test_update_progress_with_gates_check_failure(self, mock_evaluate):
        """Test update_progress with gate failure."""
        # Mock failed gate evaluation
        mock_evaluate.return_value = {
            "memory_rss": GateResult(
                "memory_rss", GateStatus.FAIL, "High memory", 3e9, 2e9
            )
        }

        self.trainer.start_training()

        # Update at checkpoint interval
        self.trainer.update_progress(1000, 0.1)

        # Should have halted training due to critical failure
        assert not self.trainer.is_training
        assert "memory_rss: High memory" in self.trainer.halt_reason
        self.halt_callback.assert_called_once()

    @patch("ztb.training.eval_gates.EvalGates.evaluate_all")
    def test_consecutive_failures_trigger_halt(self, mock_evaluate):
        """Test that consecutive failures trigger auto-halt."""
        # Mock non-critical failure
        mock_evaluate.return_value = {
            "eval_above_baseline": GateResult(
                "eval_above_baseline", GateStatus.FAIL, "Below baseline", -0.1, 0.0
            )
        }

        self.trainer.start_training()
        self.trainer.max_consecutive_failures = 2

        # First failure
        self.trainer.update_progress(1000, 0.1)
        assert self.trainer.is_training
        assert self.trainer.consecutive_failures == 1

        # Second failure - should trigger halt
        self.trainer.update_progress(2000, 0.05)
        assert not self.trainer.is_training
        assert "eval_above_baseline: Below baseline" in self.trainer.halt_reason

    def test_get_training_status(self):
        """Test get_training_status method."""
        self.trainer.start_training()
        self.trainer.update_progress(500, 0.1)

        status = self.trainer.get_training_status()

        assert status["is_training"] is True
        assert status["current_step"] == 500
        assert status["consecutive_failures"] == 0
        assert "gate_results" in status

    def test_save_and_load_checkpoint(self):
        """Test checkpoint save and load functionality."""
        self.trainer.start_training()
        self.trainer.update_progress(1500, 0.1)
        self.trainer.consecutive_failures = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.json"

            # Save checkpoint
            self.trainer.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Create new trainer and load checkpoint
            new_trainer = PPOTrainer()
            new_trainer.load_checkpoint(str(checkpoint_path))

            assert new_trainer.current_step == 1500
            assert new_trainer.rewards_history == [0.1]
            assert new_trainer.steps_history == [1500]
            assert new_trainer.consecutive_failures == 2

    def test_load_checkpoint_file_not_found(self):
        """Test loading checkpoint when file doesn't exist."""
        new_trainer = PPOTrainer()
        new_trainer.load_checkpoint("nonexistent_checkpoint.json")

        # Should not crash, trainer remains in initial state
        assert new_trainer.current_step == 0

    @patch("ztb.training.eval_gates.EvalGates.evaluate_all")
    def test_should_auto_halt_logic(self, mock_evaluate):
        """Test the internal _should_auto_halt logic."""
        # Test critical memory failure
        mock_evaluate.return_value = {
            "memory_rss": GateResult(
                "memory_rss", GateStatus.FAIL, "High memory", 3e9, 2e9
            )
        }

        self.trainer.start_training()
        self.trainer.update_progress(1000, 0.1)

        # Should have halted due to critical failure
        assert not self.trainer.is_training

    def test_update_progress_when_not_training(self):
        """Test update_progress does nothing when not training."""
        self.trainer.update_progress(1000, 0.1)

        assert self.trainer.current_step == 0
        assert len(self.trainer.rewards_history) == 0

    def test_halt_callback_called_with_reason(self):
        """Test that halt callback is called with correct reason."""
        self.trainer.start_training()

        halt_reason = "Test halt reason"
        self.trainer.stop_training(halt_reason)

        self.halt_callback.assert_called_once_with(halt_reason)
