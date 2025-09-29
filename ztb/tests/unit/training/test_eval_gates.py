"""
Unit tests for evaluation gates with auto-halt functionality.
"""

from unittest.mock import MagicMock, patch

from ztb.training.eval_gates import EvalGates, GateResult, GateStatus


class TestEvalGatesAutoHalt:
    """Test cases for auto-halt functionality in EvalGates."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gates = EvalGates(enabled=True)

    def test_should_halt_training_no_failures(self):
        """Test should_halt_training with no gate failures."""
        gate_results = {
            "memory_rss": GateResult("memory_rss", GateStatus.PASS, "OK", 1e9, 2e9),
            "reward_trend_300k": GateResult(
                "reward_trend_300k", GateStatus.PASS, "Positive", 0.1, 0.0
            ),
        }

        should_halt, reason = self.gates.should_halt_training(gate_results)

        assert not should_halt
        assert reason == ""

    def test_should_halt_training_critical_failure_memory(self):
        """Test should_halt_training with critical memory failure."""
        gate_results = {
            "memory_rss": GateResult(
                "memory_rss", GateStatus.FAIL, "Memory exceeded", 3e9, 2e9
            ),
            "reward_trend_300k": GateResult(
                "reward_trend_300k", GateStatus.PASS, "OK", 0.1, 0.0
            ),
        }

        should_halt, reason = self.gates.should_halt_training(gate_results)

        assert should_halt
        assert "Critical gate failure" in reason
        assert "memory_rss" in reason

    def test_should_halt_training_critical_failure_duplicate_steps(self):
        """Test should_halt_training with critical duplicate steps failure."""
        gate_results = {
            "no_dup_steps": GateResult(
                "no_dup_steps", GateStatus.FAIL, "Duplicates found", 5, 0
            ),
            "memory_rss": GateResult("memory_rss", GateStatus.PASS, "OK", 1e9, 2e9),
        }

        should_halt, reason = self.gates.should_halt_training(gate_results)

        assert should_halt
        assert "Critical gate failure" in reason
        assert "no_dup_steps" in reason

    def test_should_halt_training_consecutive_failures(self):
        """Test should_halt_training with too many consecutive failures."""
        gate_results = {
            "eval_above_baseline": GateResult(
                "eval_above_baseline", GateStatus.FAIL, "Below baseline", -0.1, 0.0
            )
        }

        should_halt, reason = self.gates.should_halt_training(
            gate_results, consecutive_failures=3, max_consecutive_failures=3
        )

        assert should_halt
        assert "Too many consecutive failures" in reason

    def test_should_halt_training_persistent_negative_trend(self):
        """Test should_halt_training with persistent negative reward trend."""
        gate_results = {
            "reward_trend_300k": GateResult(
                "reward_trend_300k", GateStatus.FAIL, "Negative trend", -0.05, 0.0
            )
        }

        should_halt, reason = self.gates.should_halt_training(
            gate_results, consecutive_failures=2, max_consecutive_failures=3
        )

        assert should_halt
        assert "Persistent negative reward trend" in reason

    def test_should_halt_training_no_halt_for_single_failure(self):
        """Test should_halt_training does not halt for single non-critical failure."""
        gate_results = {
            "eval_above_baseline": GateResult(
                "eval_above_baseline", GateStatus.FAIL, "Below baseline", -0.1, 0.0
            )
        }

        should_halt, reason = self.gates.should_halt_training(
            gate_results, consecutive_failures=1, max_consecutive_failures=3
        )

        assert not should_halt
        assert reason == ""

    def test_should_halt_training_empty_results(self):
        """Test should_halt_training with empty gate results."""
        should_halt, reason = self.gates.should_halt_training({})

        assert not should_halt
        assert reason == ""

    def test_should_halt_training_mixed_results(self):
        """Test should_halt_training with mixed pass/fail results."""
        gate_results = {
            "memory_rss": GateResult("memory_rss", GateStatus.PASS, "OK", 1e9, 2e9),
            "eval_above_baseline": GateResult(
                "eval_above_baseline", GateStatus.FAIL, "Below baseline", -0.1, 0.0
            ),
            "reward_trend_300k": GateResult(
                "reward_trend_300k", GateStatus.PASS, "OK", 0.1, 0.0
            ),
        }

        should_halt, reason = self.gates.should_halt_training(
            gate_results, consecutive_failures=1, max_consecutive_failures=3
        )

        # Should not halt because failure is not critical and consecutive count is low
        assert not should_halt
        assert reason == ""

    @patch("psutil.Process")
    def test_memory_gate_failure_triggers_halt(self, mock_process):
        """Test that memory gate failure properly triggers halt logic."""
        # Mock memory usage exceeding threshold
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 3 * 1024 * 1024 * 1024  # 3GB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        # Run evaluation
        gate_results = self.gates.evaluate_all()

        # Check that memory gate failed
        assert "memory_rss" in gate_results
        assert gate_results["memory_rss"].status == GateStatus.FAIL

        # Test halt logic
        should_halt, reason = self.gates.should_halt_training(gate_results)
        assert should_halt
        assert "Critical gate failure" in reason


class TestEvalGatesIntegration:
    """Integration tests for EvalGates with auto-halt."""

    def test_full_evaluation_flow_with_halt(self):
        """Test full evaluation flow that triggers auto-halt."""
        gates = EvalGates(enabled=True)

        # Simulate training data that should trigger halt
        rewards = [0.1, 0.05, 0.02, -0.01, -0.02]  # Declining trend
        steps = [100000, 200000, 300000, 400000, 500000]

        # Mock high memory usage
        with patch("psutil.Process") as mock_process:
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 3 * 1024 * 1024 * 1024  # 3GB
            mock_process.return_value.memory_info.return_value = mock_memory_info

            gate_results = gates.evaluate_all(
                rewards=rewards, steps=steps, final_eval_reward=-0.02
            )

            # Should have multiple failures
            failed_gates = [
                r for r in gate_results.values() if r.status == GateStatus.FAIL
            ]
            assert len(failed_gates) >= 2  # Memory + reward trend or eval

            # Should trigger halt
            should_halt, reason = gates.should_halt_training(gate_results)
            assert should_halt
            assert "Critical gate failure" in reason or "Persistent negative" in reason
