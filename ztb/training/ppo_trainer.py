"""
PPO Trainer with auto-halt functionality for training gates.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ztb.training.eval_gates import EvalGates, GateResult, GateStatus

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO Trainer with evaluation gates and auto-halt functionality."""

    def __init__(
        self,
        eval_gates: Optional[EvalGates] = None,
        halt_callback: Optional[Callable[[str], None]] = None,
        checkpoint_interval: int = 10000,
    ):
        """
        Initialize PPO trainer.

        Args:
            eval_gates: Evaluation gates for training validation
            halt_callback: Callback function called when training should halt
            checkpoint_interval: Steps between checkpoints
        """
        self.eval_gates = eval_gates or EvalGates()
        self.halt_callback = halt_callback
        self.checkpoint_interval = checkpoint_interval

        # Training state
        self.current_step = 0
        self.rewards_history: List[float] = []
        self.steps_history: List[int] = []
        self.is_training = False
        self.halt_reason: Optional[str] = None

        # Auto-halt state
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.last_gate_check_step = 0

    def start_training(self) -> None:
        """Start training session."""
        self.is_training = True
        self.halt_reason = None
        self.consecutive_failures = 0
        self.last_gate_check_step = 0
        logger.info("Training started")

    def stop_training(self, reason: str = "Manual stop") -> None:
        """Stop training session."""
        self.is_training = False
        self.halt_reason = reason
        logger.info(f"Training stopped: {reason}")

        if self.halt_callback:
            self.halt_callback(reason)

    def update_progress(self, step: int, reward: float) -> None:
        """
        Update training progress and check gates.

        Args:
            step: Current training step
            reward: Current episode reward
        """
        if not self.is_training:
            return

        self.current_step = step
        self.rewards_history.append(reward)
        self.steps_history.append(step)

        # Check gates periodically
        if step - self.last_gate_check_step >= self.checkpoint_interval:
            self._check_gates_and_halt_if_needed()

    def _check_gates_and_halt_if_needed(self) -> None:
        """Check evaluation gates and halt training if necessary."""
        if not self.eval_gates.enabled:
            return

        # Run gate checks
        gate_results = self.eval_gates.evaluate_all(
            rewards=self.rewards_history,
            steps=self.steps_history,
            final_eval_reward=self.rewards_history[-1] if self.rewards_history else 0.0,
        )

        # Count failures
        failed_gates = [r for r in gate_results.values() if r.status == GateStatus.FAIL]

        if failed_gates:
            self.consecutive_failures += 1
            logger.warning(f"Gate check failed: {len(failed_gates)} gates failed")

            # Check if we should auto-halt
            if self._should_auto_halt(gate_results):
                reasons = [f"{r.name}: {r.reason}" for r in failed_gates]
                halt_reason = f"Auto-halt: {len(failed_gates)} gates failed - {', '.join(reasons)}"
                self.stop_training(halt_reason)
                return
        else:
            # Reset consecutive failures on success
            self.consecutive_failures = 0

        self.last_gate_check_step = self.current_step

    def _should_auto_halt(self, gate_results: Dict[str, GateResult]) -> bool:
        """
        Determine if training should auto-halt based on gate results.

        Auto-halt conditions:
        1. Critical gates fail (memory, duplicate steps)
        2. Consecutive failures exceed threshold
        3. Reward trend is consistently negative

        Args:
            gate_results: Results from gate evaluation

        Returns:
            True if training should halt
        """
        if not gate_results:
            return False

        # Critical failures that should always halt
        critical_gates = ["memory_rss", "no_dup_steps"]
        for gate_name in critical_gates:
            if (
                gate_name in gate_results
                and gate_results[gate_name].status == GateStatus.FAIL
            ):
                logger.error(f"Critical gate failed: {gate_name}")
                return True

        # Consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures: {self.consecutive_failures}")
            return True

        # Persistent negative reward trend
        if "reward_trend_300k" in gate_results:
            trend_result = gate_results["reward_trend_300k"]
            if (
                trend_result.status == GateStatus.FAIL
                and self.consecutive_failures >= 2
            ):
                logger.error("Persistent negative reward trend")
                return True

        return False

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.

        Returns:
            Status dictionary with training state and gate results
        """
        status = {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "halt_reason": self.halt_reason,
            "consecutive_failures": self.consecutive_failures,
        }

        if self.eval_gates.enabled and self.rewards_history:
            gate_results = self.eval_gates.evaluate_all(
                rewards=self.rewards_history,
                steps=self.steps_history,
                final_eval_reward=self.rewards_history[-1]
                if self.rewards_history
                else 0.0,
            )
            status["gate_results"] = {
                name: {
                    "status": result.status.value,
                    "reason": result.reason,
                    "value": result.value,
                }
                for name, result in gate_results.items()
            }

        return status

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_data = {
            "current_step": self.current_step,
            "rewards_history": self.rewards_history,
            "steps_history": self.steps_history,
            "consecutive_failures": self.consecutive_failures,
            "last_gate_check_step": self.last_gate_check_step,
            "halt_reason": self.halt_reason,
            "is_training": self.is_training,
        }

        # Save to file (simplified - in real implementation would use proper serialization)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            import json

            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to load checkpoint from
        """
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        with open(checkpoint_path, "r") as f:
            import json

            checkpoint_data = json.load(f)

        self.current_step = checkpoint_data.get("current_step", 0)
        self.rewards_history = checkpoint_data.get("rewards_history", [])
        self.steps_history = checkpoint_data.get("steps_history", [])
        self.consecutive_failures = checkpoint_data.get("consecutive_failures", 0)
        self.last_gate_check_step = checkpoint_data.get("last_gate_check_step", 0)
        self.halt_reason = checkpoint_data.get("halt_reason")
        self.is_training = checkpoint_data.get("is_training", False)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
