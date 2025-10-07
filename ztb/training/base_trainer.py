"""
Base trainer classes and protocols for training abstraction.

This module provides abstract base classes and protocols for different
types of trainers to reduce code duplication and improve maintainability.
"""

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from stable_baselines3.common.callbacks import BaseCallback

from ztb.training.eval_gates import EvalGates, GateResult, GateStatus
from ztb.training.trainer_params import TrainerParams
from ztb.types.generics import ConfigurableMixin, StatisticsTracker
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainerProtocol(Protocol):
    """Protocol for all trainer implementations."""

    def train(self, session_id: str) -> Any:
        """Train the model and return it."""
        ...

    def get_reward_stats(self) -> Dict[str, float]:
        """Get training reward statistics."""
        ...


class BaseTrainer(ABC, ConfigurableMixin[Dict[str, Any]]):
    """
    Abstract base class for all trainers.

    This class provides common functionality for training management,
    evaluation gates, checkpointing, and progress tracking.
    """

class BaseTrainer(ABC, ConfigurableMixin[Dict[str, Any]]):
    """
    Abstract base class for all trainers.

    This class provides common functionality for training management,
    evaluation gates, checkpointing, and progress tracking.
    """

    def __init__(
        self,
        params: TrainerParams,
    ):
        super().__init__(dict(params.config))  # ConfigurableMixin expects dict
        self.data_path = params.data_path
        self.checkpoint_dir = Path(params.checkpoint_dir)
        self.eval_gates = params.eval_gates or EvalGates()
        self.halt_callback = params.halt_callback
        self.checkpoint_interval = params.checkpoint_interval

        # Statistics tracking
        self.stats_tracker = StatisticsTracker[Dict[str, float]]()

        # Training state
        self.current_step = 0
        self.rewards_history: deque[float] = deque(maxlen=50000)
        self.steps_history: deque[int] = deque(maxlen=50000)
        self.is_training = False
        self.halt_reason: Optional[str] = None

        # Statistics for efficiency (Welford's online algorithm)
        self.reward_sum = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0

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
        """Update training progress and check evaluation gates."""
        if not self.is_training:
            return

        self.current_step = step
        self.rewards_history.append(reward)
        self.steps_history.append(step)

        # Record statistics using StatisticsTracker
        stats = {
            "step": step,
            "reward": reward,
            "mean_reward": self.reward_mean,
            "reward_std": (self.reward_m2 / self.reward_count)**0.5 if self.reward_count > 1 else 0.0
        }
        self.stats_tracker.update_statistics("training_progress", stats)

        # Update running statistics using Welford's online algorithm
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2

        # Check gates periodically
        if step - self.last_gate_check_step >= self.checkpoint_interval:
            self._check_gates_and_halt_if_needed()

    def get_reward_stats(self) -> Dict[str, float]:
        """Get training reward statistics."""
        if self.reward_count == 0:
            return {"mean": 0.0, "variance": 0.0, "std": 0.0, "count": 0}

        variance = self.reward_m2 / self.reward_count if self.reward_count > 1 else 0.0
        std = variance**0.5

        return {
            "mean": self.reward_mean,
            "variance": variance,
            "std": std,
            "count": self.reward_count,
        }

    def _check_gates_and_halt_if_needed(self) -> None:
        """Check evaluation gates and halt training if necessary."""
        if not self.eval_gates.enabled:
            return

        gate_results = self.eval_gates.evaluate_all(
            rewards=self.rewards_history,
            steps=self.steps_history,
            final_eval_reward=self.rewards_history[-1] if self.rewards_history else 0.0,
        )

        failed_gates = [r for r in gate_results.values() if r.status == GateStatus.FAIL]
        if failed_gates:
            self.consecutive_failures += 1
            logger.warning(f"Gate check failed: {len(failed_gates)} gates failed")
            if self._should_auto_halt(gate_results):
                self.stop_training("Auto-halt: gate failure")
        else:
            self.consecutive_failures = 0

        self.last_gate_check_step = self.current_step

    def _should_auto_halt(self, gate_results: Dict[str, GateResult]) -> bool:
        """Determine if training should be auto-halted based on gate results."""
        if not gate_results:
            return False

        # Critical gates that always cause halt
        critical_gates = ["memory_rss", "no_dup_steps"]
        for gate_name in critical_gates:
            if (
                gate_name in gate_results
                and gate_results[gate_name].status == GateStatus.FAIL
            ):
                return True

        # Halt after too many consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures: {self.consecutive_failures}")
            return True

        # Conditional halt for reward trend
        if "reward_trend_300k" in gate_results:
            trend_result = gate_results["reward_trend_300k"]
            if (
                trend_result.status == GateStatus.FAIL
                and self.consecutive_failures >= 2
            ):
                return True

        return False

    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        status: Dict[str, Any] = {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "halt_reason": self.halt_reason,
            "consecutive_failures": self.consecutive_failures,
            "reward_stats": self.get_reward_stats(),
        }

        if self.eval_gates.enabled and self.rewards_history:
            stats = self.get_reward_stats()
            gate_results = self.eval_gates.evaluate_all(
                rewards=self.rewards_history,
                steps=self.steps_history,
                final_eval_reward=(
                    self.rewards_history[-1] if self.rewards_history else 0.0
                ),
                reward_stats=stats,
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
        """Save training state to checkpoint file."""
        from ztb.utils.file_utils import safe_json_dump
        from ztb.utils.path_utils import ensure_dir

        checkpoint_data = {
            "current_step": self.current_step,
            "rewards_history": list(self.rewards_history),
            "steps_history": list(self.steps_history),
            "consecutive_failures": self.consecutive_failures,
            "last_gate_check_step": self.last_gate_check_step,
            "halt_reason": self.halt_reason,
            "is_training": self.is_training,
        }

        ensure_dir(Path(checkpoint_path).parent)
        safe_json_dump(checkpoint_data, Path(checkpoint_path), indent=2)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training state from checkpoint file."""
        from ztb.utils.file_utils import safe_json_load

        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint_data = safe_json_load(Path(checkpoint_path))
        self.current_step = checkpoint_data.get("current_step", 0)
        self.rewards_history = deque(checkpoint_data.get("rewards_history", []), maxlen=50000)
        self.steps_history = deque(checkpoint_data.get("steps_history", []), maxlen=50000)
        self.consecutive_failures = checkpoint_data.get("consecutive_failures", 0)
        self.last_gate_check_step = checkpoint_data.get("last_gate_check_step", 0)
        self.halt_reason = checkpoint_data.get("halt_reason")
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    @abstractmethod
    def _create_callback(self) -> BaseCallback:
        """Create training callback - must be implemented by subclasses."""
        pass

    @abstractmethod
    def train(self, session_id: str) -> Any:
        """Train the model - must be implemented by subclasses."""
        pass


class CheckpointMixin:
    """
    Mixin class providing checkpoint functionality.
    
    DEPRECATED: This functionality is now integrated into BaseTrainer.
    This class is kept for backward compatibility only.
    """

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save training state to checkpoint file."""
        # This method should be overridden by BaseTrainer
        raise NotImplementedError("save_checkpoint should be implemented by BaseTrainer")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training state from checkpoint file."""
        # This method should be overridden by BaseTrainer
        raise NotImplementedError("load_checkpoint should be implemented by BaseTrainer")