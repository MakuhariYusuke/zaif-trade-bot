"""
PPO Trainer with auto-halt functionality for training gates.
"""

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ztb.training.eval_gates import EvalGates, GateResult, GateStatus

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PPOTrainer:
    """PPO Trainer with evaluation gates and auto-halt functionality."""

    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any],
        checkpoint_dir: str,
        eval_gates: Optional[EvalGates] = None,
        halt_callback: Optional[Callable[[str], None]] = None,
        checkpoint_interval: int = 10000,
    ):
        """
        Initialize PPO trainer.

        Args:
            data_path: Path to training data
            config: Training configuration
            checkpoint_dir: Directory for checkpoints
            eval_gates: Evaluation gates for training validation
            halt_callback: Callback function called when training should halt
            checkpoint_interval: Steps between checkpoints
        """
        self.eval_gates = eval_gates or EvalGates()
        self.halt_callback = halt_callback
        self.checkpoint_interval = checkpoint_interval
        self.data_path = data_path
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.model: Optional[PPO] = None

        # Training state
        self.current_step = 0
        self.rewards_history: deque[float] = deque(
            maxlen=50000
        )  # Keep last 50k rewards for efficiency
        self.steps_history: deque[int] = deque(maxlen=50000)  # Keep last 50k steps
        self.is_training = False
        self.halt_reason: Optional[str] = None

        # Statistics for efficiency
        self.reward_sum = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0  # For variance calculation

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

        # Update history (deque automatically manages size)
        self.rewards_history.append(reward)
        self.steps_history.append(step)

        # Update online statistics (Welford's algorithm)
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2

        # Check gates periodically
        if step - self.last_gate_check_step >= self.checkpoint_interval:
            self._check_gates_and_halt_if_needed()

    def get_reward_stats(self) -> Dict[str, float]:
        """
        Get reward statistics efficiently.

        Returns:
            Dictionary with mean, variance, std, count
        """
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
                rewards=list(self.rewards_history),
                steps=list(self.steps_history),
                final_eval_reward=(
                    self.rewards_history[-1] if self.rewards_history else 0.0
                ),
                reward_stats=stats,  # Pass statistics for efficiency
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
            "rewards_history": list(
                self.rewards_history
            ),  # Convert deque to list for JSON serialization
            "steps_history": list(
                self.steps_history
            ),  # Convert deque to list for JSON serialization
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

    def _create_callback(self) -> BaseCallback:
        """Create training callback."""
        class TrainingCallback(BaseCallback):
            def __init__(self, trainer: "PPOTrainer"):
                super().__init__()
                self.trainer = trainer

            def _on_step(self) -> bool:
                if self.locals.get("done"):
                    reward = self.locals.get("rewards", 0)
                    self.trainer.update_progress(self.num_timesteps, reward)
                return not self.trainer.halt_reason

        return TrainingCallback(self)

    def train(self, session_id: str) -> PPO:
        """Train the PPO model."""
        if self.model is None:
            # Create PPO model
            env = self.config.get("env")
            if env is None:
                raise ValueError("env is required in config")
            self.model = PPO(
                policy=self.config.get("policy", "MlpPolicy"),
                env=env,
                learning_rate=self.config.get("learning_rate", 3e-4),
                n_steps=self.config.get("n_steps", 2048),
                batch_size=self.config.get("batch_size", 64),
                n_epochs=self.config.get("n_epochs", 10),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_range=self.config.get("clip_range", 0.2),
                clip_range_vf=self.config.get("clip_range_vf"),
                normalize_advantage=self.config.get("normalize_advantage", True),
                ent_coef=self.config.get("ent_coef", 0.0),
                vf_coef=self.config.get("vf_coef", 0.5),
                max_grad_norm=self.config.get("max_grad_norm", 0.5),
                use_sde=self.config.get("use_sde", False),
                sde_sample_freq=self.config.get("sde_sample_freq", -1),
                target_kl=self.config.get("target_kl"),
                tensorboard_log=self.config.get("tensorboard_log"),
                policy_kwargs=self.config.get("policy_kwargs"),
                verbose=self.config.get("verbose", 1),
                seed=self.config.get("seed"),
                device=self.config.get("device", "auto"),
                _init_setup_model=self.config.get("_init_setup_model", True),
            )

        # Start training session
        self.start_training()

        # Train the model
        total_timesteps = self.config.get("total_timesteps", 100000)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self._create_callback(),
            tb_log_name=session_id,
        )

        return self.model
