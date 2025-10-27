"""Training state management component."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.training.unified_trainer.trainer import UnifiedTrainer


class TrainingStateManager:
    """Manages training state across episodes and timesteps."""

    def __init__(self, trainer: "UnifiedTrainer"):
        """Initialize training state manager with reference to trainer."""
        self.trainer = trainer
        self.logger = get_logger(__name__)

        # Training progress state
        self.current_timestep = 0
        self.current_episode = 0
        self.total_timesteps = 0
        self.episodes_completed = 0

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.loss_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.policy_loss_history: List[float] = []

        # Rolling statistics
        self.reward_window_size = 100
        self.loss_window_size = 50

        # Checkpoint state
        self.last_checkpoint_step = 0
        self.best_reward = float('-inf')
        self.best_model_path: Optional[str] = None

        # Training metadata
        self.start_time = datetime.now()
        self.last_save_time = self.start_time

    def reset_episode_state(self) -> None:
        """Reset state for a new training episode."""
        self.current_episode += 1
        self.episode_rewards.append(0.0)
        self.episode_lengths.append(0)
        self.logger.debug(f"Started episode {self.current_episode}")

    def update_timestep(self, reward: float, loss: Optional[float] = None,
                       value_loss: Optional[float] = None, policy_loss: Optional[float] = None) -> None:
        """Update state for current timestep.

        Args:
            reward: Reward received this timestep
            loss: Total loss value
            value_loss: Value function loss
            policy_loss: Policy loss
        """
        self.current_timestep += 1
        self.total_timesteps += 1

        # Update episode statistics
        if self.episode_rewards:
            self.episode_rewards[-1] += reward
        if self.episode_lengths:
            self.episode_lengths[-1] += 1

        # Update loss history
        if loss is not None:
            self.loss_history.append(loss)
        if value_loss is not None:
            self.value_loss_history.append(value_loss)
        if policy_loss is not None:
            self.policy_loss_history.append(policy_loss)

        # Maintain rolling windows
        self._maintain_rolling_windows()

    def end_episode(self) -> None:
        """Mark current episode as completed."""
        self.episodes_completed += 1
        self.logger.debug(
            f"Episode {self.current_episode} completed: "
            f"reward={self.episode_rewards[-1]:.2f}, "
            f"length={self.episode_lengths[-1]}"
        )

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            Dictionary with training statistics
        """
        elapsed_time = datetime.now() - self.start_time

        stats = {
            "current_timestep": self.current_timestep,
            "current_episode": self.current_episode,
            "total_timesteps": self.total_timesteps,
            "episodes_completed": self.episodes_completed,
            "elapsed_time_seconds": elapsed_time.total_seconds(),
            "average_reward": np.mean(self.episode_rewards[-self.reward_window_size:]) if self.episode_rewards else 0.0,
            "average_episode_length": np.mean(self.episode_lengths[-self.reward_window_size:]) if self.episode_lengths else 0.0,
            "average_loss": np.mean(self.loss_history[-self.loss_window_size:]) if self.loss_history else 0.0,
            "best_reward": self.best_reward,
            "total_episodes": len(self.episode_rewards),
        }

        # Add loss components if available
        if self.value_loss_history:
            stats["average_value_loss"] = np.mean(self.value_loss_history[-self.loss_window_size:])
        if self.policy_loss_history:
            stats["average_policy_loss"] = np.mean(self.policy_loss_history[-self.loss_window_size:])

        return stats

    def should_checkpoint(self, checkpoint_interval: int) -> bool:
        """Check if it's time to create a checkpoint.

        Args:
            checkpoint_interval: Number of timesteps between checkpoints

        Returns:
            True if checkpoint should be created
        """
        return (self.total_timesteps - self.last_checkpoint_step) >= checkpoint_interval

    def update_checkpoint_state(self, model_path: str, current_reward: float) -> None:
        """Update checkpoint state after saving.

        Args:
            model_path: Path where model was saved
            current_reward: Current average reward
        """
        self.last_checkpoint_step = self.total_timesteps
        self.last_save_time = datetime.now()

        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_model_path = model_path
            self.logger.info(f"New best model saved: reward={current_reward:.2f}")

    def get_recent_performance(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """Get recent training performance metrics.

        Args:
            window_size: Size of rolling window (uses default if None)

        Returns:
            Dictionary with recent performance metrics
        """
        window = window_size or self.reward_window_size

        return {
            "recent_avg_reward": np.mean(self.episode_rewards[-window:]) if self.episode_rewards else 0.0,
            "recent_avg_length": np.mean(self.episode_lengths[-window:]) if self.episode_lengths else 0.0,
            "recent_avg_loss": np.mean(self.loss_history[-window:]) if self.loss_history else 0.0,
            "reward_std": np.std(self.episode_rewards[-window:]) if len(self.episode_rewards) >= window else 0.0,
            "length_std": np.std(self.episode_lengths[-window:]) if len(self.episode_lengths) >= window else 0.0,
        }

    def _maintain_rolling_windows(self) -> None:
        """Maintain rolling windows for performance tracking."""
        # Keep only recent data in loss histories
        max_history = max(self.reward_window_size, self.loss_window_size) * 2

        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]
        if len(self.value_loss_history) > max_history:
            self.value_loss_history = self.value_loss_history[-max_history:]
        if len(self.policy_loss_history) > max_history:
            self.policy_loss_history = self.policy_loss_history[-max_history:]

    def validate_state_consistency(self) -> bool:
        """Validate that training state is consistent.

        Returns:
            True if state is consistent, False otherwise
        """
        try:
            # Check episode counts
            if len(self.episode_rewards) != len(self.episode_lengths):
                self.logger.warning("Episode rewards and lengths arrays have different lengths")
                return False

            # Check timestep consistency
            expected_total = sum(self.episode_lengths)
            if abs(expected_total - self.total_timesteps) > 1:  # Allow small discrepancy
                self.logger.warning(
                    f"Total timesteps mismatch: expected {expected_total}, got {self.total_timesteps}"
                )
                return False

            # Check for invalid values
            if any(not np.isfinite(r) for r in self.episode_rewards):
                self.logger.warning("Non-finite values found in episode rewards")
                return False

            if any(l <= 0 for l in self.episode_lengths):
                self.logger.warning("Invalid episode lengths found")
                return False

            return True

        except Exception as e:
            self.logger.error(f"State validation error: {e}")
            return False