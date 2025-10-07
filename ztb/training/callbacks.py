"""
Base training callback classes for common functionality.

This module provides abstract base classes and common implementations
for training callbacks to reduce code duplication across training scripts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from ztb.training.base_trainer import BaseTrainer
    from rich.progress import Progress, TaskID


class BaseTrainingCallback(BaseCallback, ABC):
    """
    Abstract base class for training callbacks with common functionality.

    This class provides common attributes and methods for tracking training
    progress, episode statistics, and action distributions.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.action_counts: List[Dict[str, int]] = []
        self.portfolio_values: List[float] = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Default step handler - can be overridden by subclasses."""
        return True

    @abstractmethod
    def _on_rollout_end(self) -> None:
        """Abstract method for handling rollout end - must be implemented by subclasses."""
        pass

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return {
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "action_counts": self.action_counts.copy(),
            "portfolio_values": self.portfolio_values.copy(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.action_counts.clear()
        self.portfolio_values.clear()
        self.episode_count = 0


class SimpleTrainingCallback(BaseTrainingCallback):
    """
    Simple training callback for basic episode tracking.

    This callback tracks episode rewards, lengths, and basic action counts.
    """

    def _on_rollout_end(self) -> None:
        """Handle rollout end by logging episode info."""
        if not hasattr(self, 'locals') or 'rewards' not in self.locals:
            return

        # Log episode info
        episode_reward = sum(self.locals["rewards"])
        episode_length = len(self.locals["rewards"])
        actions = self.locals.get("actions", [])

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1

        # Basic action counting
        action_count: dict[str, int] = {}
        for action in actions:
            action_str = str(action)
            action_count[action_str] = action_count.get(action_str, 0) + 1

        self.action_counts.append(action_count)


class TradingTrainingCallback(BaseTrainingCallback):
    """
    Training callback specialized for trading environments.

    This callback includes portfolio value tracking and trading-specific metrics.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.position_sizes: List[float] = []
        self.trade_counts: List[int] = []

    def _on_rollout_end(self) -> None:
        """Handle rollout end with trading-specific metrics."""
        if not hasattr(self, 'locals') or 'rewards' not in self.locals:
            return

        # Basic episode tracking
        episode_reward = sum(self.locals["rewards"])
        episode_length = len(self.locals["rewards"])
        actions = self.locals.get("actions", [])

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1

        # Trading-specific metrics
        # Note: These would need to be extracted from the environment info
        # This is a placeholder for actual implementation
        portfolio_value = getattr(self, '_last_portfolio_value', 10000.0)
        position_size = getattr(self, '_last_position_size', 0.0)

        self.portfolio_values.append(portfolio_value)
        self.position_sizes.append(position_size)
        self.trade_counts.append(len(actions))  # Simplified trade count

        # Action counting with trading semantics
        action_count: dict[str, int] = {}
        for action in actions:
            if hasattr(action, '__iter__') and len(action) > 0:
                action_val = action[0] if isinstance(action, (list, tuple)) else action
            else:
                action_val = action

            if isinstance(action_val, (int, float)):
                if action_val > 0.1:
                    action_name = "BUY"
                elif action_val < -0.1:
                    action_name = "SELL"
                else:
                    action_name = "HOLD"
            else:
                action_name = str(action_val)

            action_count[action_name] = action_count.get(action_name, 0) + 1

        self.action_counts.append(action_count)

    def get_trading_stats(self) -> Dict[str, Any]:
        """Get trading-specific statistics."""
        base_stats = self.get_episode_stats()
        base_stats.update({
            "position_sizes": self.position_sizes.copy(),
            "trade_counts": self.trade_counts.copy(),
        })
        return base_stats


class ProgressTrainingCallback(BaseCallback):
    """
    Training callback with integrated progress tracking.
    
    This callback integrates with BaseTrainer to update training progress
    and optionally display a progress bar using the rich library.
    """

    def __init__(
        self,
        trainer: "BaseTrainer",
        enable_progress_bar: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.trainer = trainer
        self.enable_progress_bar = enable_progress_bar
        self.progress: Optional["Progress"] = None
        self.task_id: Optional["TaskID"] = None

    def _on_training_start(self) -> None:
        """Initialize progress bar when training starts."""
        if not self.enable_progress_bar:
            return

        try:
            from rich.console import Console
            from rich.progress import Progress

            console = Console()
            self.progress = Progress(console=console)
            
            # Get total timesteps from trainer config if available
            total_timesteps = 100000  # Default
            if hasattr(self.trainer, "config"):
                total_timesteps = self.trainer.config.get("total_timesteps", 100000)
            
            self.task_id = self.progress.add_task(
                "[green]Training...",
                total=total_timesteps,
                completed=0
            )
            self.progress.start()
        except ImportError:
            from ztb.utils.logging_utils import get_logger
            logger = get_logger(__name__)
            logger.warning("Rich not available, progress bar disabled")
            self.enable_progress_bar = False
        except Exception as e:
            from ztb.utils.logging_utils import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Failed to start progress bar: {e}")
            self.enable_progress_bar = False

    def _on_step(self) -> bool:
        """Update progress on each step."""
        # Update trainer progress
        if self.locals.get("done"):
            reward = self.locals.get("rewards", 0)
            if isinstance(reward, (list, tuple)) and len(reward) > 0:
                reward = reward[0]
            reward_float = float(reward) if isinstance(reward, (int, float)) else 0.0
            self.trainer.update_progress(self.num_timesteps, reward_float)

        # Update progress bar
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, completed=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        """Clean up progress bar when training ends."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None


class EntropyScheduleCallback(BaseCallback):
    """
    Callback for applying entropy coefficient scheduling during training.
    
    This callback supports various entropy schedules like cosine decay
    to gradually reduce exploration as training progresses.
    """

    def __init__(
        self,
        schedule_type: str = "cosine_decay",
        initial_ent_coef: float = 0.01,
        final_ent_coef: Optional[float] = None,
        total_timesteps: int = 100000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.schedule_type = schedule_type
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef if final_ent_coef is not None else initial_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        """Update entropy coefficient based on schedule."""
        if self.schedule_type == "cosine_decay":
            from ztb.training.policy_utils import apply_cosine_decay_entropy
            from sb3_contrib import MaskablePPO
            
            if isinstance(self.model, MaskablePPO):
                apply_cosine_decay_entropy(
                    self.model,
                    self.num_timesteps,
                    self.total_timesteps,
                    self.initial_ent_coef,
                    self.final_ent_coef,
                )
        
        return True


class CompositeTrainingCallback(BaseCallback):
    """
    Composite callback that combines multiple callbacks.
    
    This callback allows combining progress tracking, entropy scheduling,
    gradient probe guard, and trainer updates in a single callback instance.
    """

    def __init__(
        self,
        trainer: "BaseTrainer",
        enable_progress_bar: bool = True,
        enable_entropy_schedule: bool = False,
        entropy_schedule_type: str = "cosine_decay",
        initial_ent_coef: float = 0.01,
        final_ent_coef: Optional[float] = None,
        enable_grad_probe_guard: bool = False,
        grad_probe_config: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.trainer = trainer
        self.enable_progress_bar = enable_progress_bar
        self.enable_entropy_schedule = enable_entropy_schedule
        self.enable_grad_probe_guard = enable_grad_probe_guard
        
        # Progress tracking
        self.progress: Optional["Progress"] = None
        self.task_id: Optional["TaskID"] = None
        
        # Entropy scheduling
        self.schedule_type = entropy_schedule_type
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef if final_ent_coef is not None else initial_ent_coef
        self.total_timesteps = 100000  # Will be updated in _on_training_start
        
        # Gradient probe guard
        self.grad_probe_guard: Optional[Any] = None
        if self.enable_grad_probe_guard:
            try:
                from ztb.training.grad_probe_guard import GradProbeGuard, GradProbeConfig
                
                # Create config from dict if provided
                if grad_probe_config:
                    config = GradProbeConfig(**grad_probe_config)
                else:
                    config = GradProbeConfig()
                
                # Get checkpoint_dir and session_id from trainer
                checkpoint_dir = getattr(trainer, "checkpoint_dir", "checkpoints")
                session_id = getattr(trainer, "session_id", None)
                
                self.grad_probe_guard = GradProbeGuard(
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    session_id=session_id,
                    verbose=verbose,
                )
                
                from ztb.utils.logging_utils import get_logger
                logger = get_logger(__name__)
                logger.info("GradProbeGuard enabled")
                
            except ImportError as e:
                from ztb.utils.logging_utils import get_logger
                logger = get_logger(__name__)
                logger.warning(f"GradProbeGuard disabled: {e}")
                self.enable_grad_probe_guard = False

    def _on_training_start(self) -> None:
        """Initialize progress bar and entropy schedule."""
        # Get total timesteps from trainer config
        if hasattr(self.trainer, "config"):
            self.total_timesteps = self.trainer.config.get("total_timesteps", 100000)
        
        # Initialize progress bar
        if self.enable_progress_bar:
            try:
                from rich.console import Console
                from rich.progress import Progress

                console = Console()
                self.progress = Progress(console=console)
                self.task_id = self.progress.add_task(
                    "[green]Training...",
                    total=self.total_timesteps,
                    completed=0
                )
                self.progress.start()
            except (ImportError, Exception) as e:
                from ztb.utils.logging_utils import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Progress bar disabled: {e}")
                self.enable_progress_bar = False
        
        # Initialize grad probe guard
        if self.enable_grad_probe_guard and self.grad_probe_guard:
            self.grad_probe_guard.init_callback()
            self.grad_probe_guard.model = self.model
            self.grad_probe_guard.training_env = self.training_env
            self.grad_probe_guard.num_timesteps = self.num_timesteps

    def _on_step(self) -> bool:
        """Update progress, entropy coefficient, and check gradient probes on each step."""
        # Update trainer progress
        if self.locals.get("done"):
            reward = self.locals.get("rewards", 0)
            if isinstance(reward, (list, tuple)) and len(reward) > 0:
                reward = reward[0]
            reward_float = float(reward) if isinstance(reward, (int, float)) else 0.0
            self.trainer.update_progress(self.num_timesteps, reward_float)

        # Update progress bar
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, completed=self.num_timesteps)

        # Update entropy coefficient
        if self.enable_entropy_schedule and self.schedule_type == "cosine_decay":
            from ztb.training.policy_utils import apply_cosine_decay_entropy
            from sb3_contrib import MaskablePPO
            
            if isinstance(self.model, MaskablePPO):
                apply_cosine_decay_entropy(
                    self.model,
                    self.num_timesteps,
                    self.total_timesteps,
                    self.initial_ent_coef,
                    self.final_ent_coef,
                )
        
        # Check gradient probes
        if self.enable_grad_probe_guard and self.grad_probe_guard:
            self.grad_probe_guard.num_timesteps = self.num_timesteps
            self.grad_probe_guard.locals = self.locals
            
            # Call grad probe guard's _on_step
            if not self.grad_probe_guard._on_step():
                from ztb.utils.logging_utils import get_logger
                logger = get_logger(__name__)
                logger.error("🛑 Training halted by GradProbeGuard")
                return False  # Halt training

        return True

    def _on_training_end(self) -> None:
        """Clean up progress bar when training ends."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None
