"""
Mixins for enhancing trainer functionality.

This module provides mixin classes that can be added to trainers
to provide additional functionality like progress tracking.
"""

from typing import Optional, TYPE_CHECKING

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

logger = get_logger(__name__)


class ProgressTrackingMixin:
    """
    Mixin class providing rich progress bar functionality.
    
    This can be mixed into trainer classes to add visual progress tracking
    during training with the rich library.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.progress: Optional["Progress"] = None
        self.task_id: Optional["TaskID"] = None
        self._progress_enabled = True

    def enable_progress_bar(self) -> None:
        """Enable progress bar display."""
        self._progress_enabled = True

    def disable_progress_bar(self) -> None:
        """Disable progress bar display."""
        self._progress_enabled = False
        if self.progress:
            self.stop_progress_bar()

    def start_progress_bar(self, total_steps: int, description: str = "Training") -> None:
        """
        Start the progress bar.
        
        Args:
            total_steps: Total number of steps for the progress bar.
            description: Description text to display with the progress bar.
        """
        if not self._progress_enabled:
            return

        try:
            from rich.console import Console
            from rich.progress import Progress

            console = Console()
            self.progress = Progress(console=console)
            self.task_id = self.progress.add_task(
                f"[green]{description}...",
                total=total_steps,
                completed=0
            )
            self.progress.start()
            logger.debug(f"Progress bar started for {total_steps} steps")
        except ImportError:
            logger.warning("Rich library not available, progress bar disabled")
            self._progress_enabled = False
        except Exception as e:
            logger.warning(f"Failed to start progress bar: {e}")
            self._progress_enabled = False

    def update_progress_bar(self, completed: int) -> None:
        """
        Update the progress bar.
        
        Args:
            completed: Number of completed steps.
        """
        if not self._progress_enabled or not self.progress or self.task_id is None:
            return

        try:
            self.progress.update(self.task_id, completed=completed)
        except Exception as e:
            logger.debug(f"Failed to update progress bar: {e}")

    def stop_progress_bar(self) -> None:
        """Stop and close the progress bar."""
        if self.progress:
            try:
                self.progress.stop()
                logger.debug("Progress bar stopped")
            except Exception as e:
                logger.debug(f"Failed to stop progress bar: {e}")
            finally:
                self.progress = None
                self.task_id = None


class EntropyScheduleMixin:
    """
    Mixin class providing entropy coefficient scheduling.
    
    This can be mixed into trainer classes to add entropy coefficient
    scheduling during training (e.g., cosine decay).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._entropy_schedule_enabled = False
        self._entropy_schedule_type: Optional[str] = None
        self._initial_ent_coef = 0.0
        self._final_ent_coef = 0.0

    def configure_entropy_schedule(
        self,
        schedule_type: str,
        initial_ent_coef: float,
        final_ent_coef: Optional[float] = None,
    ) -> None:
        """
        Configure entropy coefficient scheduling.
        
        Args:
            schedule_type: Type of schedule ("cosine_decay", "linear", etc.)
            initial_ent_coef: Initial entropy coefficient value.
            final_ent_coef: Final entropy coefficient value. Defaults to initial value.
        """
        self._entropy_schedule_enabled = True
        self._entropy_schedule_type = schedule_type
        self._initial_ent_coef = initial_ent_coef
        self._final_ent_coef = final_ent_coef if final_ent_coef is not None else initial_ent_coef
        logger.info(
            f"Entropy schedule configured: {schedule_type}, "
            f"initial={initial_ent_coef:.4f}, final={self._final_ent_coef:.4f}"
        )

    def update_entropy_coefficient(self, current_step: int, total_steps: int) -> None:
        """
        Update entropy coefficient based on the configured schedule.
        
        Args:
            current_step: Current training step.
            total_steps: Total number of training steps.
        """
        if not self._entropy_schedule_enabled or not hasattr(self, "model"):
            return

        model = getattr(self, "model")
        if model is None:
            return

        from ztb.training.policy_utils import apply_cosine_decay_entropy

        if self._entropy_schedule_type == "cosine_decay":
            apply_cosine_decay_entropy(
                model,
                current_step,
                total_steps,
                self._initial_ent_coef,
                self._final_ent_coef,
            )
