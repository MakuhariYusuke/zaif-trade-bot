"""
Training UI Manager - Handles training user interface management.

This module separates UI-related logic from the main trainer class,
including progress display, status updates, and user interaction.
"""

from typing import Any

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class TrainingUIManager:
    """
    Manages training user interface and progress display.

    This class handles:
    - Progress bar management
    - Status message display
    - User interaction handling
    - UI state management
    """

    def __init__(self, logger: Any):
        """
        Initialize TrainingUIManager.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.ui = None  # Will be set during initialization
        self._progress_bars = {}
        self._status_messages = []

    def initialize_ui(self, ui_instance: Any) -> None:
        """
        Initialize UI instance.

        Args:
            ui_instance: UI instance to manage
        """
        self.ui = ui_instance

    def display_welcome_message(self, config: dict[str, Any]) -> None:
        """
        Display welcome message with training configuration.

        Args:
            config: Training configuration
        """
        try:
            if self.ui:
                self.ui.display_welcome(config)
        except Exception as e:
            self.logger.error(f"Failed to display welcome message: {e}")

    def display_training_start(self, algorithm: str, total_timesteps: int) -> None:
        """
        Display training start message.

        Args:
            algorithm: Training algorithm
            total_timesteps: Total training timesteps
        """
        try:
            if self.ui:
                self.ui.display_training_start(algorithm, total_timesteps)
        except Exception as e:
            self.logger.error(f"Failed to display training start: {e}")

    def update_progress(
        self,
        current_step: int,
        total_steps: int,
        metrics: dict[str, Any] | None = None
    ) -> None:
        """
        Update training progress.

        Args:
            current_step: Current training step
            total_steps: Total training steps
            metrics: Current training metrics
        """
        try:
            if self.ui:
                self.ui.update_progress(current_step, total_steps, metrics)
        except Exception as e:
            self.logger.error(f"Failed to update progress: {e}")

    def display_training_complete(
        self,
        final_metrics: dict[str, Any],
        training_time: float
    ) -> None:
        """
        Display training completion message.

        Args:
            final_metrics: Final training metrics
            training_time: Total training time
        """
        try:
            if self.ui:
                self.ui.display_training_complete(final_metrics, training_time)
        except Exception as e:
            self.logger.error(f"Failed to display training complete: {e}")

    def display_error(self, error_message: str, details: dict[str, Any] | None = None) -> None:
        """
        Display error message.

        Args:
            error_message: Error message
            details: Error details
        """
        try:
            if self.ui:
                self.ui.display_error(error_message, details)
            else:
                self.logger.error(f"Training error: {error_message}")
                if details:
                    self.logger.error(f"Error details: {details}")
        except Exception as e:
            self.logger.error(f"Failed to display error: {e}")

    def prompt_user_confirmation(self, message: str) -> bool:
        """
        Prompt user for confirmation.

        Args:
            message: Confirmation message

        Returns:
            True if user confirms, False otherwise
        """
        try:
            if self.ui:
                return self.ui.prompt_confirmation(message)
            else:
                # Default to True if no UI available
                self.logger.warning(f"No UI available for confirmation: {message}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to prompt confirmation: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up UI resources."""
        try:
            if self.ui and hasattr(self.ui, 'cleanup'):
                self.ui.cleanup()
        except Exception as e:
            self.logger.error(f"Failed to cleanup UI: {e}")
