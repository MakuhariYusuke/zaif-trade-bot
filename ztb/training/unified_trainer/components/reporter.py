"""
Training Reporter - Handles training reporting and logging.

This module separates reporting-related logic from the main trainer class,
including metrics logging, result formatting, and report generation.
"""

from typing import Any, Dict, List, Optional

from ztb.training.constants import ENV_EVAL_FREQUENCY
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingReporter:
    """
    Manages training reporting and result formatting.

    This class handles:
    - Training metrics logging
    - Result formatting and display
    - Report generation
    - Performance statistics calculation
    """

    def __init__(self, logger: Any):
        """
        Initialize TrainingReporter.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self._training_history = []
        self._metrics_buffer = []

    def log_training_start(
        self, algorithm: str, config: Dict[str, Any], total_timesteps: int
    ) -> None:
        """
        Log training start event.

        Args:
            algorithm: Training algorithm
            config: Training configuration
            total_timesteps: Total training timesteps
        """
        try:
            start_info = {
                "event": "training_start",
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "config": config,
                "timestamp": self._get_timestamp(),
            }

            self._training_history.append(start_info)
            self.logger.info(
                f"Training started: {algorithm} with {total_timesteps} timesteps"
            )

        except Exception as e:
            self.logger.error(f"Failed to log training start: {e}")

    def log_training_progress(
        self, step: int, total_steps: int, metrics: Dict[str, Any]
    ) -> None:
        """
        Log training progress.

        Args:
            step: Current training step
            total_steps: Total training steps
            metrics: Current metrics
        """
        try:
            progress_info = {
                "event": "training_progress",
                "step": step,
                "total_steps": total_steps,
                "metrics": metrics,
                "timestamp": self._get_timestamp(),
            }

            self._training_history.append(progress_info)
            self._metrics_buffer.append(metrics)

            # Log significant milestones
            if step % (total_steps // 10) == 0 or step == total_steps:
                progress_pct = (step / total_steps) * 100
                self.logger.info(
                    f"Training progress: {progress_pct:.1f}% ({step}/{total_steps})"
                )

            # Log evaluation milestones
            if step % ENV_EVAL_FREQUENCY == 0:
                self.logger.info(f"Evaluation checkpoint at step {step}")

        except Exception as e:
            self.logger.error(f"Failed to log training progress: {e}")

    def log_training_complete(
        self,
        final_metrics: Dict[str, Any],
        training_time: float,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Log training completion.

        Args:
            final_metrics: Final training metrics
            training_time: Total training time
            model_path: Path to saved model
        """
        try:
            complete_info = {
                "event": "training_complete",
                "final_metrics": final_metrics,
                "training_time": training_time,
                "model_path": model_path,
                "timestamp": self._get_timestamp(),
            }

            self._training_history.append(complete_info)
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Final metrics: {final_metrics}")

        except Exception as e:
            self.logger.error(f"Failed to log training complete: {e}")

    def log_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log training error.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        try:
            error_info = {
                "event": "training_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "timestamp": self._get_timestamp(),
            }

            self._training_history.append(error_info)
            self.logger.error(f"Training error: {error}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")

    def generate_training_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report.

        Returns:
            Training report dictionary
        """
        try:
            if not self._training_history:
                return {"error": "No training history available"}

            report = {
                "training_summary": self._generate_summary(),
                "performance_metrics": self._calculate_performance_metrics(),
                "training_history": self._training_history.copy(),
                "generated_at": self._get_timestamp(),
            }

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate training report: {e}")
            return {"error": f"Report generation failed: {e}"}

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate training summary."""
        if not self._training_history:
            return {}

        start_events = [
            e for e in self._training_history if e["event"] == "training_start"
        ]
        complete_events = [
            e for e in self._training_history if e["event"] == "training_complete"
        ]

        summary = {
            "total_sessions": len(start_events),
            "completed_sessions": len(complete_events),
        }

        if complete_events:
            latest_complete = complete_events[-1]
            summary.update(
                {
                    "last_training_time": latest_complete.get("training_time"),
                    "final_metrics": latest_complete.get("final_metrics"),
                }
            )

        return summary

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from training history."""
        if not self._metrics_buffer:
            return {}

        try:
            # Aggregate metrics across training
            all_rewards = []
            all_losses = []

            for metrics in self._metrics_buffer:
                if "reward" in metrics:
                    all_rewards.append(metrics["reward"])
                if "loss" in metrics:
                    all_losses.append(metrics["loss"])

            perf_metrics = {}

            if all_rewards:
                perf_metrics.update(
                    {
                        "avg_reward": sum(all_rewards) / len(all_rewards),
                        "max_reward": max(all_rewards),
                        "min_reward": min(all_rewards),
                        "reward_volatility": self._calculate_volatility(all_rewards),
                    }
                )

            if all_losses:
                perf_metrics.update(
                    {
                        "avg_loss": sum(all_losses) / len(all_losses),
                        "final_loss": all_losses[-1] if all_losses else None,
                        "loss_volatility": self._calculate_volatility(all_losses),
                    }
                )

            return perf_metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return {}

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of values."""
        if len(values) < 2:
            return 0.0

        # calculate_volatility expects prices and calculates returns std.
        # Here we want std of raw values (losses).
        # So we should use numpy directly or a simple std function.
        import numpy as np

        return float(np.std(values))

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from ztb.training.utils.common_utils import get_timestamp

        return get_timestamp()

    def clear_history(self) -> None:
        """Clear training history and metrics buffer."""
        self._training_history.clear()
        self._metrics_buffer.clear()
        self.logger.info("Training history cleared")
