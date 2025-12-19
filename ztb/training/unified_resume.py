"""
Unified Training Resume API

Provides a consistent interface for resuming training across different algorithms and scripts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from ztb.utils.checkpoint import TrainingStateManager

logger = logging.getLogger(__name__)


class ResumeCapableTrainer(Protocol):
    """Protocol for trainers that support resuming"""

    def resume_training(
        self,
        training_state_path: str,
        additional_timesteps: Optional[int] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resume training from a saved state"""
        ...


@dataclass
class ResumeOptions:
    """Configuration options for training resumption"""

    training_state_path: str
    additional_timesteps: Optional[int] = None
    override_config: Optional[Dict[str, Any]] = None
    validate_compatibility: bool = True
    auto_save_state: bool = True
    backup_existing: bool = True


class UnifiedResumeManager:
    """Unified manager for training resumption across different algorithms"""

    def __init__(self, training_state_dir: str = "models/training_states"):
        self.training_state_manager = TrainingStateManager(training_state_dir)
        self.training_state_dir = Path(training_state_dir)

    def resume_training(
        self, trainer: ResumeCapableTrainer, options: ResumeOptions
    ) -> Dict[str, Any]:
        """Resume training with unified interface"""

        try:
            # Validate training state exists
            if not Path(options.training_state_path).exists():
                raise FileNotFoundError(
                    f"Training state not found: {options.training_state_path}"
                )

            # Load and validate training state
            training_state = self.training_state_manager.load_training_state(
                options.training_state_path
            )

            if options.validate_compatibility and options.override_config:
                # Perform compatibility validation
                validation = self.training_state_manager.validate_resume_compatibility(
                    training_state, options.override_config
                )

                if not validation["compatible"]:
                    error_msg = "Resume validation failed: " + "; ".join(
                        validation["errors"]
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        logger.warning(f"Resume validation warning: {warning}")

            # Resume training
            logger.info(f"Resuming training from {options.training_state_path}")
            result = trainer.resume_training(
                options.training_state_path,
                options.additional_timesteps,
                options.override_config,
            )

            # Auto-save new training state if requested
            if options.auto_save_state and "model" in result:
                try:
                    new_state_path = self.training_state_manager.save_training_state(
                        model=result["model"],
                        total_timesteps=result.get("total_timesteps", 0),
                        episode_count=result.get("episode_count", 0),
                        episode_rewards=result.get("episode_rewards", []),
                        episode_lengths=result.get("episode_lengths", []),
                        config=options.override_config or {},
                        training_time=result.get("training_time", 0.0),
                    )
                    result["new_training_state_path"] = new_state_path
                    logger.info(f"Saved new training state to {new_state_path}")
                except Exception as e:
                    logger.warning(f"Failed to auto-save training state: {e}")

            return result

        except Exception as e:
            logger.error(f"Training resumption failed: {e}")
            raise

    def list_available_training_states(self) -> Dict[str, Any]:
        """List all available training states with metadata"""

        states = self.training_state_manager.list_training_states()

        # Group by algorithm/model type if possible
        grouped_states = {}
        for state in states:
            # Extract algorithm from config if available
            algorithm = "unknown"
            try:
                # This would need to be enhanced based on actual config structure
                algorithm = "SAC"  # Default assumption
            except Exception:
                pass

            if algorithm not in grouped_states:
                grouped_states[algorithm] = []

            grouped_states[algorithm].append(state)

        return {
            "total_states": len(states),
            "states_by_algorithm": grouped_states,
            "all_states": states,
        }

    def cleanup_old_training_states(self, keep_last: int = 5) -> Dict[str, Any]:
        """Clean up old training states, keeping only the most recent ones"""

        states = self.training_state_manager.list_training_states()

        if len(states) <= keep_last:
            return {
                "cleaned": 0,
                "message": f"Only {len(states)} states exist, no cleanup needed",
            }

        # Sort by timestamp (newest first) and remove old ones
        states.sort(key=lambda x: x["timestamp"], reverse=True)
        states_to_remove = states[keep_last:]

        removed_count = 0
        for state in states_to_remove:
            try:
                Path(state["filepath"]).unlink()
                removed_count += 1
                logger.info(f"Removed old training state: {state['filepath']}")
            except Exception as e:
                logger.warning(
                    f"Failed to remove training state {state['filepath']}: {e}"
                )

        return {
            "cleaned": removed_count,
            "kept": keep_last,
            "message": f"Cleaned up {removed_count} old training states",
        }


def create_resume_options(
    training_state_path: str,
    additional_timesteps: Optional[int] = None,
    override_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ResumeOptions:
    """Helper function to create ResumeOptions with sensible defaults"""

    return ResumeOptions(
        training_state_path=training_state_path,
        additional_timesteps=additional_timesteps,
        override_config=override_config,
        **kwargs,
    )


# Convenience functions for common use cases
def resume_sac_training(
    config_path: str,
    training_state_path: str,
    additional_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience function to resume SAC training"""

    # Load configuration
    import json

    from ztb.training.sac_trainer import SACTrainer

    with open(config_path, "r") as f:
        config = json.load(f)

    # Create trainer
    trainer = SACTrainer(config)

    # Create resume options
    options = create_resume_options(
        training_state_path=training_state_path,
        additional_timesteps=additional_timesteps,
        override_config=config,
    )

    # Create resume manager and resume training
    resume_manager = UnifiedResumeManager()
    return resume_manager.resume_training(trainer, options)


def resume_sac_v434_2_training(
    data_path: str,
    training_state_path: str,
    additional_timesteps: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to resume SAC v434.2 training"""

    # This would need to be implemented based on the specific trainer
    # For now, return a placeholder
    logger.info(f"Resuming SAC v434.2 training from {training_state_path}")
    return {
        "status": "not_implemented",
        "message": "SAC v434.2 resume not yet implemented",
    }
