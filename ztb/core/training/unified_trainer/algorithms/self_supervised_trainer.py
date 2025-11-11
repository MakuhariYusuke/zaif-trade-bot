"""Self-supervised learning trainer implementation."""

import logging
import os
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch

from ztb.multimodal.pretraining import SelfSupervisedTrainer as SSPTrainer
from ztb.training.config.configuration_manager import ConfigurationManager
from ztb.training.unified_trainer.base.base_trainer import BaseAlgorithmTrainer
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback
from ztb.training.utils.training_stats import TrainingStats


class SelfSupervisedTrainer(BaseAlgorithmTrainer):
    """Self-supervised learning trainer with enhanced features from SACTrainer."""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        gradient_accumulation_steps: int = 1,
        system_optimizer: Optional[Any] = None,
        optimizer_tracker: Optional["OptimizerFeatureTracker"] = None,
    ):
        super().__init__(config, logger)

        # model will be instantiated later; annotate as optional to satisfy mypy
        self.model: Optional[SSPTrainer] = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.system_optimizer = system_optimizer
        self.optimizer_tracker = optimizer_tracker
        self.training_stats: TrainingStats = {}

    def validate_config(self) -> bool:
        """Validate self-supervised learning configuration using unified configuration manager."""
        try:
            # Use configuration manager for validation
            config_manager = ConfigurationManager(self.logger)

            # Validate the configuration
            errors = config_manager.validate_config_file(self.config, "training")
            if errors:
                for error in errors:
                    self.logger.error(f"Configuration validation error: {error}")
                return False

            # Additional SSP-specific validation
            ssp_config = self.config.get("training", {}).get("ssp_hyperparameters", {})
            if not ssp_config:
                self.logger.error("Missing SSP hyperparameters section")
                return False

            # Validate SSP-specific parameters
            required_ssp_params = ["learning_rate", "batch_size", "num_epochs"]
            for param in required_ssp_params:
                if param not in ssp_config:
                    self.logger.error(f"Missing SSP hyperparameter: {param}")
                    return False

            # Validate parameter types and ranges
            if (
                not isinstance(ssp_config.get("learning_rate"), (int, float))
                or ssp_config["learning_rate"] <= 0
            ):
                self.logger.error("learning_rate must be a positive number")
                return False

            if (
                not isinstance(ssp_config.get("batch_size"), int)
                or ssp_config["batch_size"] <= 0
            ):
                self.logger.error("batch_size must be a positive integer")
                return False

            if (
                not isinstance(ssp_config.get("num_epochs"), int)
                or ssp_config["num_epochs"] <= 0
            ):
                self.logger.error("num_epochs must be a positive integer")
                return False

            # Validate data file exists
            data_path = config_manager.get_config_value(
                self.config, "training.data_config.data_path"
            )
            if data_path and not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return False

            self.logger.info(
                "Self-supervised learning configuration validation successful"
            )
            return True

        except Exception as e:
            self.logger.error(f"SSP configuration validation failed: {e}")
            return False

    def train(self) -> bool:
        """Execute self-supervised learning training with unified error handling and structured logging."""
        return self.execute_training_pipeline(
            algorithm_name="Self-supervised learning",
            training_method=self._execute_self_supervised_training,
        )

    def _execute_self_supervised_training(
        self,
        total_timesteps: int,
        callback: TrainingProgressCallback,
        start_time: float,
    ) -> bool:
        """Execute core self-supervised learning training logic with structured logging."""
        # Load and prepare data
        data_config = self.config.get("training", {}).get("data_config", {})
        data_path = data_config.get("data_path", "data/btc_jpy_featured_dataset.csv")

        if not os.path.exists(data_path):
            raise DataError(f"Data file not found: {data_path}")

        self.log_structured_event("data", "loading", {"path": data_path})
        df = pd.read_csv(data_path)
        self.log_structured_event("data", "loaded", {"rows": len(df)})

        # Get SSP hyperparameters
        ssp_config = self.config.get("training", {}).get("ssp_hyperparameters", {})

        # Create self-supervised learning model
        self.log_structured_event(
            "model", "creation", {"algorithm": "SelfSupervised", "type": "SSPTrainer"}
        )
        from ztb.multimodal.pretraining.config import get_config as get_ssp_config

        # Get SSP configuration
        ssp_model_config = get_ssp_config()
        # Override with training config
        ssp_model_config.update(ssp_config)

        # Extract parameters for SSPTrainer initialization
        input_dim = ssp_model_config.get("input_dim", 156)
        device = ssp_model_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint_dir = ssp_model_config.get(
            "checkpoint_dir", "checkpoints/pretraining"
        )

        self.model = SSPTrainer(
            input_dim=input_dim,
            device=device,
            checkpoint_dir=checkpoint_dir,
            memory_manager=None,  # Will be set up separately if needed
        )

        # Training parameters
        num_epochs = ssp_config.get("num_epochs", 100)
        batch_size = ssp_config.get("batch_size", 32)
        total_steps = num_epochs * (len(df) // batch_size)  # Approximate total steps

        # Narrow self.model locally to help static analyzers and avoid repeated Optional access  # type: ignore[unreachable]
        model = self.model
        if model is None:
            raise ModelError("Model not initialized before training")

        # Set up dynamic LR scheduler with model optimizer if enabled
        if self.lr_scheduler and hasattr(model, "optimizer"):
            self.lr_scheduler.optimizer = model.optimizer
            self.log_structured_event("optimizer", "setup", {"scheduler": "dynamic_lr"})

        # Train the model
        self.log_structured_event(
            "training",
            "execution",
            {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "total_steps": total_steps,
            },
        )

        # Convert DataFrame to torch.Tensor
        train_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension
        val_tensor = train_tensor  # Use same data for validation for now

        model.train_all_stages(
            train_data=train_tensor, val_data=val_tensor, config=ssp_model_config
        )

        # Training completed
        training_time = time.time() - start_time

        # Clean up metrics collection
        self.cleanup_metrics_collection()

        # Cleanup training environment
        self.cleanup_training_environment()

        # Save model
        model_name = self.config.get("model_name", "ssp_model")
        model_path = self.save_model(model, model_name, ".pth")

        # Collect training statistics
        self.training_stats = self.collect_training_stats(
            training_time=training_time,
            total_timesteps=total_steps,
            model_path=model_path,
            steps_per_second=total_steps / training_time,
            status="completed",
        )

        self.log_training_completion(training_time, self.training_stats)
        return True

    def get_training_stats(self) -> Dict[str, Any]:
        """Get self-supervised learning training statistics."""
        return dict(self.training_stats)

    def load_model(self, model_path: str) -> bool:
        """Load a trained self-supervised learning model from file."""
        try:
            self.logger.info(f"Loading SSP model from {model_path}")

            # Create a new SSPTrainer instance and load the checkpoint
            ssp_config = self.config.get("training", {}).get("ssp_hyperparameters", {})
            input_dim = ssp_config.get("input_dim", 156)
            device = ssp_config.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
            checkpoint_dir = ssp_config.get("checkpoint_dir", "checkpoints/pretraining")

            self.model = SSPTrainer(
                input_dim=input_dim,
                device=device,
                checkpoint_dir=checkpoint_dir,
                memory_manager=None,
            )
            self.model.load_checkpoint(model_path)
            self.logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False

    def validate_training(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate trained self-supervised learning model."""
        try:
            # Use provided path or get from training stats
            validation_model_path: Optional[str] = None
            if model_path is None:
                validation_model_path = self.training_stats.get("model_path")

            if not validation_model_path or not os.path.exists(validation_model_path):
                return {
                    "validation_success": False,
                    "error": f"Model path not found: {validation_model_path}",
                }

            # Load model for validation
            if not self.load_model(validation_model_path):
                return {"validation_success": False, "error": "Failed to load model"}

            # Basic validation - model loaded successfully
            return {
                "validation_success": True,
                "model_path": validation_model_path,
                "algorithm": "SelfSupervised",
            }

        except Exception as e:
            self.logger.error(f"SSP validation failed: {e}")
            return {"validation_success": False, "error": str(e)}
