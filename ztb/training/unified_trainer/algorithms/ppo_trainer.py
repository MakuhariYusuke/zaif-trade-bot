"""PPO algorithm trainer implementation."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ztb.training.config.configuration_manager import ConfigurationManager
from ztb.training.environments.environment_config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.training.unified_trainer.base.base_trainer import BaseAlgorithmTrainer
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback
from ztb.training.utils.distributed_training import get_distributed_info
from ztb.training.utils.training_stats import TrainingStats

# from stable_baselines3.common.monitor import Monitor  # Removed to prevent reward corruption


class PPOTrainer(BaseAlgorithmTrainer):
    """PPO algorithm trainer with enhanced features from SACTrainer."""

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
        self.model: Optional[PPO] = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.system_optimizer = system_optimizer
        self.optimizer_tracker = optimizer_tracker
        self.training_stats: TrainingStats = {}

    def validate_config(self) -> bool:
        """Validate PPO configuration using unified configuration manager."""
        try:
            # Use configuration manager for validation
            config_manager = ConfigurationManager(self.logger)

            # Validate the configuration
            training_schema = config_manager.schemas.get("training")
            if training_schema:
                errors = training_schema.validate(self.config)
                if errors:
                    for error in errors:
                        self.logger.error(f"Configuration validation error: {error}")
                    return False

            # Additional PPO-specific validation
            ppo_config = self.config.get("training", {}).get("ppo_hyperparameters", {})
            if not ppo_config:
                self.logger.error("Missing PPO hyperparameters section")
                return False

            # Validate PPO-specific parameters
            required_ppo_params = ["learning_rate", "n_steps", "batch_size", "n_epochs"]
            for param in required_ppo_params:
                if param not in ppo_config:
                    self.logger.error(f"Missing PPO hyperparameter: {param}")
                    return False

            # Validate parameter types and ranges
            if (
                not isinstance(ppo_config.get("learning_rate"), (int, float))
                or ppo_config["learning_rate"] <= 0
            ):
                self.logger.error("learning_rate must be a positive number")
                return False

            if (
                not isinstance(ppo_config.get("batch_size"), int)
                or ppo_config["batch_size"] <= 0
            ):
                self.logger.error("batch_size must be a positive integer")
                return False

            if (
                not isinstance(ppo_config.get("n_steps"), int)
                or ppo_config["n_steps"] <= 0
            ):
                self.logger.error("n_steps must be a positive integer")
                return False

            if (
                not isinstance(ppo_config.get("n_epochs"), int)
                or ppo_config["n_epochs"] <= 0
            ):
                self.logger.error("n_epochs must be a positive integer")
                return False

            # Validate data file exists
            data_path = config_manager.get_config_value(
                self.config, "training.data_config.data_path"
            )
            if data_path and not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return False

            self.logger.info("PPO configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"PPO configuration validation failed: {e}")
            return False

    def train(self) -> bool:
        """Execute PPO training with unified error handling and structured logging."""
        return self.execute_training_pipeline(
            algorithm_name="PPO", training_function=self._execute_ppo_training
        )

    def _execute_ppo_training(
        self,
        total_timesteps: int,
        callback: TrainingProgressCallback,
        start_time: float,
    ) -> bool:
        """Execute core PPO training logic with structured logging."""
        # Load and prepare data
        data_config = self.config.get("training", {}).get("data_config", {})
        data_path = data_config.get("data_path", "data/btc_jpy_featured_dataset.csv")

        if not os.path.exists(data_path):
            raise DataError(f"Data file not found: {data_path}")

        self.log_structured_event("data", "loading", {"path": data_path})
        df = pd.read_csv(data_path)
        self.log_structured_event("data", "loaded", {"rows": len(df)})

        # Create environment configuration from unified config
        env_config_dict = (
            self.config.get("training", {}).get("environment", {}).get("config", {})
        )
        # Note: behavior_optimization, market_regime, and dynamic_reward_shaping are handled by reward calculator
        env_config = (
            EnvironmentConfig.from_dict(env_config_dict)
            if env_config_dict
            else EnvironmentConfig()
        )

        # Create environment
        self.log_structured_event(
            "environment", "creation", {"type": "HeavyTradingEnv"}
        )
        env = HeavyTradingEnv(data=df, config=env_config)
        # Remove Monitor wrapper to prevent reward corruption from Pendulum environment
        # wrapped_env: Monitor = Monitor(env)
        wrapped_env = env

        # Get PPO hyperparameters
        ppo_config = self.config.get("training", {}).get("ppo_hyperparameters", {})

        # Create PPO model
        self.log_structured_event(
            "model", "creation", {"algorithm": "PPO", "policy": "MlpPolicy"}
        )
        self.model = PPO(
            "MlpPolicy",
            wrapped_env,
            learning_rate=ppo_config.get("learning_rate", 0.0003),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.0),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            verbose=0,  # We'll handle logging ourselves
        )

        # Check for distributed training
        dist_info = get_distributed_info()
        if dist_info["is_distributed"]:
            adjusted_timesteps = total_timesteps // dist_info["world_size"]
            self.log_structured_event(
                "distributed",
                "setup",
                {
                    "rank": dist_info["rank"],
                    "world_size": dist_info["world_size"],
                    "original_timesteps": total_timesteps,
                    "adjusted_timesteps": adjusted_timesteps,
                },
            )
            total_timesteps = adjusted_timesteps

        # Narrow self.model locally to help static analyzers and avoid repeated Optional access  # type: ignore[unreachable]
        model = self.model
        if model is None:
            raise ModelError("Model not initialized before training")

        # Set up dynamic LR scheduler with model optimizer if enabled
        if (
            self.lr_scheduler
            and hasattr(model, "policy")
            and hasattr(model.policy, "optimizer")
        ):
            self.lr_scheduler.optimizer = model.policy.optimizer
            self.log_structured_event("optimizer", "setup", {"scheduler": "dynamic_lr"})

        # Execute training
        self.log_structured_event(
            "training", "execution", {"timesteps": total_timesteps}
        )
        model.learn(
            total_timesteps=total_timesteps, callback=callback, progress_bar=True
        )

        # Training completed
        training_time = time.time() - start_time

        # Clean up metrics collection
        self.cleanup_metrics_collection()

        # Cleanup training environment
        self.cleanup_training_environment()

        # Save model
        model_name = self.config.get("model_name", "ppo_model")
        model_path = self.save_model(model, model_name, ".zip")

        # Collect training statistics
        self.training_stats = self.collect_training_stats(
            training_time=training_time,
            total_timesteps=total_timesteps,
            model_path=model_path,
            steps_per_second=total_timesteps / training_time,
            final_reward=callback.reward_history[-1] if callback.reward_history else 0,
            action_distribution=self._calculate_final_action_distribution(callback),
        )

        self.log_training_completion(training_time, self.training_stats)
        return True

    def get_training_stats(self) -> Dict[str, Any]:
        """Get PPO training statistics."""
        return dict(self.training_stats)

    def load_model(self, model_path: str) -> bool:
        """Load a trained PPO model from file."""
        try:
            self.logger.info(f"Loading PPO model from {model_path}")
            self.model = PPO.load(model_path)
            self.logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False

    def validate_training(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate trained PPO model."""
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
                "algorithm": "PPO",
            }

        except Exception as e:
            self.logger.error(f"PPO validation failed: {e}")
            return {"validation_success": False, "error": str(e)}

    def _calculate_final_action_distribution(
        self, callback: TrainingProgressCallback
    ) -> Dict[str, float]:
        """Calculate final action distribution from callback data."""
        if not callback.discrete_actions:
            return {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

        total_actions = len(callback.discrete_actions)

        # Convert discrete actions to proper indices (SELL: -1 -> 2, HOLD: 0 -> 0, BUY: 1 -> 1)
        discrete_indices: List[int] = []
        for action in callback.discrete_actions:
            if action == -1:  # SELL
                discrete_indices.append(2)
            elif action == 0:  # HOLD
                discrete_indices.append(0)
            elif action == 1:  # BUY
                discrete_indices.append(1)
            else:
                # Guard unexpected values by mapping to HOLD
                discrete_indices.append(0)

        discrete_counts = np.bincount(discrete_indices, minlength=3)

        return {
            "HOLD": discrete_counts[0] / total_actions,
            "BUY": discrete_counts[1] / total_actions,
            "SELL": discrete_counts[2] / total_actions,
        }

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze training results and provide comprehensive summary."""
        try:
            self.logger.info("Analyzing PPO training results...")

            # Get final action distribution from callback if available
            action_distribution = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
            regime_distributions = {}

            # Try to get callback data from training stats
            if hasattr(self, "training_stats") and self.training_stats:
                callback = self.training_stats.get("callback")
                if callback and hasattr(callback, "discrete_actions"):
                    action_distribution = self._calculate_final_action_distribution(
                        callback
                    )

                    # Get regime-specific distributions if available
                    if (
                        hasattr(callback, "regime_action_counts")
                        and callback.regime_action_counts
                    ):
                        for regime, counts in callback.regime_action_counts.items():
                            total_regime_actions = sum(counts)
                            if total_regime_actions > 0:
                                regime_distributions[regime] = {
                                    "BUY": counts[0] / total_regime_actions,
                                    "SELL": counts[1] / total_regime_actions,
                                    "HOLD": counts[2] / total_regime_actions,
                                    "total_actions": total_regime_actions,
                                }

            # Calculate training metrics
            training_metrics = {
                "algorithm": "PPO",
                "final_action_distribution": action_distribution,
                "regime_distributions": regime_distributions,
                "total_training_steps": self.training_stats.get("total_steps", 0)
                if hasattr(self, "training_stats")
                else 0,
                "training_time": self.training_stats.get("training_time", 0)
                if hasattr(self, "training_stats")
                else 0,
            }

            # Log analysis results
            self.logger.info(
                f"Final action distribution: HOLD={action_distribution['HOLD']:.1%}, "
                f"BUY={action_distribution['BUY']:.1%}, SELL={action_distribution['SELL']:.1%}"
            )

            if regime_distributions:
                self.logger.info("Per-regime action distributions:")
                for regime, dist in regime_distributions.items():
                    self.logger.info(
                        f"  {regime}: HOLD={dist['HOLD']:.1%}, BUY={dist['BUY']:.1%}, "
                        f"SELL={dist['SELL']:.1%} ({dist['total_actions']} actions)"
                    )

            return training_metrics

        except Exception as e:
            self.logger.error(f"Failed to analyze PPO results: {e}")
            return {"error": str(e)}
