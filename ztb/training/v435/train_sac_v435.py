#!/usr/bin/env python3
"""
SAC v435 Training Script
Enhanced SAC model with improved reward function and adaptive features
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_schema

logger = logging.getLogger(__name__)


class SACv435Trainer:
    """SAC v435 Enhanced Trainer"""

    def __init__(self, config_path: str = "config/v435/sac_v435_config.json"):
        """
        Initialize v435 trainer

        Args:
            config_path: Configuration file path
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.env = None

        logger.info("SAC v435 Trainer initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def setup_environment(self) -> DummyVecEnv:
        """Setup training environment"""
        logger.info("Setting up v435 environment")

        # Load data
        data_path = self.config["data"]["primary_dataset"]
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data: {len(df)} rows from {data_path}")

        # Create environment with v435 schema
        env = create_env_from_schema("sac_v435", df)

        self.env = DummyVecEnv([lambda: env])
        logger.info("Environment setup complete")

        return self.env

    def setup_model(self) -> SAC:
        """Setup SAC model with v435 configuration"""
        logger.info("Setting up v435 SAC model")

        training_config = self.config["training"]

        model_params = {
            "policy": "MlpPolicy",
            "env": self.env,
            "learning_rate": training_config["learning_rate"],
            "buffer_size": training_config["buffer_size"],
            "learning_starts": training_config["learning_starts"],
            "batch_size": training_config["batch_size"],
            "tau": training_config["tau"],
            "gamma": training_config["gamma"],
            "ent_coef": training_config["ent_coef"],
            "target_entropy": training_config["target_entropy"],
            "verbose": 1,
            "tensorboard_log": self.config["output"]["tensorboard_log"],
        }

        self.model = SAC(**model_params)
        logger.info("Model setup complete")

        return self.model

    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []

        output_dir = Path(self.config["output"]["model_dir"])

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="sac_v435",
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=10000,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        return callbacks

    def train(self) -> Dict[str, Any]:
        """Execute training"""
        logger.info("Starting v435 training")

        try:
            # Setup components
            self.setup_environment()
            self.setup_model()
            callbacks = self.setup_callbacks()

            # Training
            total_timesteps = self.config["training"]["total_timesteps"]
            logger.info(f"Training for {total_timesteps} timesteps")

            self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

            # Save final model
            output_dir = Path(self.config["output"]["model_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = output_dir / "sac_v435_final.zip"
            self.model.save(model_path)

            logger.info(f"Training complete. Model saved to {model_path}")

            result = {
                "status": "success",
                "model_path": str(model_path),
                "total_timesteps": total_timesteps,
                "config": self.config,
            }

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }


def main():
    """Main training function"""
    trainer = SACv435Trainer()
    result = trainer.train()

    # Save training results
    results_dir = Path(trainer.config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "training_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Training results saved to {results_file}")


if __name__ == "__main__":
    main()
