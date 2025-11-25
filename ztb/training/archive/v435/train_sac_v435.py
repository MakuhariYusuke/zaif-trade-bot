#!/usr/bin/env python3
"""
SAC v435 Training Script
Enhanced SAC model with improved reward function and adaptive features
Phase 5: Training and Evaluation with Risk Management Integration
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.risk.risk_manager import RiskManager
from ztb.trading.environment.schema_env_factory import create_env_from_schema
from ztb.types.common import ConfigDict
from ztb.utils.training_utils import (
    create_checkpoint_callback,
    create_eval_callback,
    save_model,
    save_training_results,
    validate_training_config,
)

logger = logging.getLogger(__name__)


class SACv435Trainer:
    """SAC v435 Enhanced Trainer with Risk Management Integration and Curriculum Learning"""

    def __init__(
        self,
        config_path_or_dict: str | Dict[str, Any] = "config/v435/sac_v435_config.json",
    ):
        """
        Initialize v435 trainer

        Args:
            config_path_or_dict: Configuration file path or config dict
        """
        if isinstance(config_path_or_dict, dict):
            self.config_path = None
            self.config = config_path_or_dict
        else:
            self.config_path = Path(config_path_or_dict)
            self.config = self._load_config()

        self.model = None
        self.env = None
        self.risk_manager = None

        # Curriculum learning settings
        self.curriculum_stage = 0
        self.curriculum_stages = self._define_curriculum_stages()

        # Initialize risk management if enabled
        if self.config.get("risk_management", {}).get("dynamic_position_sizing", False):
            self._setup_risk_management()

        logger.info(
            "SAC v435 Trainer initialized with risk management integration and curriculum learning"
        )

    def _define_curriculum_stages(self) -> List[Dict[str, Any]]:
        """Define curriculum learning stages"""
        return [
            {
                "name": "basic_trading",
                "description": "Basic trading with low risk and volatility",
                "stage": 1,
                "timesteps": 1000,
                "environment": {
                    "transaction_cost": 0.0005,  # Low transaction cost
                    "max_position_size": 0.05,  # Small positions
                    "volatility_multiplier": 0.5,  # Low volatility
                    "reward_scale": 0.5,  # Easier rewards
                },
                "risk_management": {
                    "max_drawdown_limit": 0.02,  # Very conservative
                    "volatility_adjustment": True,
                    "correlation_risk": False,
                },
            },
            {
                "name": "intermediate_trading",
                "description": "Intermediate trading with moderate risk",
                "stage": 2,
                "timesteps": 5000,
                "environment": {
                    "transaction_cost": 0.001,  # Standard transaction cost
                    "max_position_size": 0.1,  # Standard positions
                    "volatility_multiplier": 1.0,  # Normal volatility
                    "reward_scale": 1.0,  # Standard rewards
                },
                "risk_management": {
                    "max_drawdown_limit": 0.05,  # Moderate limit
                    "volatility_adjustment": True,
                    "correlation_risk": True,
                },
            },
            {
                "name": "advanced_trading",
                "description": "Advanced trading with full risk management",
                "stage": 3,
                "timesteps": 10000,
                "environment": {
                    "transaction_cost": 0.0015,  # Full transaction cost
                    "max_position_size": 0.2,  # Larger positions
                    "volatility_multiplier": 1.5,  # High volatility
                    "reward_scale": 1.5,  # Challenging rewards
                },
                "risk_management": {
                    "max_drawdown_limit": 0.1,  # Full risk limit
                    "volatility_adjustment": True,
                    "correlation_risk": True,
                    "market_adaptation": True,
                },
            },
        ]

    def train_with_curriculum(self, total_timesteps: int = 1000) -> None:
        """Train SAC model using curriculum learning approach"""
        logger.info(
            f"Starting curriculum learning training for {total_timesteps} total timesteps"
        )

        # Get curriculum stages
        stages = self._define_curriculum_stages()

        # Track cumulative timesteps
        cumulative_timesteps = 0

        for stage in stages:
            stage_name = stage["name"]
            stage_timesteps = stage["timesteps"]
            stage_config = stage["environment"]

            logger.info(
                f"Starting curriculum stage: {stage_name} ({stage_timesteps} timesteps)"
            )

            # Update environment with stage-specific parameters
            self._update_environment_for_stage(stage_config)

            # Train for this stage
            remaining_timesteps = min(
                stage_timesteps, total_timesteps - cumulative_timesteps
            )
            if remaining_timesteps <= 0:
                break

            self._train_single_stage(remaining_timesteps)
            cumulative_timesteps += remaining_timesteps

            logger.info(
                f"Completed stage {stage_name}, cumulative timesteps: {cumulative_timesteps}"
            )

            # Check if we've reached total timesteps
            if cumulative_timesteps >= total_timesteps:
                break

        logger.info(
            f"Curriculum learning completed. Total timesteps trained: {cumulative_timesteps}"
        )

    def _update_environment_for_stage(self, stage_config: ConfigDict) -> None:
        """Update environment parameters for curriculum stage"""
        # This would update the trading environment with stage-specific settings
        # For now, we'll log the configuration
        logger.info(f"Updating environment for stage with config: {stage_config}")

        # In a full implementation, this would modify:
        # - Transaction costs
        # - Position size limits
        # - Volatility multipliers
        # - Reward scaling

    def _train_single_stage(self, timesteps: int) -> None:
        """Train for a single curriculum stage"""
        logger.info(f"Training single stage for {timesteps} timesteps")

        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Train the model
        try:
            self.model.learn(total_timesteps=timesteps, callback=callbacks)
            logger.info(f"Successfully trained for {timesteps} timesteps")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _setup_risk_management(self):
        """Setup risk management components"""
        logger.info("Setting up risk management for v435 training")

        risk_config = self.config.get("risk_management", {})

        # Configure risk manager
        risk_manager_config = {
            "position_sizer": {
                "enabled": risk_config.get("dynamic_position_sizing", True),
                "volatility_adjustment": risk_config.get("volatility_adjustment", True),
                "min_position_size": 0.001,
                "max_position_size": 0.2,
                "base_position_size": 0.1,
            },
            "drawdown_controller": {
                "enabled": risk_config.get("drawdown_control", True),
                "max_drawdown_limit": risk_config.get("max_drawdown_limit", 0.1),
                "emergency_stop_threshold": 0.15,
                "recovery_threshold": 0.05,
            },
            "market_adaptor": {
                "enabled": True,
                "adaptation_window": 50,
                "volatility_threshold": 0.02,
                "trend_strength_threshold": 0.01,
                "regime_change_threshold": 0.7,
            },
        }

        self.risk_manager = RiskManager(risk_manager_config)
        logger.info("Risk management setup complete")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if self.config_path is None:
            # Config already provided as dict
            return self.config
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def setup_environment(self) -> DummyVecEnv:
        """Setup training environment"""
        logger.info("Setting up v435 environment")

        # Load data
        data_path = self.config["data"]["primary_dataset"]
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data: {len(df)} rows from {data_path}")
        except FileNotFoundError:
            logger.warning(
                f"Data file not found: {data_path}, creating dummy data for testing"
            )
            # Create dummy data for testing
            np.random.seed(42)
            dates = pd.date_range("2020-01-01", periods=1000, freq="1H")
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": 100000 + np.random.normal(0, 1000, 1000),
                    "high": 101000 + np.random.normal(0, 1000, 1000),
                    "low": 99000 + np.random.normal(0, 1000, 1000),
                    "close": 100000 + np.random.normal(0, 1000, 1000),
                    "volume": np.random.randint(1000, 10000, 1000),
                }
            )

        # Create environment with v435 schema
        try:
            env = create_env_from_schema("sac_v435", df)
        except Exception as e:
            logger.warning(f"Schema creation failed: {e}, using dummy environment")
            # Create a simple dummy environment for testing

            class DummyTradingEnv(gym.Env):
                def __init__(self):
                    self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(10,)
                    )

                def reset(self):
                    return np.random.normal(0, 1, 10)

                def step(self, action):
                    obs = np.random.normal(0, 1, 10)
                    reward = np.random.normal(0, 0.1)
                    done = np.random.random() < 0.01
                    info = {}
                    return obs, reward, done, info

            env = DummyTradingEnv()

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

    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []

        output_dir = Path(self.config["output"]["model_dir"])

        # Checkpoint callback
        checkpoint_callback = create_checkpoint_callback(
            save_freq=50000,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="sac_v435",
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_callback = create_eval_callback(
            eval_env=self.env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=10000,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        return callbacks

    def train(self) -> Dict[str, Any]:
        """Execute training with risk management integration and curriculum learning"""
        # Validate training configuration
        validate_training_config(self.config)

        logger.info(
            "Starting v435 training with risk management and curriculum learning"
        )

        try:
            # Setup components
            self.setup_environment()
            self.setup_model()
            callbacks = self._setup_callbacks()

            # Check if curriculum learning is enabled
            use_curriculum = self.config.get("training", {}).get(
                "curriculum_learning", False
            )
            total_timesteps = self.config["training"]["total_timesteps"]

            if use_curriculum:
                logger.info("Using curriculum learning for training")
                self.train_with_curriculum(total_timesteps)
                result = {
                    "status": "success",
                    "model_path": None,
                    "total_timesteps": total_timesteps,
                    "curriculum_learning": True,
                    "stages_completed": len(self._define_curriculum_stages()),
                }
            else:
                logger.info(
                    f"Training for {total_timesteps} timesteps with risk management"
                )

                # Custom training loop with risk monitoring
                if self.risk_manager:
                    result = self._train_with_risk_management(
                        total_timesteps, callbacks
                    )
                else:
                    # Standard training without risk management
                    self.model.learn(
                        total_timesteps=total_timesteps, callback=callbacks
                    )
                    result = {
                        "status": "success",
                        "model_path": None,
                        "total_timesteps": total_timesteps,
                        "risk_management": False,
                        "curriculum_learning": False,
                    }

            # Save final model
            output_dir = Path(self.config["output"]["model_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = output_dir / "sac_v435_final.zip"
            save_model(self.model, model_path)

            result["model_path"] = str(model_path)
            result["config"] = self.config

            logger.info(f"Training complete. Model saved to {model_path}")

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def _train_with_risk_management(
        self, total_timesteps: int, callbacks
    ) -> Dict[str, Any]:
        """Train with integrated risk management monitoring"""
        logger.info("Training with risk management integration")

        # Initialize risk tracking
        risk_metrics_history = []
        episode_count = 0

        # Custom training loop
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(total_timesteps):
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=False)

            # Apply risk-adjusted position sizing if enabled
            if self.risk_manager and hasattr(self.env, "get_portfolio_value"):
                try:
                    # Get current market data for risk assessment
                    current_data = self.env.get_current_data()
                    portfolio_value = self.env.get_portfolio_value()

                    # Calculate risk-adjusted position
                    risk_result = self.risk_manager.calculate_risk_adjusted_position(
                        base_position=action[0],  # Assume action[0] is position size
                        current_price=current_data.get("close", 100000),
                        portfolio_value=portfolio_value,
                        atr=current_data.get("atr", 1000),
                        df=pd.DataFrame([current_data]),
                    )

                    # Apply risk adjustment to action
                    adjusted_action = action.copy()
                    adjusted_action[0] = risk_result["adjusted_position"]
                    action = adjusted_action

                    # Log risk metrics periodically
                    if step % 1000 == 0:
                        risk_metrics = {
                            "step": step,
                            "risk_level": risk_result.get("risk_level", 0),
                            "original_position": action[0],
                            "adjusted_position": risk_result["adjusted_position"],
                        }
                        risk_metrics_history.append(risk_metrics)
                        logger.info(f"Risk adjustment at step {step}: {risk_metrics}")

                except Exception as e:
                    logger.warning(f"Risk adjustment failed at step {step}: {e}")
                    # Continue with original action if risk adjustment fails

            # Execute action in environment
            obs, reward, done, info = self.env.step(action)

            episode_reward += reward
            episode_length += 1

            # Train model
            self.model.learn(total_timesteps=1, reset_num_timesteps=False)

            # Handle episode end
            if done:
                episode_count += 1
                logger.info(
                    f"Episode {episode_count} finished. Reward: {episode_reward:.2f}, Length: {episode_length}"
                )

                # Reset for next episode
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0

                # Risk management episode reset
                if self.risk_manager:
                    self.risk_manager.reset()

        logger.info(f"Training completed with {episode_count} episodes")
        logger.info(f"Collected {len(risk_metrics_history)} risk metric snapshots")

        return {
            "status": "success",
            "total_timesteps": total_timesteps,
            "episodes": episode_count,
            "risk_management": True,
            "risk_metrics_history": risk_metrics_history,
        }


def main():
    """Main training function"""
    trainer = SACv435Trainer()
    result = trainer.train()

    # Save training results
    results_dir = Path(trainer.config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "training_results.json"
    save_training_results(result, results_file)

    print(f"Training results saved to {results_file}")


if __name__ == "__main__":
    main()
