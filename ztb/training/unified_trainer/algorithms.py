#!/usr/bin/env python3
"""
Algorithm-specific training implementations for Unified Trainer.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, cast
from ztb.types.common import ConfigDict, TrainingStats
from ztb.optimization.system_optimizer import SystemOptimizer

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.components.memory_manager import MemoryManager
# NOTE: removed an incomplete import that caused a syntax error in this file.
# If specific constants are required, re-import them explicitly by name.
from ztb.multimodal.pretraining import SelfSupervisedTrainer as SSPTrainer
from ztb.multimodal.pretraining.config import get_config as get_ssp_config
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD, SAC_CONTINUOUS_THRESHOLD_NEG
from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import ensure_dict

# Import distributed training utilities
from ztb.training.distributed.distributed_training import (
    get_distributed_info,
)


class TrainingProgressCallback(BaseCallback):
    """Enhanced callback for monitoring training progress and action distribution."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1, system_optimizer: Optional[SystemOptimizer] = None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.system_optimizer = system_optimizer
        # Provide explicit type annotations so mypy can reason about these lists
        self.continuous_actions: List[float] = []
        self.discrete_actions: List[int] = []
        self.reward_history: List[float] = []
        self.episode_rewards: List[float] = []
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def _on_step(self) -> bool:
        # Record continuous action taken
        try:
            actions = self.locals.get('actions')
            if actions is not None:
                continuous_action = actions[0]
                if isinstance(continuous_action, np.ndarray):
                    continuous_action = continuous_action.item()
                self.continuous_actions.append(continuous_action)

                # Convert to discrete action for tracking
                discrete_action = self._continuous_to_discrete_action(continuous_action)
                self.discrete_actions.append(discrete_action)
                # Use logger for debug output instead of print
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.debug(f"Recorded action {continuous_action:.6f} -> {discrete_action}")
                else:
                    logging.debug(f"Recorded action {continuous_action:.6f} -> {discrete_action}")
            else:
                # Use logger if available
                if self.logger is not None:
                    self.logger.debug("Actions not available - actions: %s", actions)
                else:
                    logging.debug("Actions not available - actions: %s", actions)
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.warning(f"Failed to record action: {e}")
            else:
                logging.warning(f"Failed to record action: {e}")

        # Record reward
        try:
            # locals is a dict-like mapping provided by SB3; guard access
            rewards = self.locals.get('rewards') if isinstance(self.locals, dict) else None
            if rewards:
                reward = rewards[0]
                self.reward_history.append(reward)
        except Exception as e:
            if self.logger is not None:
                self.logger.warning("Failed to record reward: %s", e)
            else:
                logging.warning("Failed to record reward: %s", e)

        # Log progress
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        return True

    def _continuous_to_discrete_action(
        self,
        continuous_action: float,
        buy_threshold: float = SAC_CONTINUOUS_THRESHOLD,
        sell_threshold: float = SAC_CONTINUOUS_THRESHOLD_NEG,
    ) -> int:
        """Convert continuous action (-1 to 1) to discrete action (0=HOLD, 1=BUY, 2=SELL)."""
        # Use the centralized continuous_to_discrete_action function for consistency
        # continuous_to_discrete_action may be untyped to mypy; cast to int explicitly
        # Use centralized util; ensure input is a float
        try:
            ca = float(continuous_action)
        except Exception:
            ca = 0.0

        result = continuous_to_discrete_action(ca)
        try:
            return int(result)
        except Exception:
            # Fallback to HOLD when unexpected
            return 0

    def _log_progress(self) -> None:
        """Log training progress and action distribution."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0

        # Always show progress, even if no actions recorded yet
        if self.discrete_actions:
            total_actions = len(self.discrete_actions)
            # Shift discrete actions to non-negative range for bincount: SELL(-1)->0, HOLD(0)->1, BUY(1)->2
            shifted_actions = np.array(self.discrete_actions) + 1
            discrete_counts = np.bincount(shifted_actions, minlength=3)

            action_dist = {
                'SELL': discrete_counts[0] / total_actions,
                'HOLD': discrete_counts[1] / total_actions,
                'BUY': discrete_counts[2] / total_actions
            }

            # Prefer the callback's logger when available
            # Prefer instance logger
            if self.logger is not None:
                self.logger.info(
                    "Step %6d | Elapsed: %6.1fs | SPS: %5.1f | HOLD: %.1%% | BUY: %.1%% | SELL: %.1%% | Rewards: %d recorded",
                    self.n_calls, elapsed, steps_per_sec, action_dist['HOLD'] * 100.0, action_dist['BUY'] * 100.0, action_dist['SELL'] * 100.0, len(self.reward_history)
                )
            else:
                logging.info(
                    "Step %6d | Elapsed: %6.1fs | SPS: %5.1f | HOLD: %.1%% | BUY: %.1%% | SELL: %.1%% | Rewards: %d recorded",
                    self.n_calls, elapsed, steps_per_sec, action_dist['HOLD'] * 100.0, action_dist['BUY'] * 100.0, action_dist['SELL'] * 100.0, len(self.reward_history)
                )
        else:
            # Show progress even when no actions recorded yet
            if self.logger is not None:
                self.logger.info(
                    "Step %6d | Elapsed: %6.1fs | SPS: %5.1f | No actions recorded yet | Rewards: %d recorded",
                    self.n_calls, elapsed, steps_per_sec, len(self.reward_history)
                )
            else:
                logging.info(
                    "Step %6d | Elapsed: %6.1fs | SPS: %5.1f | No actions recorded yet | Rewards: %d recorded",
                    self.n_calls, elapsed, steps_per_sec, len(self.reward_history)
                )

        # Apply system optimizations during training
        if self.system_optimizer:
            try:
                with self.system_optimizer.optimize_training_step(f"training_step_{self.n_calls}"):
                    pass  # System optimization is applied in the context manager
            except Exception as e:
                if self.logger is not None:
                    self.logger.warning("System optimization failed: %s", e)
                else:
                    logging.warning("System optimization failed: %s", e)


class BaseAlgorithmTrainer(ABC):
    """Base class for algorithm-specific trainers."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration for this algorithm."""
        pass

    @abstractmethod
    def train(self) -> bool:
        """Execute training for this algorithm."""
        pass

    @abstractmethod
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        pass


class SACTrainer(BaseAlgorithmTrainer):
    """SAC algorithm trainer with enhanced UI and monitoring."""

    def __init__(self, config: ConfigDict, logger: Optional[logging.Logger] = None, gradient_accumulation_steps: int = 1, system_optimizer: Optional[SystemOptimizer] = None):
        super().__init__(config, logger)
        # model will be instantiated later; annotate as optional to satisfy mypy
        self.model: Optional[SAC] = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.system_optimizer = system_optimizer
        self.training_stats: TrainingStats = {}

    def validate_config(self) -> bool:
        """Validate SAC configuration."""
        try:
            # Check required SAC hyperparameters
            sac_config = self.config.get('sac_hyperparameters', {})
            required_keys = ['learning_rate', 'buffer_size', 'learning_starts', 'batch_size']
            for key in required_keys:
                if key not in sac_config:
                    self.logger.error(f"Missing SAC hyperparameter: {key}")
                    return False

            # Check environment config
            env_config = self.config.get('environment', {})
            required_env_keys = ['initial_balance', 'transaction_cost', 'max_position_size']
            for key in required_env_keys:
                if key not in env_config:
                    self.logger.error(f"Missing environment config: {key}")
                    return False

            # Check data file
            data_path = self.config.get('data_path', 'btc_jpy_real_dataset.csv')
            if not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return False

            self.logger.info("SAC configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"SAC configuration validation failed: {e}")
            return False

    def train(self) -> bool:
        """Execute SAC training with enhanced monitoring."""
        try:
            self.logger.info("🚀 Starting SAC training...")

            # Extract configurations
            sac_config = self.config.get('sac_hyperparameters', {})
            env_config_dict = self.config.get('environment', {})
            reward_settings = self.config.get('reward_settings', {})

            # Load data
            data_path = self.config.get('data_path', 'btc_jpy_real_dataset.csv')
            self.logger.info(f"📊 Loading data from {data_path}")
            df = pd.read_csv(data_path)
            self.logger.info(f"✅ Loaded {len(df)} data points")

            # Optimize data types for memory efficiency
            self.logger.info("🔧 Optimizing data types for memory efficiency...")
            for col in df.select_dtypes(include=[np.float64]).columns:
                df[col] = df[col].astype(np.float32)
            self.logger.info("✅ Data type optimization completed")

            # Create environment config
            env_config = EnvironmentConfig()
            env_config.initial_portfolio_value = env_config_dict.get('initial_balance', 200000)
            env_config.transaction_cost = env_config_dict.get('transaction_cost', 1e-05)
            env_config.max_position_size = env_config_dict.get('max_position_size', 1.0)
            env_config.use_standardized_observations = env_config_dict.get('use_standardized_observations', True)
            env_config.curriculum_stage = env_config_dict.get('curriculum_stage', 'profit_optimized')
            env_config.use_continuous_actions = True  # SAC requires continuous actions

            # Set reward settings
            env_config.reward_scaling = reward_settings.get('reward_scale', 500.0)
            env_config.reward_clip_value = reward_settings.get('reward_clip_max', 200.0)

            # Create environment
            self.logger.info("🏗️  Creating trading environment...")
            env = HeavyTradingEnv(df=df, config=env_config)
            env = Monitor(env)

            # Create SAC model
            self.logger.info("🤖 Creating SAC model...")
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=sac_config.get('learning_rate', 0.0003),
                buffer_size=sac_config.get('buffer_size', 20000),
                learning_starts=sac_config.get('learning_starts', 500),
                batch_size=sac_config.get('batch_size', 128),
                tau=sac_config.get('tau', 0.005),
                gamma=sac_config.get('gamma', 0.99),
                train_freq=sac_config.get('train_freq', 1),
                gradient_steps=self.gradient_accumulation_steps,  # Use gradient accumulation
                ent_coef=sac_config.get('ent_coef', 0.01),
                target_update_interval=sac_config.get('target_update_interval', 1),
                target_entropy=sac_config.get('target_entropy', -1.0),
                verbose=0  # We'll handle logging ourselves
            )

            # Training parameters
            total_timesteps = self.config.get('total_timesteps', 50000)
            self.logger.info(f"🎯 Training for {total_timesteps:,} timesteps")

            # Check for distributed training
            dist_info = get_distributed_info()
            if dist_info['is_distributed']:
                self.logger.info(f"🔄 Distributed training enabled: rank {dist_info['rank']}/{dist_info['world_size']}")
                # Adjust total_timesteps for distributed training (each process handles subset)
                total_timesteps = total_timesteps // dist_info['world_size']
                self.logger.info(f"Adjusted total_timesteps per process: {total_timesteps:,}")

            # Create progress callback
            callback = TrainingProgressCallback(check_freq=1000, system_optimizer=self.system_optimizer)

            # Start training
            start_time = time.time()
            self.logger.info("🏃 Training started...")

            # Use memory efficient processing during training
            import gc
            gc.disable()  # Disable automatic GC during training for performance

            # Narrow self.model locally to help static analyzers and avoid repeated Optional access
            model = self.model
            if model is None:
                self.logger.error("Model not initialized before training")
                return False

            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )

            gc.enable()  # Re-enable GC after training
            gc.collect()  # Force garbage collection

            # Training completed
            training_time = time.time() - start_time
            self.logger.info(f"✅ Training completed in {training_time:.1f} seconds")

            # Save model
            model_name = self.config.get('model_name', 'sac_model')
            model_path = f"models/{model_name}.zip"
            os.makedirs("models", exist_ok=True)

            self.logger.info(f"💾 Saving model to {model_path}")
            # Use local model reference and guard presence
            model = self.model
            if model is not None:
                model.save(model_path)
            else:
                self.logger.warning("Model missing at save time; skipping save")

            # Collect training statistics
            self.training_stats = {
                'total_timesteps': total_timesteps,
                'training_time': training_time,
                'steps_per_second': total_timesteps / training_time,
                'model_path': model_path,
                'final_reward': callback.reward_history[-1] if callback.reward_history else 0,
                'action_distribution': self._calculate_final_action_distribution(callback)
            }

            self.logger.info(f"📈 Training stats: {self.training_stats}")
            return True

        except Exception as e:
            self.logger.error(f"❌ SAC training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _calculate_final_action_distribution(self, callback: TrainingProgressCallback) -> Dict[str, float]:
        """Calculate final action distribution from callback data."""
        if not callback.discrete_actions:
            return {'HOLD': 0.0, 'BUY': 0.0, 'SELL': 0.0}

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
            'HOLD': discrete_counts[0] / total_actions,
            'BUY': discrete_counts[1] / total_actions,
            'SELL': discrete_counts[2] / total_actions
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        # Cast to expected return type to satisfy static checkers
        return cast(Dict[str, Any], self.training_stats.copy())


class PPOTrainer(BaseAlgorithmTrainer):
    """PPO algorithm trainer (placeholder for future implementation)."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, gradient_accumulation_steps: int = 1):
        super().__init__(config, logger)
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def validate_config(self) -> bool:
        """Validate PPO configuration."""
        self.logger.info("PPO configuration validation not yet implemented")
        return True

    def train(self) -> bool:
        """Execute PPO training."""
        self.logger.info("PPO training not yet implemented")
        return True

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {}


class SelfSupervisedTrainer(BaseAlgorithmTrainer):
    """Self-supervised pre-training trainer for financial data."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize self-supervised trainer."""
        super().__init__(config, logger)
        # Explicitly type these attributes to help static analysis
        self.ssp_trainer: Optional[SSPTrainer] = None
        self.train_data: Optional[torch.Tensor] = None
        self.val_data: Optional[torch.Tensor] = None

        # Initialize memory manager for self-supervised training
        memory_log_path = self.config.get('memory_log_path', 'logs/memory_self_supervised.csv')
        self.memory_manager = MemoryManager(
            memory_log_path=memory_log_path,
            memory_logging_enabled=self.config.get('memory_logging_enabled', True),
            memory_log_interval_steps=self.config.get('memory_log_interval', 1000)
        )

    def validate_config(self) -> bool:
        """Validate self-supervised pre-training configuration."""
        required_keys = ['input_dim', 'device']
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required configuration key: {key}")
                return False

        # Validate data paths if provided
        if 'train_data_path' in self.config:
            if not os.path.exists(self.config['train_data_path']):
                self.logger.error(f"Training data file not found: {self.config['train_data_path']}")
                return False

        if 'val_data_path' in self.config:
            if not os.path.exists(self.config['val_data_path']):
                self.logger.error(f"Validation data file not found: {self.config['val_data_path']}")
                return False

        self.logger.info("Self-supervised pre-training configuration validated successfully")
        return True

    def _load_data(self) -> bool:
        """Load training and validation data."""
        try:
            input_dim = self.config['input_dim']

            # Load training data
            if 'train_data_path' in self.config:
                self.logger.info(f"Loading training data from {self.config['train_data_path']}")
                train_df = pd.read_csv(self.config['train_data_path'])
                # Convert to tensor format (assuming CSV has time series data)
                # Use a local variable to avoid calling methods on an attribute typed Optional
                train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
                train_tensor = train_tensor.unsqueeze(0)  # Add batch dimension if needed
                self.train_data = train_tensor
            else:
                self.logger.warning("No training data path provided, using synthetic data for demonstration")
                # Generate synthetic data for demonstration
                batch_size = self.config.get('synthetic_batch_size', 100)
                seq_len = self.config.get('seq_len', 100)
                self.train_data = torch.randn(batch_size, seq_len, input_dim)

            # Load validation data
            if 'val_data_path' in self.config:
                self.logger.info(f"Loading validation data from {self.config['val_data_path']}")
                val_df = pd.read_csv(self.config['val_data_path'])
                val_tensor = torch.tensor(val_df.values, dtype=torch.float32)
                val_tensor = val_tensor.unsqueeze(0)
                self.val_data = val_tensor
            else:
                # Generate synthetic validation data
                val_batch_size = self.config.get('synthetic_val_batch_size', 20)
                seq_len = self.config.get('seq_len', 100)
                self.val_data = torch.randn(val_batch_size, seq_len, input_dim)

            train_shape = self.train_data.shape if self.train_data is not None else None
            val_shape = self.val_data.shape if self.val_data is not None else None
            self.logger.info(f"Data loaded - Train: {train_shape}, Val: {val_shape}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False

    def train(self) -> bool:
        """Execute self-supervised pre-training."""
        try:
            self.logger.info("Starting self-supervised pre-training")

            # Log initial memory usage
            self.memory_manager.log_memory_usage("SSP_training_start")

            # Load data
            if not self._load_data():
                return False

            # Initialize self-supervised trainer with memory manager
            ssp = SSPTrainer(
                input_dim=self.config['input_dim'],
                device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints/pretraining'),
                memory_manager=self.memory_manager,
            )
            # Use local variable for calls, then assign to self.ssp_trainer
            ssp_trainer_local: SSPTrainer = ssp

            # Get training configuration
            ssp_config_type = self.config.get('config_type', 'default')
            ssp_config = get_ssp_config(ssp_config_type)

            # Override with custom config if provided
            if 'custom_config' in self.config:
                from ztb.multimodal.pretraining.config import update_config
                ssp_config = update_config(ssp_config, self.config['custom_config'])

            # Train all stages
            self.logger.info(f"Training with configuration: {ssp_config_type}")
            try:
                ssp_trainer_local.train_all_stages(self.train_data, self.val_data, ssp_config)
            except Exception as e:
                self.logger.error(f"SSP training failed during train_all_stages: {e}")
                raise

            # Save final checkpoint
            checkpoint_name = self.config.get('checkpoint_name', 'final')
            ssp_trainer_local.save_checkpoint(checkpoint_name)

            # Save training history
            history_path = self.config.get('history_path', 'training_history.json')
            ssp_trainer_local.save_training_history(history_path)

            # Assign local trainer to instance once all calls succeeded
            self.ssp_trainer = ssp_trainer_local

            # Log final memory usage
            self.memory_manager.log_memory_usage("SSP_training_complete")

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("Self-supervised pre-training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Self-supervised pre-training failed: {e}")
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if self.ssp_trainer is None:
            return {}

        stats: Dict[str, Any] = {
            'training_history': getattr(self.ssp_trainer, 'training_history', {}),
            'encoders_available': list(getattr(self.ssp_trainer, 'get_pretrained_encoders', lambda: {})().keys()),
            'data_shapes': {
                'train': self.train_data.shape if self.train_data is not None else None,
                'val': self.val_data.shape if self.val_data is not None else None
            }
        }
        return stats


def create_algorithm_trainer(
    algorithm: str, 
    config: Dict[str, Any], 
    logger: Optional[logging.Logger] = None,
    gradient_accumulation_steps: int = 1,
    system_optimizer: Optional[Any] = None
) -> BaseAlgorithmTrainer:
    """Factory function to create algorithm-specific trainer."""
    algorithm = algorithm.lower()

    if algorithm == "sac":
        return SACTrainer(config, logger, gradient_accumulation_steps, system_optimizer)
    elif algorithm == "ppo":
        return PPOTrainer(config, logger, gradient_accumulation_steps)
    elif algorithm == "self_supervised":
        return SelfSupervisedTrainer(config, logger)
    elif algorithm == "multimodal":
        # Create multimodal config from unified config
        # Temporarily disabled due to circular import issues
        # multimodal_config = MultimodalConfig(
        #     price_feature_dim=config.get('price_feature_dim', 156),
        #     text_embedding_dim=config.get('text_embedding_dim', 768),
        #     economic_feature_dim=config.get('economic_feature_dim', 10),
        #     action_dim=config.get('action_dim', 3),
        #     hidden_dim=config.get('multimodal_hidden_dim', 256),
        #     num_heads=config.get('multimodal_num_heads', 8)
        # )
        # return MultimodalSACTrainer(multimodal_config, config, config.get('env_config', {}))
        raise NotImplementedError("Multimodal training temporarily disabled due to circular import issues")
    elif algorithm == "online_learning":
        # Create online learning config from unified config
        # Temporarily disabled due to circular import issues
        # online_config = OnlineLearningConfig(
        #     learning_mode=config.get('online_learning_mode', 'incremental'),
        #     batch_size=config.get('online_batch_size', 32),
        #     max_memory_samples=config.get('online_memory_samples', 10000),
        #     adaptation_trigger_threshold=config.get('online_adaptation_threshold', 0.1)
        # )
        # return OnlineLearningSACTrainer(online_config, config, config.get('env_config', {}))
        raise NotImplementedError("Online learning training temporarily disabled due to circular import issues")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")