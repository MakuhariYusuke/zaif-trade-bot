#!/usr/bin/env python3
"""
Unified Training Runner for Zaif Trade Bot.

Integrates multiple training approaches into a single interface:

ALGORITHMS:
-----------
1. PPO Training (algorithm: 'ppo')
   - Uses PPOTrainer from ppo_trainer.py
   - Standard PPO algorithm with Stable Baselines3
   - Supports evaluation, checkpointing, tensorboard logging
   - Best for: Standard reinforcement learning training

2. Base ML Reinforcement (algorithm: 'base_ml')
   - Uses MLReinforcementExperiment from base_ml_reinforcement.py
   - Base class for ML reinforcement experiments
   - Simple step-based training loop (currently dummy implementation)
   - Best for: Custom reinforcement learning experiments, prototyping

3. Iterative Training (algorithm: 'iterative')
   - Uses logic from run_1m.py
   - Multi-iteration training with resume capability
   - Supports streaming data, validation, Discord notifications
   - Best for: Long-running training sessions, production training

4. Ensemble Training (algorithm: 'ensemble')
   - Uses EnsembleTradingSystem from ensemble.py
   - Combines multiple trained PPO models for improved predictions
   - Supports weighted voting and risk management
   - Best for: Leveraging multiple models for robust trading decisions

5. Curriculum Learning (algorithm: 'curriculum')
   - Uses curriculum_learning.py for progressive learning
   - P0→P2 staged learning with forced balance → balanced transition → full curriculum
   - Addresses action distribution bias through progressive difficulty
   - Best for: Resolving persistent BUY/SELL bias in trading policies

USAGE:
------
python -m ztb.training.unified_trainer --config config.json --algorithm ppo

TRADING MODES:
---------------
- scalping: High-frequency scalping with 15s timeframe, smaller positions, higher transaction costs
- normal: Standard trading with 1m timeframe, full feature set, normal position sizes

Examples:
python -m ztb.training.unified_trainer --config unified_training_config.json  # scalping mode
python -m ztb.training.unified_trainer --config unified_training_config_normal.json  # normal mode
"""

# Set environment variables before any imports to avoid PyTorch issues
import logging
import os

from ztb.utils.errors import safe_operation
from ztb.utils.logging_utils import get_logger
from ztb.utils.data_utils import load_csv_data_optimized

logger = get_logger(__name__)

os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
# Disable CUDA to reduce memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# Additional memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type

import numpy as np

from ztb.utils.file_utils import safe_json_load
from ztb.utils.path_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

# Type definitions for better type safety
MemoryOptimizationConfig = Dict[str, Optional[int]]
EnvironmentConfig = Dict[str, Union[float, int]]
PPOCoreConfig = Dict[str, Union[float, int, bool, None]]
UnifiedConfig = Dict[str, Any]  # Keep flexible for unified structure
TrainingResult = Union[Any, None]  # Training methods can return various types

# Conditional imports based on algorithm
ppo_available = True
try:
    # Delay torch import by importing PPOTrainer only when needed
    pass
except ImportError:
    ppo_available = False
    logger.warning("PPO trainer not available (torch import failed)")
    PPOTrainer = None

from enum import Enum

# Import trading module to register environments
from ztb.training.entrypoints.base_ml_reinforcement import MLReinforcementExperiment

# Import Protocol and Enum types for type safety
from ztb.utils import DiscordNotifier


class UnifiedAlgorithm(Enum):
    """Supported training algorithms in UnifiedTrainer."""

    PPO = "ppo"
    BASE_ML = "base_ml"
    ITERATIVE = "iterative"
    ENSEMBLE = "ensemble"
    CURRICULUM = "curriculum"


@dataclass
class UnifiedTrainerConfig:
    """Configuration for UnifiedTrainer."""

    algorithm: UnifiedAlgorithm
    force: bool = False
    dry_run: bool = False
    enable_streaming: bool = False
    stream_batch_size: int = 256
    max_features: Optional[int] = None
    offline_mode: bool = False
    total_timesteps: Optional[int] = None  # Added to fix attribute access error


class UnifiedTrainer:
    """
    Unified training interface for different algorithms.

    WORK ASSIGNMENT:
    ---------------
    - PPO Algorithm: @trading-team - Standard RL training, evaluation, logging
    - Base ML Algorithm: @ml-research-team - Custom experiments, prototyping
    - Iterative Algorithm: @production-team - Long-running training, monitoring
    """

    def __init__(
        self,
        config: Dict[str, Any],
        force: bool = False,
        dry_run: bool = False,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        max_features: Optional[int] = None,
        total_timesteps: Optional[int] = None,
    ):
        """
        Initialize UnifiedTrainer.
        
        Args:
            config: Training configuration dictionary
            force: Force execution without prompts
            dry_run: Validate without executing
            enable_streaming: Enable streaming data pipeline
            stream_batch_size: Batch size for streaming
            max_features: Maximum number of features
            total_timesteps: Override total_timesteps from config (for quick validation runs)
        """
        # Override total_timesteps if specified
        if total_timesteps is not None:
            config = config.copy()  # Don't modify original
            config["total_timesteps"] = total_timesteps
            logger.info(f"Overriding total_timesteps: {total_timesteps:,}")
        
        # Create config object for better type safety
        algorithm_str = config.get("algorithm", "ppo")
        try:
            algorithm_enum = UnifiedAlgorithm(algorithm_str)
        except ValueError:
            raise ValueError(
                f"Unknown algorithm: {algorithm_str}. Supported: {[a.value for a in UnifiedAlgorithm]}"
            )

        self.config_obj = UnifiedTrainerConfig(
            algorithm=algorithm_enum,
            force=force,
            dry_run=dry_run,
            enable_streaming=enable_streaming,
            stream_batch_size=stream_batch_size,
            max_features=max_features,
            offline_mode=config.get("offline_mode", False),
            total_timesteps=total_timesteps,  # Added to pass the parameter
        )

        # Keep original config for backward compatibility
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.enable_streaming = enable_streaming
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.algorithm = config.get("algorithm", "ppo")
        self.logger = get_logger(__name__)

        # Cache for unified config to avoid repeated computation
        self._unified_config_cache: Optional[Dict[str, Any]] = None

        # Initialize Discord notifier (disabled in offline mode)
        if config.get("offline_mode", False):
            self.notifier = DiscordNotifier(webhook_url=None)  # Explicitly disable
        else:
            self.notifier = DiscordNotifier()
    
    # ==================================================================================
    # CONFIGURATION MANAGEMENT HELPERS (Bug #52 fix - unified configuration interface)
    # ==================================================================================
    
    def get_memory_optimization_config(self) -> MemoryOptimizationConfig:
        """
        Extract memory optimization parameters from config.
        
        These parameters control memory usage during training:
        - data_rows_limit: Maximum number of data rows to load
        - max_features: Maximum number of features to use (variance-based selection)
        
        Returns:
            Dict containing memory optimization settings
            
        Note:
            These settings were added as part of Bug #52 fix to prevent
            memory-related crashes during training.
        """
        return {
            "data_rows_limit": self.config.get("data_rows_limit"),
            "max_features": self.config.get("max_features"),
        }
    
    def get_environment_config(self) -> EnvironmentConfig:
        """
        Extract environment-specific parameters from config.
        
        Returns:
            Dict containing environment settings like max_position_size,
            initial_balance, transaction_cost, etc.
        """
        from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
        
        return {
            "max_position_size": self.config.get("max_position_size", DEFAULT_PPO_CONFIG.get("max_position_size", 1.0)),
            "initial_balance": self.config.get("initial_balance", 1000000),
            "transaction_cost": self.config.get("transaction_cost", DEFAULT_PPO_CONFIG.get("transaction_cost", 0.001)),
            "reward_scaling": self.config.get("reward_scaling", DEFAULT_PPO_CONFIG.get("reward_scaling", 1.0)),
        }
    
    def get_ppo_core_config(self) -> PPOCoreConfig:
        """
        Extract PPO algorithm-specific parameters from config.
        
        Returns:
            Dict containing PPO hyperparameters (learning_rate, n_steps, etc.)
        """
        from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
        
        return {
            "learning_rate": self.config.get("learning_rate", DEFAULT_PPO_CONFIG.get("learning_rate", 3e-4)),
            "n_steps": self.config.get("n_steps", DEFAULT_PPO_CONFIG.get("n_steps", 1024)),
            "batch_size": self.config.get("batch_size", DEFAULT_PPO_CONFIG.get("batch_size", 32)),
            "n_epochs": self.config.get("n_epochs", DEFAULT_PPO_CONFIG.get("n_epochs", 10)),
            "gamma": self.config.get("gamma", DEFAULT_PPO_CONFIG.get("gamma", 0.99)),
            "gae_lambda": self.config.get("gae_lambda", DEFAULT_PPO_CONFIG.get("gae_lambda", 0.95)),
            "clip_range": self.config.get("clip_range", DEFAULT_PPO_CONFIG.get("clip_range", 0.2)),
            "clip_range_vf": self.config.get("clip_range_vf", DEFAULT_PPO_CONFIG.get("clip_range_vf")),
            "normalize_advantage": self.config.get("normalize_advantage", DEFAULT_PPO_CONFIG.get("normalize_advantage", True)),
            "ent_coef": self.config.get("ent_coef", DEFAULT_PPO_CONFIG.get("ent_coef", 0.0)),
            "vf_coef": self.config.get("vf_coef", DEFAULT_PPO_CONFIG.get("vf_coef", 0.5)),
            "max_grad_norm": self.config.get("max_grad_norm", DEFAULT_PPO_CONFIG.get("max_grad_norm", 0.5)),
            "use_sde": self.config.get("use_sde", DEFAULT_PPO_CONFIG.get("use_sde", False)),
            "sde_sample_freq": self.config.get("sde_sample_freq", DEFAULT_PPO_CONFIG.get("sde_sample_freq", -1)),
            "target_kl": self.config.get("target_kl", DEFAULT_PPO_CONFIG.get("target_kl")),
            "verbose": self.config.get("verbose", DEFAULT_PPO_CONFIG.get("verbose", 1)),
        }
    
    def build_unified_config(self) -> Dict[str, Any]:
        """
        Build a unified configuration dict with all settings properly organized.
        
        This method caches the result to avoid repeated computation.
        
        Returns:
            Unified config dict with structure:
            {
                "ppo": {PPO core hyperparameters},
                "memory_optimization": {data_rows_limit, max_features},
                "environment": {env-specific settings},
                ... (all other top-level settings for backward compatibility)
            }
            
        Note:
            This structure was designed to fix Bug #52 configuration propagation issues.
            All downstream trainers should receive this unified config.
        """
        # DO NOT use cache if total_timesteps has been overridden
        # This ensures command-line overrides always take effect
        use_cache = (
            self._unified_config_cache is not None and 
            (not hasattr(self.config_obj, 'total_timesteps') or 
             self.config_obj.total_timesteps is None)
        )
        
        if use_cache:
            return self._unified_config_cache  # type: ignore[return-value]
            
        from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
        
        # Build structured config
        ppo_core = self.get_ppo_core_config()
        memory_opt = self.get_memory_optimization_config()
        environment = self.get_environment_config()
        
        # Extract total_timesteps from multiple possible locations
        # Priority: top-level (after CLI override) > training section > ppo section > default
        total_timesteps = (
            self.config.get("total_timesteps") or
            (self.config.get("training", {}) or {}).get("total_timesteps") or
            (self.config.get("ppo", {}) or {}).get("total_timesteps") or
            DEFAULT_PPO_CONFIG.get("total_timesteps", 100000)  # Use .get() for safety
        )
        
        # Build base unified structure first
        unified_base = {
            # Structured sections
            "ppo": {
                **ppo_core,
                **environment,  # PPOConfig expects these fields
                "total_timesteps": total_timesteps,
            },
            "memory_optimization": memory_opt,
            "environment": environment,
        }
        
        # Merge with original config (for backward compatibility)
        # BUT preserve our explicit overrides
        unified = {
            **unified_base,
            **self.config,  # Original config (may contain extra settings)
        }
        
        # CRITICAL: Explicit overrides AFTER merging to ensure they take precedence
        # These must come last to override anything from self.config
        unified["data_rows_limit"] = memory_opt["data_rows_limit"]
        unified["max_features"] = memory_opt["max_features"]
        unified["total_timesteps"] = total_timesteps
        unified["ppo"]["total_timesteps"] = total_timesteps
        
        # Only cache if no override (otherwise next call might use stale cache)
        if not hasattr(self.config_obj, 'total_timesteps') or self.config_obj.total_timesteps is None:
            self._unified_config_cache = unified
        
        return unified

    def train(self) -> TrainingResult:
        """Execute training based on algorithm."""
        return safe_operation(
            logger=logger,
            operation=self._train_impl,
            context="training_execution",
            default_result=None,
        )

    def _train_impl(self) -> TrainingResult:
        """Implementation of training execution."""
        if self.config_obj.algorithm == UnifiedAlgorithm.PPO:
            return self._train_ppo()
        elif self.config_obj.algorithm == UnifiedAlgorithm.BASE_ML:
            return self._train_base_ml()
        elif self.config_obj.algorithm == UnifiedAlgorithm.ITERATIVE:
            return self._train_iterative()
        elif self.config_obj.algorithm == UnifiedAlgorithm.ENSEMBLE:
            return self._train_ensemble()
        elif self.config_obj.algorithm == UnifiedAlgorithm.CURRICULUM:
            return self._train_curriculum()
        else:
            raise ValueError(f"Unknown algorithm: {self.config_obj.algorithm}")

    def _train_ppo(self) -> TrainingResult:
        """Train using PPO algorithm with optional SELL bias mitigation."""
        # Set environment variables before importing torch
        import os

        os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        trainer_class: Optional[Type[Any]] = None

        # Check if SELL bias mitigation is enabled
        enable_sell_mitigation = self.config.get("enable_sell_mitigation", False)

        if enable_sell_mitigation:
            self.logger.info("SELL bias mitigation enabled - using enhanced trainer")
            try:
                from ztb.training.experiments.sell_mitigation_ppo_trainer import SELLBiasMitigationPPOTrainer
                trainer_class = SELLBiasMitigationPPOTrainer
            except ImportError as e:
                self.logger.warning(f"SELL mitigation trainer not available: {e}. Falling back to standard PPO.")
                enable_sell_mitigation = False

        if not enable_sell_mitigation:
            try:
                from ztb.training.core.ppo_trainer import PPOTrainerAutoHalt as PPOTrainer
                trainer_class = PPOTrainer
            except ImportError as e:
                raise ImportError(
                    f"PPO training is not available due to import failure: {e}. Try using 'base_ml' algorithm instead."
                )

        # Ensure trainer_class is not None
        if trainer_class is None:
            raise ImportError("No suitable PPO trainer class could be loaded.")

        # ============================================================================
        # UNIFIED CONFIGURATION MANAGEMENT (Bug #52 fix)
        # ============================================================================
        # Apply total_timesteps override from command-line before building config
        if hasattr(self.config_obj, 'total_timesteps') and self.config_obj.total_timesteps is not None:
            self.logger.info(f"Overriding total_timesteps: {self.config_obj.total_timesteps:,}")
            self.config["total_timesteps"] = self.config_obj.total_timesteps
        
        # Build unified config using centralized methods
        # This ensures consistent configuration across all trainers
        unified_config = self.build_unified_config()

        # Get checkpoint interval from config (default: 25000 for 1M training = 40 checkpoints)
        checkpoint_interval = self.config.get("checkpoint_interval", 25000)

        # Create trainer with SELL mitigation if enabled
        if enable_sell_mitigation:
            # Import SELLMitigationParams
            from ztb.training.config.trainer_params import SELLMitigationParams
            
            # Build Lagrange parameters dict from config
            lagrange_params = {}
            if self.config.get("enable_lagrange", True):
                lagrange_params = {
                    "r_target": self.config.get("lagrange_r_target", 0.15),
                    "tolerance": self.config.get("lagrange_tolerance", 0.05),
                    "eta": self.config.get("lagrange_eta", 0.01),
                    "lambda_max": self.config.get("lagrange_lambda_max", 1.0),
                    "warmup_steps": self.config.get("lagrange_warmup_steps", 0),
                }
                self.logger.info(f"Lagrange parameters: {lagrange_params}")
            
            # Create mitigation params with unified config
            # Note: unified_config contains all required fields for PPOConfig
            mitigation_params = SELLMitigationParams(
                data_path=self.config.get("data_path"),  # type: ignore[arg-type]
                config=unified_config,  # type: ignore[arg-type]
                checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
                checkpoint_interval=checkpoint_interval,
                enable_lagrange=self.config.get("enable_lagrange", True),
                enable_probes=self.config.get("enable_probes", False),
                enable_weights=self.config.get("enable_weights", False),
                enable_pan=self.config.get("enable_pan", True),
                enable_target_entropy=self.config.get("enable_target_entropy", False),
                enable_stratified_sampling=self.config.get("enable_stratified_sampling", False),
                allow_reverse=self.config.get("allow_reverse", False),
                probe_csv_path=self.config.get("probe_csv_path"),
                lagrange_params=lagrange_params if lagrange_params else None,
            )
            
            trainer = trainer_class(params=mitigation_params)
        else:
            # Import TrainerParams for standard PPO
            from ztb.training.config.trainer_params import TrainerParams
            
            # Use unified config directly (no need for additional wrapping)
            # The unified config already has the proper structure:
            # {"ppo": {...}, "memory_optimization": {...}, ...all top-level settings}
            trainer_params = TrainerParams(
                data_path=self.config.get("data_path"),  # type: ignore[arg-type]
                config=unified_config,  # type: ignore[arg-type]
                checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
                checkpoint_interval=checkpoint_interval,
            )
            
            trainer = trainer_class(params=trainer_params)

        model = trainer.train(session_id=self.config.get("session_id", "ppo_session"))

        # Save final model to models directory
        if model is not None:
            import os
            import gc
            from pathlib import Path

            import pandas as pd

            model_dir = Path(self.config.get("model_dir", "models"))
            model_dir.mkdir(exist_ok=True)
            model_path = (
                model_dir / f"{self.config.get('session_id', 'ppo_session')}.zip"
            )
            
            # Clear memory before saving large model
            self.logger.info("Preparing to save model...")
            gc.collect()
            
            try:
                self.logger.info(f"Saving model to {model_path}...")
                model.save(str(model_path))
                self.logger.info(f"✅ Final model saved to {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save model: {e}")
                raise
            finally:
                # Clear memory after save
                gc.collect()

            # Save feature schema for evaluation consistency
            try:
                from ztb.utils.feature_schema import create_and_save_schema

                data_path = self.config.get("data_path")
                if data_path:
                    df = load_csv_data_optimized(data_path)
                    # Auto-detect feature columns (exclude meta columns)
                    exclude_cols = {
                        "ts",
                        "timestamp",
                        "exchange",
                        "pair",
                        "episode_id",
                        "side",
                        "source",
                    }
                    feature_columns = [
                        col
                        for col in df.columns
                        if col not in exclude_cols
                        and pd.api.types.is_numeric_dtype(df[col])
                    ]
                    
                    schema = create_and_save_schema(df, model_dir, feature_columns)
                    self.logger.info(
                        f"Feature schema saved ({len(feature_columns)} features, "
                        f"hash: {schema.compute_hash()[:16]}...)"
                    )

                    # Save normalization statistics
                    try:
                        from ztb.utils.normalization import NormalizationStats, save_scaler

                        # Compute normalization stats from training data
                        feature_data = df[feature_columns].values
                        mean = np.mean(feature_data, axis=0)
                        std = np.std(feature_data, axis=0)
                        n_samples = len(df)

                        norm_stats = NormalizationStats(
                            feature_names=feature_columns,
                            mean=mean,
                            std=std,
                            n_samples=n_samples,
                            metadata={"data_path": str(data_path)},
                        )
                        save_scaler(model_dir, norm_stats)
                        self.logger.info(
                            f"Normalization stats saved ({n_samples} samples, "
                            f"hash: {norm_stats.compute_hash()[:16]}...)"
                        )
                    except Exception as norm_error:
                        self.logger.warning(f"Failed to save normalization stats: {norm_error}")

            except Exception as e:
                self.logger.warning(f"Failed to save feature schema: {e}")

        return model

    def _train_base_ml(self) -> TrainingResult:
        """Train using base ML reinforcement."""
        experiment = MLReinforcementExperiment(
            self.config, total_steps=self.config.get("total_steps", 1000)
        )
        return experiment.run()

    def _train_iterative(self) -> TrainingResult:
        """Train using iterative approach (from run_1m.py)."""
        # Apply trading mode presets
        trading_mode = self.config.get("trading_mode", "normal")
        if trading_mode == "scalping":
            # Scalping mode presets
            self.config.setdefault("feature_set", "scalping")
            self.config.setdefault("timeframe", "15s")
            self.config.setdefault("reward_scaling", 0.5)
            self.config.setdefault("transaction_cost", 0.002)
            self.config.setdefault("max_position_size", 0.3)
            self.config.setdefault(
                "total_timesteps", 1000000
            )  # Longer training for scalping
            # Update session IDs for scalping
            if "scalping" not in self.config.get("session_id", ""):
                self.config["session_id"] = (
                    f"scalping_{self.config.get('session_id', 'session')}"
                )
                self.config["correlation_id"] = (
                    f"scalping_{self.config.get('correlation_id', 'correlation')}"
                )
        else:
            # Normal trading mode presets
            self.config.setdefault("feature_set", "full")
            self.config.setdefault("timeframe", "1m")
            self.config.setdefault("reward_scaling", 1.0)
            self.config.setdefault("transaction_cost", 0.001)
            self.config.setdefault("max_position_size", 1.0)
            self.config.setdefault("total_timesteps", 100000)

        # Long-running operation confirmation
        total_timesteps = self.config.get("total_timesteps", 100000)
        if total_timesteps >= 100_000 and not self.force:
            from ztb.utils.long_running_confirm import confirm_long_running_operation

            if not confirm_long_running_operation(
                operation_name=f"PPO Training ({self.config.get('session_id', 'iterative_session')})",
                estimated_time=f"~{total_timesteps // 1000}k steps, several hours",
                risk_description="High CPU/memory usage, large log files, potential system slowdown",
                message="This will train a PPO model for a long time. Continue?",
            ):
                logger.info("Training cancelled by user")
                return None

        # Dry run mode
        logger.debug(f"config feature_set = {self.config.get('feature_set', 'full')}")
        if self.dry_run:
            logger.info(
                f"Dry run: would train with session_id {self.config.get('session_id', 'iterative_session')}"
            )
            logger.info(
                f"Data path: {self.config.get('data_path', 'ml-dataset-enhanced.csv')}"
            )
            logger.info(f"Total timesteps: {total_timesteps}")
            logger.info("Setup validation complete")
            return None

        # Import and use run_1m logic
        from ztb.training.scripts.run_1m import main as run_1m_main

        # Get checkpoint interval from config (default: 10000 for iterative training)
        checkpoint_interval = self.config.get("checkpoint_interval", 10000)

        # Set up arguments for run_1m
        sys.argv = [
            "run_1m.py",
            "--data-path",
            self.config.get("data_path", "ml-dataset-enhanced.csv"),
            "--correlation-id",
            self.config.get("session_id", "iterative_session"),
            "--total-timesteps",
            str(total_timesteps),
            "--iterations",
            str(self.config.get("iterations", 10)),
            "--steps-per-iteration",
            str(self.config.get("steps_per_iteration", 100000)),
            "--feature-set",
            self.config.get("feature_set", "full"),
            "--timeframe",
            self.config.get("timeframe", "1m"),
            "--checkpoint-dir",
            self.config.get("checkpoint_dir", "checkpoints"),
            "--checkpoint-interval",
            str(checkpoint_interval),
            "--log-dir",
            self.config.get("log_dir", "logs"),
            "--model-dir",
            self.config.get("model_dir", "models"),
            "--reward-trade-frequency-penalty",
            str(self.config.get("reward_trade_frequency_penalty", 0.3)),
            "--reward-trade-frequency-halflife",
            str(self.config.get("reward_trade_frequency_halflife", 12.0)),
            "--reward-trade-cooldown-steps",
            str(self.config.get("reward_trade_cooldown_steps", 3)),
            "--reward-trade-cooldown-penalty",
            str(self.config.get("reward_trade_cooldown_penalty", 0.5)),
            "--reward-max-consecutive-trades",
            str(self.config.get("reward_max_consecutive_trades", 3)),
            "--reward-consecutive-trade-penalty",
            str(self.config.get("reward_consecutive_trade_penalty", 0.4)),
            "--transaction-cost",
            str(self.config.get("transaction_cost", 0.001)),
            "--max-position-size",
            str(self.config.get("max_position_size", 1.0)),
        ]

        # DEBUG: Print sys.argv
        logger.debug(f"sys.argv = {sys.argv}")
        logger.debug(f"feature-set value = {self.config.get('feature_set', 'full')}")

        # Add optional arguments
        if self.dry_run:
            sys.argv.append("--dry-run")
        if self.force:
            sys.argv.append("--force")
        if self.enable_streaming:  # type: ignore
            sys.argv.extend(
                [
                    "--enable-streaming",
                    "--stream-batch-size",
                    str(self.stream_batch_size),
                ]
            )
        if self.max_features is not None:
            sys.argv.extend(["--max-features", str(self.max_features)])
        if self.config.get("offline_mode", False):
            sys.argv.append("--offline-mode")

        # DEBUG: Print final config and sys.argv before calling run_1m_main
        logger.debug(f"Final config feature_set = {self.config.get('feature_set')}")
        logger.debug(f"Final sys.argv = {sys.argv}")

        return run_1m_main()

    def _train_ensemble(self) -> TrainingResult:
        """Train using ensemble approach (load and combine existing models)."""
        from ztb.training.models.ensemble import EnsembleTradingSystem

        # Get model configurations from config
        model_configs = self.config.get("ensemble_models", [])
        if not model_configs:
            raise ValueError(
                "No ensemble_models specified in config for ensemble training"
            )

        # Create ensemble system
        ensemble_system = EnsembleTradingSystem(model_configs)

        self.logger.info(
            f"Ensemble system initialized with {len(ensemble_system.ensemble.models)} models"
        )

        # For ensemble, we don't train but validate the setup
        if self.dry_run:
            self.logger.info("Dry run: ensemble system setup validated")
            return ensemble_system

        # Save ensemble configuration for later use
        import json

        ensemble_config_path = (
            Path(self.config.get("model_dir", "models")) / "ensemble_config.json"
        )
        with open(ensemble_config_path, "w") as f:
            json.dump(
                {
                    "model_configs": model_configs,
                    "created_at": str(datetime.now()),
                    "session_id": self.config.get("session_id", "ensemble_session"),
                },
                f,
                indent=2,
            )

        self.logger.info(f"Ensemble configuration saved to {ensemble_config_path}")

        return ensemble_system

    def _train_curriculum(self) -> Optional[bool]:
        """Train using curriculum learning approach (P0→P2 staged learning)."""
        from ztb.training.experiments.curriculum_learning import main as curriculum_main

        # Set up environment for curriculum learning
        self.logger.info("Starting curriculum learning (P0→P2 staged approach)")

        # Validate data path
        data_path = self.config.get("data_path", "ml-dataset-enhanced.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Curriculum learning uses its own main function
        # We need to temporarily modify the working directory or config
        original_cwd = os.getcwd()

        try:
            # Change to project root for curriculum learning
            project_root = get_project_root()
            os.chdir(project_root)

            # Run curriculum learning
            curriculum_main()

            # Return success indicator
            return True

        except Exception as e:
            self.logger.error(f"Curriculum learning failed: {e}")
            return False

        finally:
            # Restore original working directory
            os.chdir(original_cwd)


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file."""
    config = safe_json_load(Path(config_path))
    if config is None:
        raise FileNotFoundError(f"Could not load config from {config_path}")
    return config


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Training Runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "base_ml", "iterative", "ensemble", "curriculum"],
        help="Override algorithm from config file",
    )
    parser.add_argument(
        "--data-path", type=str, help="Override data path from config file"
    )
    parser.add_argument(
        "--total-timesteps", type=int, help="Override total timesteps from config file"
    )
    parser.add_argument(
        "--session-id", type=str, help="Override session ID from config file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO). Overrides --verbose flag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without long-running operation confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - validate setup without training",
    )
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        help="Enable streaming pipeline (default: disabled)",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=256,
        help="Streaming batch size (default: 256)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to use (default: all features)",
    )

    args = parser.parse_args()

    # Setup logging
    # --log-level takes precedence over --verbose
    if hasattr(args, 'log_level') and args.log_level:
        log_level = getattr(logging, args.log_level)
    else:
        log_level = logging.DEBUG if args.verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Also set the root logger level to suppress third-party DEBUG logs
    logging.getLogger().setLevel(log_level)

    logger = get_logger(__name__)

    try:
        # Load configuration
        config = load_config(args.config)
        if config is None:
            raise FileNotFoundError(f"Could not load config from {args.config}")
        logger.info(f"Loaded config from {args.config}")

        # Override config with command line arguments
        if args.algorithm:
            config["algorithm"] = args.algorithm
        if args.data_path:
            config["data_path"] = args.data_path
        if args.total_timesteps:
            config["total_timesteps"] = args.total_timesteps
        if args.session_id:
            config["session_id"] = args.session_id

        logger.info(f"Using algorithm: {config.get('algorithm', 'ppo')}")
        logger.info(f"Session ID: {config.get('session_id', 'default')}")

        # Create and run trainer
        trainer = UnifiedTrainer(
            config,
            args.force,
            args.dry_run,
            args.enable_streaming,
            args.stream_batch_size,
            args.max_features,
        )
        result = trainer.train()

        if result is None:
            logger.warning("Training returned None - may have been cancelled or failed")
        else:
            logger.info("Training completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
