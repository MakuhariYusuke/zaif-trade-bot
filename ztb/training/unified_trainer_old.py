#!/usr/bin/env python3
"""
Unified Training Runner for Zaif Trade Bot.

DEPRECATED: This file has been modularized. Use ztb.training.unified_trainer instead.
"""

import warnings

warnings.warn(
    "ztb.training.unified_trainer is deprecated. Use the modularized ztb.training.unified_trainer/ package instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new modular structure for backward compatibility
from ztb.training.unified_trainer import (
    UnifiedAlgorithm,
    UnifiedTrainer,
    UnifiedTrainerConfig,
    configure_progress_bar,
    load_config,
)

# Set environment variables before any imports to avoid PyTorch issues
import importlib.util
import logging
import os

from ztb.utils.errors import safe_operation
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)
STABLE_BASELINES3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None

os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
# Disable CUDA to reduce memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# Additional memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type, List

# import numpy as np

from ztb.utils.config_loader import ConfigLoader
from ztb.utils.path_utils import get_project_root
from ztb.training.config.lagrange_defaults import LAGRANGE_DEFAULTS
from ztb.training.core.config_builder import ConfigBuilder  # 🆕 New config builder

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

from ztb.training.core.config_manager import ConfigManager
from ztb.training.core.algorithm_trainer import AlgorithmTrainer


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


def configure_progress_bar(
    config: Dict[str, Any],
    cli_override: Optional[bool] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Normalize progress bar settings and coordinate Stable-Baselines3 verbosity.

    Args:
        config: Mutable training configuration dictionary.
        cli_override: Optional explicit preference from CLI flags.
        log: Optional logger; defaults to module-level logger.

    Returns:
        bool: True when progress visuals should be enabled.
    """
    if config.get("_progress_configured"):
        return bool(config.get("progress_bar", False))

    logger_obj = log or logger
    progress_preference: Optional[bool] = cli_override

    legacy_top_level = config.pop("progress_bar", None)
    training_section = config.get("training")
    legacy_training = None
    if isinstance(training_section, dict):
        legacy_training = training_section.pop("progress_bar", None)

    if progress_preference is None and legacy_top_level is not None:
        progress_preference = bool(legacy_top_level)
    if progress_preference is None and legacy_training is not None:
        progress_preference = bool(legacy_training)

    ppo_config = config.setdefault("ppo", {})
    if not isinstance(ppo_config, dict):
        logger_obj.warning(
            "PPO configuration expected to be a dict, but received %s. "
            "Disabling progress bar to avoid inconsistent state.",
            type(ppo_config),
        )
        config["progress_bar"] = False
        return False

    if progress_preference is None:
        progress_preference = bool(ppo_config.get("verbose", 0))

    use_progress_bar = bool(progress_preference)

    if STABLE_BASELINES3_AVAILABLE:
        desired_verbose = 1 if use_progress_bar else 0
        current_verbose = ppo_config.get("verbose")
        if current_verbose != desired_verbose:
            logger_obj.info(
                "Stable-Baselines3 detected; adjusting PPO verbose to %s for progress control.",
                desired_verbose,
            )
        ppo_config["verbose"] = desired_verbose
    else:
        logger_obj.info(
            "Stable-Baselines3 not available; %s fallback training progress bar.",
            "enabling" if use_progress_bar else "disabling",
        )
        if not use_progress_bar:
            ppo_config["verbose"] = 0

    config["progress_bar"] = use_progress_bar
    config["_progress_configured"] = True
    return use_progress_bar


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
        super().__init__()
        
        # Store configuration
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.enable_streaming = enable_streaming
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.total_timesteps = total_timesteps
        
        # Initialize components
        self.config_manager = ConfigManager(config)
        self.config_builder = ConfigBuilder(config)  # 🆕 New config builder
        self.algorithm = str(config.get("algorithm", "ppo")).lower()
        self.logger = get_logger(__name__)
        self._config_cache: Optional[Dict[str, Any]] = None
        self._config_cache_key: Optional[tuple[bool, int, Optional[int]]] = None
        
        # Configure progress bar
        self.progress_bar_enabled = configure_progress_bar(self.config, log=self.logger)
        
        # Initialize Discord notifier (disabled in offline mode)
        if config.get("offline_mode", False):
            from ztb.utils import DiscordNotifier
            self.notifier = DiscordNotifier(webhook_url=None)  # Explicitly disable
        else:
            from ztb.utils import DiscordNotifier
            self.notifier = DiscordNotifier()

        # Preserve legacy config object for backward compatibility with tests/tools
        try:
            algorithm_enum = UnifiedAlgorithm(self.algorithm)
        except ValueError:
            algorithm_enum = UnifiedAlgorithm.PPO
        self.config_obj = UnifiedTrainerConfig(
            algorithm=algorithm_enum,
            force=force,
            dry_run=dry_run,
            enable_streaming=enable_streaming,
            stream_batch_size=stream_batch_size,
            max_features=max_features,
            offline_mode=config.get("offline_mode", False),
            total_timesteps=total_timesteps,
        )
    
    # ==================================================================================
    # CONFIGURATION MANAGEMENT HELPERS (Bug #52 fix - unified configuration interface)
    # ==================================================================================
    
    def _get_config_value(self, key: str, sections: Optional[List[str]] = None, default: Any = None) -> Any:
        """
        Get configuration value with priority order.
        
        Priority: top-level > sections (in order) > default
        
        Note: This method delegates to ConfigBuilder.get_config_value()
        """
        return self.config_builder.get_config_value(key, sections, default)
    
    def get_memory_optimization_config(self) -> MemoryOptimizationConfig:
        """
        Extract memory optimization parameters from config.
        
        Note: This method delegates to ConfigBuilder.get_memory_optimization_config()
        """
        return self.config_builder.get_memory_optimization_config()
    
    def get_environment_config(self) -> EnvironmentConfig:
        """
        Extract environment-specific parameters from config.
        
        Note: This method delegates to ConfigBuilder.get_environment_config()
        """
        return self.config_builder.get_environment_config()
    
    def get_ppo_core_config(self) -> PPOCoreConfig:
        """
        Extract PPO algorithm-specific parameters from config.
        
        Note: This method delegates to ConfigBuilder.get_ppo_core_config()
        """
        return self.config_builder.get_ppo_core_config()
    
    def get_feature_config(self) -> Dict[str, Any]:
        """
        Extract feature-related parameters from config.
        
        Note: This method delegates to ConfigBuilder.get_feature_config()
        """
        return self.config_builder.get_feature_config()
    
    def build_unified_config(self) -> Dict[str, Any]:
        """
        Build a unified configuration dict using ConfigManager.
        
        Note: This method delegates to ConfigBuilder.build_unified_config()
        """
        cache_key = (self.enable_streaming, self.stream_batch_size, self.total_timesteps)
        if self._config_cache is not None and self._config_cache_key == cache_key:
            return dict(self._config_cache)

        unified = self.config_builder.build_unified_config(
            enable_streaming=self.enable_streaming,
            stream_batch_size=self.stream_batch_size,
            total_timesteps_override=self.total_timesteps,
        )
        self._config_cache = dict(unified)
        self._config_cache_key = cache_key
        return dict(unified)

    def train(self) -> TrainingResult:
        """Execute training based on algorithm."""
        return safe_operation(
            logger=self.logger,
            operation=self._train_impl,
            context="training_execution",
            default_result=None,
        )

    def _train_impl(self) -> TrainingResult:
        """Implementation of training execution."""
        # Build unified config
        unified_config = self.build_unified_config()
        
        # Apply overrides to config
        if self.total_timesteps is not None:
            unified_config["total_timesteps"] = self.total_timesteps
            self.logger.info(f"Overriding total_timesteps: {self.total_timesteps:,}")
        
        # 訓練開始時のログ出力
        model_name = unified_config.get("model_name", "model")
        algorithm = str(unified_config.get("algorithm", self.algorithm)).lower()
        self.algorithm = algorithm
        
        self.logger.info(f"Starting {algorithm.upper()} training: {model_name}")
        self.logger.info(f"Configuration: {len(unified_config)} settings loaded")
        
        try:
            # Create algorithm trainer
            algorithm_trainer = AlgorithmTrainer(self.config_manager, self.progress_bar_enabled)
            
            # Execute training
            result = algorithm_trainer.train(algorithm, unified_config)
            
            # 訓練成功時のログ
            if result and isinstance(result, dict):
                self.logger.info(f"✅ Training completed successfully")
                if 'model_path' in result:
                    self.logger.info(f"   Model saved: {result['model_path']}")
                if 'log_path' in result:
                    self.logger.info(f"   Logs saved: {result['log_path']}")
            
            return result
            
        except Exception as e:
            # エラー時の詳細ログ
            self.logger.error(f"❌ Training failed: {type(e).__name__}: {str(e)}")
            self.logger.error(f"   Algorithm: {algorithm}")
            self.logger.error(f"   Model: {model_name}")
            raise



    def _train_ppo(self) -> TrainingResult:
        """Train using PPO algorithm with optional SELL bias mitigation."""
        # Set environment variables before importing torch (conditionally)
        import os

        # Only set CUDA-related env vars if CUDA is available and requested
        if self.config.get("enable_cuda_optimizations", True):
            os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Memory optimization: Set garbage collection thresholds
        if self.config.get("aggressive_memory_management", False):
            import gc
            gc.set_threshold(700, 10, 10)  # More aggressive GC

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
            # 🔧 FIX: lagrange_constraintキーもチェック（v392等で使用）
            lagrange_config = self.config.get("lagrange_constraint", {})
            
            def get_lagrange_param(key: str, default: Any = None) -> Any:
                """Check both top-level (lagrange_ prefix) and lagrange_constraint"""
                # Prioritize lagrange_ prefixed key, then lagrange_constraint, then default
                prefixed_key = f"lagrange_{key}"
                return self.config.get(prefixed_key, lagrange_config.get(key, LAGRANGE_DEFAULTS.get(key, default)))
            
            lagrange_params = {}
            # enable_lagrangeは特別扱い（プレフィックスなしとlagrange_constraint.enabledの両方をチェック）
            enable_lagrange = self.config.get("enable_lagrange", lagrange_config.get("enabled", True))
            
            if enable_lagrange:
                lagrange_params = {
                    "r_target": get_lagrange_param("r_target"),
                    "tolerance": get_lagrange_param("tolerance"),
                    "eta": get_lagrange_param("eta"),
                    "lambda_max": get_lagrange_param("lambda_max"),
                    "warmup_steps": get_lagrange_param("warmup_steps"),
                }
                self.logger.info(f"Lagrange parameters: {lagrange_params}")
            
            # Create mitigation params with unified config
            # Note: unified_config contains all required fields for PPOConfig
            mitigation_params = SELLMitigationParams(
                data_path=self.config.get("data_path"),  # type: ignore[arg-type]
                config=unified_config,  # type: ignore[arg-type]
                checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
                checkpoint_interval=checkpoint_interval,
                progress_bar=self.progress_bar_enabled,
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
                progress_bar=self.progress_bar_enabled,
            )
            
            trainer = trainer_class(params=trainer_params)

        # Memory optimization: Periodic cleanup during training
        if self.config.get("aggressive_memory_management", False):
            import gc
            # Schedule memory cleanup after training phases
            import atexit
            atexit.register(gc.collect)
        
        try:
            # Log training start with structured info
            self.logger.info("Starting PPO training", extra={
                "algorithm": "ppo",
                "session_id": self.config.get("session_id", "ppo_session"),
                "total_timesteps": unified_config.get("total_timesteps"),
                "enable_sell_mitigation": enable_sell_mitigation,
                "memory_optimization": memory_opt,
            })
            
            model = trainer.train(session_id=self.config.get("session_id", "ppo_session"))
            
            # Log training completion
            self.logger.info("PPO training completed successfully", extra={
                "session_id": self.config.get("session_id", "ppo_session"),
                "model_saved": model is not None,
            })
            
        except Exception as e:
            self.logger.error("PPO training failed", extra={
                "session_id": self.config.get("session_id", "ppo_session"),
                "error": str(e),
                "error_type": type(e).__name__,
            }, exc_info=True)
            raise
        finally:
            # Aggressive memory cleanup
            if self.config.get("aggressive_memory_management", False):
                gc.collect()

        # Save final model to models directory
        if model is not None:
            import os
            import gc
            from pathlib import Path

            # import pandas as pd

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

            # Save model schema using FeatureSchemaManager (Phase 2)
            session_id = self.config.get('session_id', 'ppo_session')
            self._save_model_schema(session_id, model_dir, df=None)

        return model

    def _train_base_ml(self) -> TrainingResult:
        """Train using base ML reinforcement."""
        unified_config = self.build_unified_config()
        experiment = MLReinforcementExperiment(
            unified_config, total_steps=unified_config.get("total_steps", 1000)
        )
        return experiment.run()

    def _train_iterative(self) -> TrainingResult:
        """Train using iterative approach (from run_1m.py)."""
        unified_config = self.build_unified_config()
        
        # Apply trading mode presets
        trading_mode = unified_config.get("trading_mode", "normal")
        if trading_mode == "scalping":
            # Scalping mode presets
            scalping_defaults = {
                "feature_set": "scalping",
                "timeframe": "15s",
                "reward_scaling": 0.5,
                "transaction_cost": 0.002,
                "max_position_size": 0.3,
                "total_timesteps": 1_000_000,
            }
            for key, value in scalping_defaults.items():
                self.config.setdefault(key, value)
                unified_config.setdefault(key, value)
            # Update session IDs for scalping
            if "scalping" not in self.config.get("session_id", ""):
                self.config["session_id"] = (
                    f"scalping_{self.config.get('session_id', 'session')}"
                )
                self.config["correlation_id"] = (
                    f"scalping_{self.config.get('correlation_id', 'correlation')}"
                )
                unified_config["session_id"] = self.config["session_id"]
                unified_config["correlation_id"] = self.config["correlation_id"]
        else:
            # Normal trading mode presets
            normal_defaults = {
                "feature_set": "full",
                "timeframe": "1m",
                "reward_scaling": 1.0,
                "transaction_cost": 0.001,
                "max_position_size": 1.0,
                "total_timesteps": 100_000,
            }
            for key, value in normal_defaults.items():
                self.config.setdefault(key, value)
                unified_config.setdefault(key, value)

        # Long-running operation confirmation
        total_timesteps = unified_config.get("total_timesteps", 100000)
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
        logger.debug(f"config feature_set = {unified_config.get('feature_set', 'full')}")
        if self.dry_run:
            logger.info(
                f"Dry run: would train with session_id {unified_config.get('session_id', 'iterative_session')}"
            )
            logger.info(
                f"Data path: {unified_config.get('data_path', 'ml-dataset-enhanced.csv')}"
            )
            logger.info(f"Total timesteps: {total_timesteps}")
            logger.info("Setup validation complete")
            return None

        # Import and use run_1m logic
        from ztb.training.scripts.run_1m import main as run_1m_main

        # Get checkpoint interval from config (default: 10000 for iterative training)
        checkpoint_interval = unified_config.get("checkpoint_interval", 10000)

        # Set up arguments for run_1m
        sys.argv = [
            "run_1m.py",
            "--data-path",
            unified_config.get("data_path", "ml-dataset-enhanced.csv"),
            "--correlation-id",
            unified_config.get("session_id", "iterative_session"),
            "--total-timesteps",
            str(total_timesteps),
            "--iterations",
            str(unified_config.get("iterations", 10)),
            "--steps-per-iteration",
            str(unified_config.get("steps_per_iteration", 100000)),
            "--feature-set",
            unified_config.get("feature_set", "full"),
            "--timeframe",
            unified_config.get("timeframe", "1m"),
            "--checkpoint-dir",
            unified_config.get("checkpoint_dir", "checkpoints"),
            "--checkpoint-interval",
            str(checkpoint_interval),
            "--log-dir",
            unified_config.get("log_dir", "logs"),
            "--model-dir",
            unified_config.get("model_dir", "models"),
            "--reward-trade-frequency-penalty",
            str(unified_config.get("reward_trade_frequency_penalty", 0.3)),
            "--reward-trade-frequency-halflife",
            str(unified_config.get("reward_trade_frequency_halflife", 12.0)),
            "--reward-trade-cooldown-steps",
            str(unified_config.get("reward_trade_cooldown_steps", 3)),
            "--reward-trade-cooldown-penalty",
            str(unified_config.get("reward_trade_cooldown_penalty", 0.5)),
            "--reward-max-consecutive-trades",
            str(unified_config.get("reward_max_consecutive_trades", 3)),
            "--reward-consecutive-trade-penalty",
            str(unified_config.get("reward_consecutive_trade_penalty", 0.4)),
            "--transaction-cost",
            str(unified_config.get("transaction_cost", 0.001)),
            "--max-position-size",
            str(unified_config.get("max_position_size", 1.0)),
        ]

        # DEBUG: Print sys.argv
        logger.debug(f"sys.argv = {sys.argv}")
        logger.debug(f"feature-set value = {unified_config.get('feature_set', 'full')}")

        # Add optional arguments
        if self.dry_run:
            sys.argv.append("--dry-run")
        if self.force:
            sys.argv.append("--force")
        if unified_config.get("enable_streaming", False):
            sys.argv.extend(
                [
                    "--enable-streaming",
                    "--stream-batch-size",
                    str(unified_config.get("stream_batch_size", 256)),
                ]
            )
        max_features = (
            unified_config.get("max_features")
            or (unified_config.get("memory_optimization", {}) or {}).get("max_features")
        )
        if max_features is not None:
            sys.argv.extend(["--max-features", str(max_features)])

        data_rows_limit = (
            unified_config.get("data_rows_limit")
            or (unified_config.get("memory_optimization", {}) or {}).get("data_rows_limit")
        )
        if data_rows_limit is not None:
            sys.argv.extend(["--data-rows-limit", str(data_rows_limit)])
        if unified_config.get("offline_mode", False):
            sys.argv.append("--offline-mode")

        # DEBUG: Print final config and sys.argv before calling run_1m_main
        logger.debug(f"Final config feature_set = {unified_config.get('feature_set')}")
        logger.debug(f"Final sys.argv = {sys.argv}")

        def _json_default(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            return str(value)

        serialized_config = json.dumps(unified_config, default=_json_default)
        os.environ["ZTB_UNIFIED_ITERATIVE_CONFIG"] = serialized_config

        try:
            return run_1m_main()
        finally:
            os.environ.pop("ZTB_UNIFIED_ITERATIVE_CONFIG", None)

    def _train_ensemble(self) -> TrainingResult:
        """Train using ensemble approach (load and combine existing models)."""
        unified_config = self.build_unified_config()
        
        from ztb.training.models.ensemble import EnsembleTradingSystem

        # Get model configurations from config
        model_configs = unified_config.get("ensemble_models", [])
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
        unified_config = self.build_unified_config()
        
        from ztb.training.experiments.curriculum_learning import main as curriculum_main

        # Set up environment for curriculum learning
        self.logger.info("Starting curriculum learning (P0→P2 staged approach)")

        # Validate data path
        data_path = unified_config.get("data_path", "ml-dataset-enhanced.csv")
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
    try:
        config = ConfigLoader.load(Path(config_path))
        return config
    except Exception:
        return None
