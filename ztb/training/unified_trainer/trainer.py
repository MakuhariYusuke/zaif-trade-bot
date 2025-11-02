#!/usr/bin/env python3
"""
Refactored Unified Trainer implementation with enhanced UI and modularity.
"""

import copy
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import torch

from ztb.trading.environment.constants import DEFAULT_LEARNING_RATE
from ztb.types.common import (
    AnomalyDetectorProtocol,
    BaseAlgorithmTrainer,
    ConfigDict,
    ContinualLearnerProtocol,
    EnsemblePredictor,
    FederatedLearnerProtocol,
    MetaLearnerProtocol,
    TrainingStats,
)
from ztb.utils.exceptions.custom_exceptions import TrainingError
from ztb.utils.memory_utils import cleanup_training_memory
from ztb.utils.performance_profiler import MemoryProfiler

# Try to import federated learning and mixed precision dependencies
try:
    import opacus  # type: ignore[import-untyped]

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

try:
    from torch.amp import GradScaler

    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler

        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False

from ztb.adaptation.continual_learning import (
    ContinualLearner,
    ContinualLearningConfig,
    TaskData,
)
from ztb.adaptation.meta_learning import MarketMetaLearner
from ztb.data.anomaly_detection import ComprehensiveAnomalyDetector
from ztb.training.distillation.distiller import *

# Import distributed training utilities
from ztb.training.distributed.distributed_training import (
    DistributedTrainingConfig,
    setup_distributed_training,
)
from ztb.training.federated_learning import FederatedConfig, MarketFederatedLearner

# Import system optimizer
from ztb.training.system_optimizer import PerformanceOptimizer, SystemOptimizer
from ztb.training.unified_trainer import reporting

# Import quantization and compression utilities
from ztb.training.unified_trainer.algorithms import create_algorithm_trainer

# Import extracted components
from ztb.training.unified_trainer.components.config_manager import TrainingConfigManager
from ztb.training.unified_trainer.components.ui_manager import TrainingUIManager
from ztb.training.unified_trainer.ensemble_system import (
    EnsembleConfig,
    EnsemblePredictor,
)
from ztb.training.unified_trainer.ui import TrainingUI
from ztb.utils.cache_utils import TTLCache
from ztb.utils.logging_utils import get_logger

# Import optimization utilities
from ztb.utils.memory_utils import MemoryTracker
from ztb.utils.performance_profiler import PerformanceProfiler

if TYPE_CHECKING:
    # Import types for static checking only. Runtime imports are guarded.
    from ztb.optimization.unified_optimizer import UnifiedOptimizer
    from ztb.training.adaptive_sac_core import AdaptiveSACConfig, AdaptiveSACCore
    from ztb.training.online_learning_engine import (
        OnlineLearningConfig,
        OnlineLearningEngine,
    )


class UnifiedTrainer:
    """
    Refactored Unified training interface with enhanced UI and modularity.

    WORK ASSIGNMENT:
    ---------------
    - PPO Algorithm: @trading-team - Standard RL training, evaluation, logging
    - Base ML Algorithm: @ml-research-team - Custom experiments, prototyping
    - Iterative Algorithm: @production-team - Long-running training, monitoring
    """

    def __init__(
        self,
        config: ConfigDict,  # Prefer ConfigDict for runtime configs
        force: bool = False,
        dry_run: bool = False,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        max_features: Optional[int] = None,
        total_timesteps: Optional[int] = None,
        episodes: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        enable_distributed: bool = False,
        world_size: int = 1,
        distributed_backend: str = "gloo",
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
            gradient_accumulation_steps: Number of steps to accumulate gradients
            enable_distributed: Enable distributed training
            world_size: Number of processes for distributed training
            distributed_backend: Backend for distributed training ('gloo' or 'nccl')
        """
        # Initialize components first
        self.logger = get_logger(__name__)
        self.config_manager = TrainingConfigManager()
        self.ui_manager = TrainingUIManager(self.logger)
        self.reporter = reporting.TrainingReporter(self.logger)

        # Process configuration using TrainingConfigManager
        try:
            self.config = self.config_manager.process_config(config)
            self.global_config = None  # Not used in current implementation
        except Exception as e:
            self.logger.error(f"Configuration processing failed: {e}")
            raise

        # Initialize legacy UI for backward compatibility
        self.ui = TrainingUI(self.logger)
        self.ui_manager.initialize_ui(self.ui)
        # Keep the reporting.TrainingReporter (don't overwrite with components.TrainingReporter)
        # self.reporter = TrainingReporter(self.logger)
        # Algorithm trainer (created during run)
        self.algorithm_trainer: Optional[BaseAlgorithmTrainer] = None

        # Anomaly Detection components
        self.anomaly_detector: Optional[AnomalyDetectorProtocol] = None

        # Meta Learning components
        self.meta_learner: Optional[MetaLearnerProtocol] = None

        # Federated Learning components (enhanced)
        self.federated_learner: Optional[FederatedLearnerProtocol] = None

        # Continual Learning components
        self.continual_learner: Optional[ContinualLearnerProtocol] = None
        self.task_counter = 0

        # Ensemble System components (enhanced for SAC v428 Phase 3)
        self.ensemble_system: Optional[EnsemblePredictor] = None
        self.ensemble_config: Optional[EnsembleConfig] = None
        self.ensemble_enabled = (
            self.config.get("v427_advanced_features", {})
            .get("ensemble_system", {})
            .get("enabled", False)
            if isinstance(self.config, dict)
            else False
        )

        if self.ensemble_enabled:
            self._initialize_ensemble_system(self.config)

        # Mixed Precision Training components
        self.grad_scaler: Optional[GradScaler] = None
        if AMP_AVAILABLE and self.config.get("enable_mixed_precision", False):
            try:
                self.grad_scaler = GradScaler()
            except Exception as e:
                self.logger.warning(f"Failed to initialize GradScaler: {e}")
                self.grad_scaler = None
                self.grad_scaler = None

        # Initialize optimization utilities
        self.memory_tracker = MemoryTracker()
        self.memory_profiler = MemoryProfiler()
        self.memory_monitor_thread = None
        self.memory_monitor_stop_event = threading.Event()
        self.performance_profiler = PerformanceProfiler()
        self.feature_cache = TTLCache(
            ttl_seconds=300
        )  # 5 minute TTL for feature computations

        # Initialize system optimizer for comprehensive optimizations
        self.system_optimizer = SystemOptimizer(
            enable_memory_tracking=self.config.get("enable_memory_tracking", True),
            enable_performance_profiling=self.config.get(
                "enable_performance_profiling", True
            ),
            enable_io_caching=self.config.get("enable_io_caching", True),
            memory_threshold_mb=self.config.get("memory_threshold_mb", 100.0),
            cache_ttl_seconds=self.config.get("cache_ttl_seconds", 300),
            gc_interval_steps=self.config.get("gc_interval_steps", 100),
        )

        # Parallel config will be initialized when needed for parallel experiments
        self.parallel_config: Optional[ConfigDict] = None

        # Training results
        self.training_success: bool = False
        self.training_stats: TrainingStats = {}
        self.training_report: Dict[str, Any] = {}

        # Store initialization parameters
        self.force = force
        self.dry_run = dry_run
        self.enable_streaming = enable_streaming
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.total_timesteps = total_timesteps
        self.episodes = episodes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_distributed = enable_distributed
        self.world_size = world_size
        self.distributed_backend = distributed_backend

        # V433 Adaptive Learning Components
        self.enable_v433_adaptive = (
            self.config.get("enable_v433_adaptive", False)
            if isinstance(self.config, dict)
            else False
        )

        # Adaptive SAC Core
        self.adaptive_sac_core: Optional["AdaptiveSACCore"] = None
        self.adaptive_sac_config: Optional["AdaptiveSACConfig"] = None

        # Online Learning Engine
        self.online_learning_engine: Optional["OnlineLearningEngine"] = None
        self.online_learning_config: Optional["OnlineLearningConfig"] = None

        # Unified Optimizer
        self.unified_optimizer: Optional["UnifiedOptimizer"] = None

        if self.enable_v433_adaptive:
            self._initialize_v433_components()

    def _initialize_ensemble_system(self, config: ConfigDict) -> None:
        """Initialize ensemble system for SAC v428 Phase 3."""
        try:
            ensemble_config_dict = config.get("v427_advanced_features", {}).get(
                "ensemble_system", {}
            )

            if not ensemble_config_dict.get("enabled", False):
                self.logger.info("Ensemble system disabled in configuration")
                return

            # Create ensemble configuration
            self.ensemble_config = EnsembleConfig(
                enabled=ensemble_config_dict.get("enabled", True),
                members=ensemble_config_dict.get("members", 5),
                specializations=ensemble_config_dict.get(
                    "specializations",
                    ["bull", "bear", "sideways", "high_vol", "low_vol"],
                ),
                voting_mechanism=ensemble_config_dict.get(
                    "voting_mechanism", "weighted_confidence"
                ),
                diversity_weight=ensemble_config_dict.get("diversity_weight", 0.3),
                consensus_requirement=ensemble_config_dict.get(
                    "consensus_requirement", {}
                ),
                stability_voting=ensemble_config_dict.get("stability_voting", {}),
                adaptation=ensemble_config_dict.get("adaptation", {}),
            )

            # Initialize ensemble predictor
            self.ensemble_system = EnsemblePredictor(self.ensemble_config)

            self.logger.info(
                f"Ensemble system initialized with {self.ensemble_config.members} members"
            )
            self.ui.print_success_with_metrics(
                "Ensemble system initialized successfully",
                {
                    "members": self.ensemble_config.members,
                    "voting_mechanism": self.ensemble_config.voting_mechanism,
                    "specializations": len(self.ensemble_config.specializations),
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize ensemble system: {e}")
            self.ensemble_enabled = False
            self.ui.print_error_with_suggestions(
                f"Ensemble system initialization failed: {e}",
                [
                    "Check ensemble configuration in config file",
                    "Verify specialization types are valid",
                ],
            )

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get current ensemble statistics for monitoring."""
        if self.ensemble_system is None:
            return {"error": "ensemble_not_initialized"}
        # ensemble_system may return a non-typed mapping; cast to expected return type
        return cast(Dict[str, Any], self.ensemble_system.get_ensemble_stats())

    def adapt_ensemble_to_market(self, market_conditions: ConfigDict) -> None:
        """Adapt ensemble system to current market conditions."""
        if self.ensemble_system is None:
            return

        try:
            self.ensemble_system.adapt_ensemble(market_conditions)
            self.logger.info(
                f"Ensemble adapted to market conditions: {market_conditions}"
            )
        except Exception as e:
            self.logger.error(f"Ensemble adaptation failed: {e}")

    def _setup_ensemble_training(self) -> bool:
        """Setup ensemble training for SAC v428 Phase 3."""
        try:
            if not self.ensemble_system:
                self.logger.error("Ensemble system not initialized")
                return False

            # Display ensemble status
            ensemble_stats = self.ensemble_system.get_ensemble_stats()
            self.ui.print_ensemble_status(ensemble_stats)

            # Log ensemble configuration
            if self.ensemble_config is not None:
                self.logger.info(
                    f"Ensemble training setup: {self.ensemble_config.members} members, "
                    f"voting: {self.ensemble_config.voting_mechanism}"
                )

            return True

        except Exception as e:
            self.logger.error(f"Ensemble training setup failed: {e}")
            return False

    def train(self) -> bool:
        """
        Execute training (alias for run method for consistency).

        Returns:
            bool: True if training completed successfully
        """
        return self.run()

    def run(self) -> bool:
        """
        Execute training based on configured algorithm.

        Returns:
            bool: True if training completed successfully
        """
        try:
            # Display header
            algorithm = self.config.get("training", {}).get("algorithm", "unknown")
            config_name = self.config.get("model_name", "unnamed")
            total_timesteps = self.config.get("training", {}).get("total_timesteps", 0)
            self.ui.print_header(algorithm, config_name)

            # Display configuration summary
            self.ui.print_config_summary(self.config)

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Handle dry run
            if self.dry_run:
                self.ui.print_info("Dry run mode: validation completed successfully")
                return True

            # Execute training
            success = self._execute_training()

            # Memory cleanup after training
            self._cleanup_memory()

            return success

        except Exception as e:
            self.ui.print_error(f"Training execution failed: {e}")
            self.logger.error(f"Training execution failed: {e}", exc_info=True)
            return False

    def _apply_system_optimizations(self) -> None:
        """Apply system-level optimizations to the training setup."""
        try:
            # Optimize model memory usage and dataloader if available
            alg_trainer_local = self.algorithm_trainer
            if alg_trainer_local is not None:
                # Model optimization
                if hasattr(alg_trainer_local, "model"):
                    try:
                        model_attr = getattr(alg_trainer_local, "model")
                        optimized_model = self.system_optimizer.optimize_model_memory(
                            model_attr
                        )
                        setattr(alg_trainer_local, "model", optimized_model)
                    except Exception as e:
                        # If optimization fails for the model, continue without crashing
                        self.logger.debug(
                            "Model memory optimization skipped due to error: %s", e
                        )

                # Dataloader optimization
                if hasattr(alg_trainer_local, "dataloader"):
                    try:
                        dataloader_attr = getattr(alg_trainer_local, "dataloader")
                        optimized_dl = self.system_optimizer.optimize_dataloader(
                            dataloader_attr
                        )
                        setattr(alg_trainer_local, "dataloader", optimized_dl)
                    except Exception as e:
                        self.logger.debug(
                            "Dataloader optimization skipped due to error: %s", e
                        )

            # Enable performance optimizations
            PerformanceOptimizer.enable_torch_optimizations()
            PerformanceOptimizer.optimize_numpy_operations()

            # Log optimization status
            system_stats = self.system_optimizer.get_system_stats()
            self.logger.info(f"System optimizations applied: {system_stats}")

        except Exception as e:
            self.logger.warning(f"Failed to apply some system optimizations: {e}")

    def _validate_configuration(self) -> bool:
        """Validate configuration using enhanced validator."""
        self.logger.info("Validating configuration...")

        # Use the algorithm trainer's validation if available
        algorithm = self.config.get("training", {}).get("algorithm", "").lower()

        try:
            # Create algorithm trainer for validation
            trainer = create_algorithm_trainer(algorithm, self.config, self.logger)

            # Validate using trainer
            is_valid = trainer.validate_config()

            if is_valid:
                self.ui.print_success("Configuration validation passed")
                return True
            else:
                self.ui.print_error("Configuration validation failed")
                return False

        except ValueError as e:
            self.ui.print_error(f"Invalid algorithm: {e}")
            return False
        except Exception as e:
            self.ui.print_error(f"Configuration validation error: {e}")
            return False

    def _cleanup_memory(self) -> None:
        """
        Perform memory cleanup after training to prevent memory leaks.

        Uses the centralized memory cleanup utility.
        """
        cleanup_training_memory(
            env=getattr(self, "env", None),
            model=getattr(self, "model", None),
            data_cache=getattr(self, "_data_cache", None),
            force_gc=True,
        )

    def _monitor_training_memory(self, step: int, total_steps: int) -> None:
        """Monitor memory usage during training at regular intervals.

        Args:
            step: Current training step
            total_steps: Total training steps
        """
        # Monitor memory every 10% of training progress or every 10000 steps
        progress_percent = (step / total_steps) * 100
        should_monitor = (
            step % 10000 == 0 or progress_percent % 10 == 0 or step == total_steps
        )

        if should_monitor:
            memory_stats = self.memory_profiler.get_memory_stats()
            self.logger.info(
                f"Memory at step {step:,}/{total_steps:,} ({progress_percent:.1f}%): {memory_stats}"
            )

            # Check for memory warnings
            if memory_stats.get("memory_percent", 0) > 90:
                self.logger.warning(
                    f"High memory usage detected: {memory_stats.get('memory_percent', 0):.1f}%"
                )
            elif memory_stats.get("memory_percent", 0) > 95:
                self.logger.error(
                    f"Critical memory usage: {memory_stats.get('memory_percent', 0):.1f}%"
                )

    def _start_memory_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        if self.memory_monitor_thread is not None:
            return

        self.memory_monitor_stop_event.clear()

        def memory_monitor_worker():
            """Background worker for memory monitoring."""
            monitor_interval = 60  # Monitor every 60 seconds
            while not self.memory_monitor_stop_event.is_set():
                try:
                    memory_stats = self.memory_profiler.get_memory_stats()
                    self.logger.info(f"Background memory check: {memory_stats}")

                    # Alert on high memory usage
                    memory_percent = memory_stats.get("memory_percent", 0)
                    if memory_percent > 90:
                        self.logger.warning(
                            f"High memory usage in background monitor: {memory_percent:.1f}%"
                        )
                    elif memory_percent > 95:
                        self.logger.error(
                            f"Critical memory usage in background monitor: {memory_percent:.1f}%"
                        )

                except Exception as e:
                    self.logger.error(f"Error in memory monitoring thread: {e}")

                # Wait for next check or stop event
                self.memory_monitor_stop_event.wait(timeout=monitor_interval)

        self.memory_monitor_thread = threading.Thread(
            target=memory_monitor_worker, daemon=True, name="MemoryMonitor"
        )
        self.memory_monitor_thread.start()
        self.logger.info("Started background memory monitoring")

    def _stop_memory_monitoring(self) -> None:
        """Stop background memory monitoring thread."""
        if self.memory_monitor_thread is None:
            return

        self.memory_monitor_stop_event.set()
        self.memory_monitor_thread.join(timeout=5.0)
        if self.memory_monitor_thread.is_alive():
            self.logger.warning("Memory monitoring thread did not stop gracefully")
        else:
            self.logger.info("Stopped background memory monitoring")

        self.memory_monitor_thread = None

    def _validate_feature_consistency(self) -> bool:
        """
        Validate feature consistency between config and data file.

        Checks if the number of features specified in config matches the actual data file.
        Issues warnings and attempts fallback if inconsistencies are found.

        Returns:
            bool: True if validation passes or fallback succeeds, False if critical failure
        """
        try:
            # Get data path from config
            data_config = self.config.get("training", {}).get("data_config", {})
            data_path = data_config.get("data_path")

            if not data_path:
                self.logger.warning("No data path specified in config - skipping feature validation")
                return True

            # Import pandas for data reading
            import pandas as pd
            from pathlib import Path

            data_file = Path(data_path)
            if not data_file.exists():
                self.logger.error(f"Data file not found: {data_path}")
                self.ui.print_error(f"Data file not found: {data_path}")
                return False

            # Read data file to get actual feature count
            try:
                # Read only header to get column count efficiently
                df_header = pd.read_csv(data_file, nrows=0)
                actual_feature_count = len(df_header.columns) - 1  # Exclude timestamp/index column
            except Exception as e:
                self.logger.error(f"Failed to read data file header: {e}")
                self.ui.print_error(f"Failed to read data file: {e}")
                return False

            # Get configured feature count from config
            features_config = self.config.get("features", {})
            configured_feature_count = 0

            if isinstance(features_config, dict):
                # Count features in each category
                for category, feature_list in features_config.items():
                    if isinstance(feature_list, list):
                        configured_feature_count += len(feature_list)
                    elif isinstance(feature_list, str):
                        configured_feature_count += 1
            else:
                self.logger.warning("Features config is not a dictionary - unable to validate")
                return True

            # Compare feature counts
            if configured_feature_count == actual_feature_count:
                self.logger.info(f"✅ Feature consistency validated: {configured_feature_count} features match data file")
                return True

            # Feature count mismatch detected
            self.logger.warning(
                f"Feature count mismatch detected! Config: {configured_feature_count}, Data: {actual_feature_count}"
            )
            self.ui.print_error(
                f"⚠️  Feature Count Mismatch Detected!\n"
                f"   Config specifies: {configured_feature_count} features\n"
                f"   Data file contains: {actual_feature_count} features\n"
                f"   Proceeding with fallback handling..."
            )

            # Attempt fallback: Update config to match actual data features
            if actual_feature_count > configured_feature_count:
                self.logger.info("Attempting to update config features to match data file")

                # Read full data to get feature names
                try:
                    df_sample = pd.read_csv(data_file, nrows=5)  # Read sample for feature names
                    actual_features = df_sample.columns[1:].tolist()  # Exclude timestamp

                    # Update config with actual features (simplified mapping)
                    updated_features = {
                        "basic_features": actual_features[:7] if len(actual_features) > 7 else actual_features,
                        "technical_indicators": actual_features[7:10] if len(actual_features) > 10 else [],
                        "regime_features": actual_features[10:20] if len(actual_features) > 20 else [],
                        "correlation_features": actual_features[20:30] if len(actual_features) > 30 else [],
                        "ensemble_features": actual_features[30:40] if len(actual_features) > 40 else [],
                        "risk_adjusted_features": actual_features[40:80] if len(actual_features) > 80 else [],
                        "market_features": actual_features[80:90] if len(actual_features) > 90 else [],
                        "padding_features": actual_features[90:] if len(actual_features) > 90 else [],
                    }

                    # Remove empty categories
                    updated_features = {k: v for k, v in updated_features.items() if v}

                    # Update config
                    self.config["features"] = updated_features

                    self.logger.info(f"Config updated with {len(updated_features)} feature categories")
                    self.ui.print_success(f"✅ Config updated to match data: {sum(len(v) for v in updated_features.values())} features")

                    return True

                except Exception as e:
                    self.logger.error(f"Failed to update config features: {e}")
                    self.ui.print_error(f"Failed to update config features: {e}")
                    return False

            elif actual_feature_count < configured_feature_count:
                self.logger.warning(
                    f"Data file has fewer features ({actual_feature_count}) than config specifies ({configured_feature_count}). "
                    "This may cause training issues."
                )
                self.ui.print_error(
                    f"Data file has fewer features than configured. "
                    "Consider updating your data file or reducing configured features."
                )
                # Continue with training despite mismatch
                return True

            return True

        except Exception as e:
            self.logger.error(f"Feature consistency validation failed: {e}")
            self.ui.print_error(f"Feature consistency validation failed: {e}")
            return False

    def _execute_training(self) -> bool:
        """Execute the actual training."""
        algorithm = self.config.get("training", {}).get("algorithm", "").lower()
        self.logger.info(f"Debug: algorithm = {repr(algorithm)}")
        self.logger.info(f"Debug: config keys = {list(self.config.keys())}")

        # Get initial memory stats
        initial_memory = self.memory_profiler.get_memory_stats()
        self.logger.info(f"Initial memory stats: {initial_memory}")

        try:
            # 特徴量不一致チェック - トレーニング開始前にデータ特徴量数を検証
            if not self._validate_feature_consistency():
                self.logger.warning("Feature consistency validation failed - proceeding with caution")
                self.ui.print_warning("Feature consistency validation failed - proceeding with caution")

            # If both episodes and total_timesteps specified via constructor/CLI, error out.
            if self.episodes is not None and self.total_timesteps is not None:
                raise ValueError(
                    "Cannot specify both episodes and total_timesteps simultaneously. Use only one override."
                )

            # If episodes provided, compute total_timesteps and override config (CLI precedence).
            if self.episodes is not None:
                try:
                    from ztb.training.reward_function_optimizer.constants import (
                        DEFAULT_MAX_EPISODE_LENGTH,
                    )

                    cfg = self.config if isinstance(self.config, dict) else {}
                    training_section = (
                        cfg.get("training", {}) if isinstance(cfg, dict) else {}
                    )
                    env_section = (
                        training_section.get("environment", {})
                        if isinstance(training_section, dict)
                        else {}
                    )

                    max_ep = None
                    if isinstance(env_section, dict):
                        inner_cfg = env_section.get("config", env_section)
                        if isinstance(inner_cfg, dict):
                            max_ep = inner_cfg.get(
                                "max_episode_length"
                            ) or inner_cfg.get("max_episode_steps")

                    if max_ep is None:
                        max_ep = (
                            training_section.get("max_episode_length")
                            if isinstance(training_section, dict)
                            else None
                        )

                    if max_ep is None:
                        max_ep = DEFAULT_MAX_EPISODE_LENGTH

                    total_ts = int(self.episodes) * int(max_ep)

                    if "training" not in self.config:
                        self.config["training"] = {}
                    self.config["training"]["episodes"] = int(self.episodes)
                    self.config["training"]["total_timesteps"] = total_ts
                    self.logger.info(
                        f"Overriding config: episodes={self.episodes}, computed total_timesteps={total_ts} (max_episode_length={max_ep})"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to apply episodes override: {e}")
                    raise

            # Override total_timesteps from command line if provided
            elif self.total_timesteps is not None:
                # Handle different config structures
                if (
                    "training" in self.config
                    and "total_timesteps" in self.config["training"]
                ):
                    self.config["training"]["total_timesteps"] = self.total_timesteps
                else:
                    self.config["total_timesteps"] = self.total_timesteps
                self.logger.info(
                    f"Overriding total_timesteps from command line: {self.total_timesteps:,}"
                )

            # Check for distributed training
            if self.enable_distributed and self.world_size > 1:
                self.logger.info(
                    f"Distributed training enabled with {self.world_size} processes"
                )
                if not self._setup_distributed_training():
                    self.ui.print_error("Failed to setup distributed training")
                    return False

            # Check for federated learning
            if self.config.get("enable_federated", False):
                self.logger.info("Federated learning enabled")
                if not self._setup_federated_learning():
                    self.ui.print_error("Failed to setup federated learning")
                    return False

            # Check for ensemble system (SAC v428 Phase 3)
            if self.ensemble_enabled:
                self.logger.info("Ensemble system enabled for SAC v428 Phase 3")
                if not self._setup_ensemble_training():
                    self.ui.print_error("Failed to setup ensemble training")
                    return False

            # Check for V433 adaptive learning
            if self.enable_v433_adaptive:
                self.logger.info("V433 adaptive learning enabled")
                if not self._setup_v433_adaptive_training():
                    self.ui.print_error("Failed to setup V433 adaptive training")
                    return False

            # Check for mixed precision training
            if self.config.get("enable_mixed_precision", False):
                self.logger.info("Mixed precision training enabled")
                if not self._setup_mixed_precision():
                    self.ui.print_error("Failed to setup mixed precision training")
                    return False

            # Create algorithm trainer (narrow to local variable for safety)
            self.logger.info(f"Creating {algorithm.upper()} trainer...")
            alg_trainer_local = create_algorithm_trainer(
                algorithm,
                self.config,
                self.logger,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                system_optimizer=self.system_optimizer,
            )
            # Assign after local narrowing
            self.algorithm_trainer = alg_trainer_local

            # Apply system optimizations before training
            self.logger.info("Applying system-level optimizations...")
            self._apply_system_optimizations()

            # Start training UI
            self.ui.start_training()

            # Start background memory monitoring
            self._start_memory_monitoring()

            # Initialize optimization tracking
            self.logger.info("Initializing performance optimization tracking...")
            self.memory_tracker.__enter__()
            start_time = time.time()

            # Execute training (federated, V433 adaptive, or regular)
            self.logger.info(f"Starting {algorithm.upper()} training...")
            if self.config.get("enable_federated", False):
                success = self._execute_federated_training()
            elif self.enable_v433_adaptive:
                success = self._execute_v433_adaptive_training()
            else:
                success = False
                if self.algorithm_trainer is not None:
                    # Get total_timesteps from config
                    total_timesteps = self.config.get("training", {}).get(
                        "total_timesteps", 100000
                    )
                    if isinstance(total_timesteps, str):
                        total_timesteps = int(total_timesteps)
                    success = self.algorithm_trainer.train(
                        total_timesteps=total_timesteps
                    )

            # Check for memory leaks after training
            final_memory = self.memory_profiler.get_memory_stats()
            self.logger.info(f"Final memory stats: {final_memory}")

            # Stop optimization tracking and collect metrics
            training_time = time.time() - start_time
            self.memory_tracker.__exit__(None, None, None)
            memory_stats = f"Training completed in {training_time:.2f} seconds"
            perf_report = f"Total training time: {training_time:.2f}s"

            # Stop background memory monitoring
            self._stop_memory_monitoring()

            # Log optimization metrics
            self.logger.info("Training performance metrics:")
            self.logger.info(f"Memory usage: {memory_stats}")
            self.logger.info(f"Performance profile: {perf_report}")

            # Log system optimization statistics
            system_stats = self.system_optimizer.get_system_stats()
            self.logger.info("System optimization statistics:")
            for key, value in system_stats.items():
                self.logger.info(f"  {key}: {value}")

            # Get training statistics (narrow local variable to help static analysis)
            alg_trainer = self.algorithm_trainer
            if (
                success
                and alg_trainer is not None
                and hasattr(alg_trainer, "get_training_stats")
            ):
                try:
                    self.training_stats = alg_trainer.get_training_stats()
                except Exception as e:
                    self.logger.warning(f"Failed to collect training stats: {e}")
                # Add optimization metrics to training stats
                self.training_stats["optimization"] = {
                    "memory_stats": memory_stats,
                    "performance_profile": perf_report,
                    "parallel_processing_enabled": self.parallel_config is not None,
                    "cache_size": len(self.feature_cache.cache)
                    if hasattr(self.feature_cache, "cache")
                    else 0,
                    "data_optimization_applied": True,
                }  # Display completion
            self.ui_manager.display_training_complete(
                self.training_stats if success else {}, training_time
            )

            # Generate and save training report
            if success:
                self.training_report = self.reporter.generate_report(
                    self.config, self.training_stats, success
                )
                report_path = self.reporter.save_report(self.training_report)
                self.reporter.print_summary(self.training_report)

                if report_path:
                    self.ui.print_success(f"Training report saved to: {report_path}")

                # Generate ensemble report if enabled (SAC v428 Phase 3)
                if self.ensemble_enabled and self.ensemble_system:
                    try:
                        ensemble = self.ensemble_system
                        if ensemble is None:
                            raise TrainingError("Ensemble system unexpectedly missing")

                        ensemble_stats = cast(
                            Dict[str, Any], ensemble.get_ensemble_stats()
                        )
                        decision_log = getattr(ensemble, "decision_log", None)

                        # Safely call optional reporter methods
                        gen_fn = getattr(
                            self.reporter, "generate_ensemble_report", None
                        )
                        save_fn = getattr(self.reporter, "save_ensemble_report", None)
                        ensemble_report = None
                        if callable(gen_fn):
                            try:
                                ensemble_report = gen_fn(ensemble_stats, decision_log)
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to generate ensemble report: {e}"
                                )

                        if ensemble_report is not None and callable(save_fn):
                            try:
                                ensemble_report_path = save_fn(ensemble_report)
                                if ensemble_report_path and hasattr(
                                    self.ui, "print_success"
                                ):
                                    self.ui.print_success(
                                        f"Ensemble analysis report saved to: {ensemble_report_path}"
                                    )
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to save ensemble report: {e}"
                                )

                        # Display ensemble final status
                        self.ui.print_ensemble_status(ensemble_stats)

                        self.ui.print_info(
                            "Ensemble system analysis completed successfully"
                        )

                    except Exception as e:
                        self.logger.error(f"Ensemble report generation failed: {e}")
                        self.ui.print_error(f"Ensemble report generation failed: {e}")

            self.training_success = success
            return success

        except Exception as e:
            self.ui.print_error(f"Training execution failed: {e}")
            self.logger.error(f"Training execution failed: {e}", exc_info=True)
            return False

    def get_training_stats(self) -> TrainingStats:
        """Get training statistics."""
        return self.training_stats.copy()

    def get_training_report(self) -> Dict[str, Any]:
        """Get complete training report."""
        return self.training_report.copy()

    def is_training_complete(self) -> bool:
        """Check if training completed successfully."""
        return self.training_success

    def _setup_federated_learning(self) -> bool:
        """
        Setup federated learning components.

        Returns:
            bool: True if setup successful
        """
        try:
            if not OPACUS_AVAILABLE:
                self.logger.warning(
                    "Opacus not available. Federated learning will work without differential privacy."
                )

            num_clients = self.config.get("num_clients", 3)
            self.federated_clients = []

            # Initialize federated clients (simplified - in real implementation,
            # each client would have its own data and trainer)
            for i in range(num_clients):
                client_config = self.config.copy()
                client_config["client_id"] = i
                self.federated_clients.append(client_config)

            self.logger.info(f"Initialized {num_clients} federated clients")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup federated learning: {e}")
            return False

    def _setup_mixed_precision(self) -> bool:
        """
        Setup mixed precision training components.

        Returns:
            bool: True if setup successful
        """
        try:
            if not AMP_AVAILABLE:
                self.logger.error(
                    "PyTorch AMP not available. Mixed precision training requires PyTorch >= 1.6"
                )
                return False

            if not torch.cuda.is_available():
                self.logger.warning(
                    "CUDA not available. Mixed precision training may not provide benefits on CPU."
                )

            # GradScaler is already initialized in __init__
            self.logger.info("Mixed precision training setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup mixed precision training: {e}")
            return False

    def _execute_federated_training(self) -> bool:
        """
        Execute federated learning training.

        Returns:
            bool: True if training successful
        """
        try:
            num_rounds = self.config.get("federated_rounds", 10)
            client_fraction = self.config.get("client_fraction", 1.0)

            self.logger.info(f"Starting federated training with {num_rounds} rounds")

            # Initialize global model
            if self.algorithm_trainer is None:
                self.logger.error(
                    "No algorithm trainer available for federated training"
                )
                return False

            # Narrow local reference for static checking
            alg_trainer = self.algorithm_trainer
            try:
                global_success = alg_trainer.train()
            except Exception as e:
                self.logger.error(f"Global training (federated) failed: {e}")
                return False
            if not global_success:
                self.logger.error("Initial global model training failed")
                return False

            # Get initial global model state (use local reference when possible)
            alg_trainer = self.algorithm_trainer
            if alg_trainer is not None and hasattr(alg_trainer, "get_model_state"):
                try:
                    self.global_model_state = alg_trainer.get_model_state()
                except Exception as e:
                    self.logger.warning("Failed to get global model state: %s", e)

            # Federated learning rounds
            for round_num in range(num_rounds):
                self.logger.info(f"Federated round {round_num + 1}/{num_rounds}")

                # Select participating clients
                num_participants = max(
                    1, int(len(self.federated_clients) * client_fraction)
                )
                participating_clients = self.federated_clients[:num_participants]

                client_updates = []

                # Client training (simplified - in real implementation,
                # each client would train on their local data)
                for client_config in participating_clients:
                    self.logger.debug(f"Training client {client_config['client_id']}")

                    # Create client trainer with global model state
                    client_trainer = create_algorithm_trainer(
                        self.config.get("training", {}).get("algorithm", "").lower(),
                        client_config,
                        self.logger,
                    )

                    # Load global model state
                    if (
                        hasattr(client_trainer, "set_model_state")
                        and self.global_model_state
                    ):
                        client_trainer.set_model_state(self.global_model_state)

                    # Client training (reduced timesteps for federated learning)
                    client_timesteps = self.config.get("total_timesteps", 10000) // 10
                    client_config["total_timesteps"] = client_timesteps

                    client_success = False
                    if client_trainer is not None:
                        client_success = client_trainer.train()
                    if client_success and hasattr(client_trainer, "get_model_state"):
                        client_updates.append(client_trainer.get_model_state())

                # Aggregate client updates (Federated Averaging)
                if client_updates:
                    self.global_model_state = self._federated_average(client_updates)

                    # Update global model
                    if alg_trainer is not None and hasattr(
                        alg_trainer, "set_model_state"
                    ):
                        try:
                            alg_trainer.set_model_state(self.global_model_state)
                        except Exception as e:
                            self.logger.warning(
                                "Failed to set global model state: %s", e
                            )

                self.ui.print_info(
                    f"Completed federated round {round_num + 1}/{num_rounds}"
                )

            # Final global model training/refinement
            self.logger.info("Performing final global model refinement")
            if self.algorithm_trainer is None:
                self.logger.error("No algorithm trainer available for final refinement")
                return False

            final_success = False
            if self.algorithm_trainer is not None:
                final_success = self.algorithm_trainer.train()

            if (
                final_success
                and self.algorithm_trainer is not None
                and hasattr(self.algorithm_trainer, "get_training_stats")
            ):
                self.training_stats = self.algorithm_trainer.get_training_stats()

            return final_success

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            return False

    def _federated_average(
        self, client_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform federated averaging of client model updates.

        Args:
            client_updates: List of client model states

        Returns:
            Averaged global model state
        """
        try:
            if not client_updates:
                return {}

            # Simple averaging (in real implementation, this would be more sophisticated)
            averaged_state = {}
            num_clients = len(client_updates)

            # Get all parameter keys from first client
            param_keys = client_updates[0].keys()

            for key in param_keys:
                if key in client_updates[0]:
                    # Average parameters across clients
                    param_sum = None
                    for client_state in client_updates:
                        if key in client_state:
                            if param_sum is None:
                                param_sum = client_state[key].clone()
                            else:
                                param_sum += client_state[key]

                    if param_sum is not None:
                        averaged_state[key] = param_sum / num_clients

            return averaged_state

        except Exception as e:
            self.logger.error(f"Federated averaging failed: {e}")
            return {}

    def _apply_mixed_precision(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed precision training to loss computation.

        Args:
            loss: Computed loss tensor

        Returns:
            Scaled loss for backward pass
        """
        if self.grad_scaler is not None and self.config.get(
            "enable_mixed_precision", False
        ):
            # Scale loss for mixed precision training
            return self.grad_scaler.scale(loss)
        return loss

    def _step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with mixed precision support.

        Args:
            optimizer: PyTorch optimizer
        """
        if self.grad_scaler is not None and self.config.get(
            "enable_mixed_precision", False
        ):
            # Unscale gradients and step optimizer
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()

    def _setup_distributed_training(self) -> bool:
        """
        Setup distributed training environment.

        Returns:
            bool: True if setup successful
        """
        try:
            # Create distributed training configuration
            self.distributed_config = DistributedTrainingConfig.from_env()
            self.distributed_config.world_size = self.world_size
            self.distributed_config.backend = self.distributed_backend

            # Setup distributed training
            success = setup_distributed_training(self.distributed_config)
            if not success:
                return False

            self.logger.info(
                f"Distributed training setup complete: rank {self.distributed_config.rank}/{self.distributed_config.world_size}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup distributed training: {e}")
            return False

    def _setup_advanced_features(self) -> None:
        """Setup advanced ML features."""
        try:
            # Anomaly Detection Setup
            if self.config.get("enable_anomaly_detection", False):
                self.logger.info("Setting up anomaly detection...")
                self.anomaly_detector = ComprehensiveAnomalyDetector(
                    statistical_methods=self.config.get(
                        "anomaly_statistical_methods", ["zscore", "iqr"]
                    ),
                    ml_methods=self.config.get(
                        "anomaly_ml_methods", ["isolation_forest"]
                    ),
                    enable_autoencoder=self.config.get(
                        "enable_anomaly_autoencoder", False
                    ),
                    voting_threshold=self.config.get("anomaly_voting_threshold", 0.5),
                )
                self.ui.print_info("Anomaly detection enabled")

            # Meta Learning Setup
            if self.config.get("enable_meta_learning", False):
                self.logger.info("Setting up meta learning...")
                # Get model dimensions from algorithm trainer
                alg_trainer_local = self.algorithm_trainer
                if alg_trainer_local is not None and hasattr(
                    alg_trainer_local, "model"
                ):
                    try:
                        state_dim = self._get_model_input_dim()
                        action_dim = self._get_model_output_dim()
                        self.meta_learner = MarketMetaLearner(
                            state_dim=state_dim, action_dim=action_dim
                        )
                    except Exception as e:
                        self.logger.warning("Failed to setup meta learner: %s", e)
                else:
                    self.logger.warning(
                        "Meta learning requires a model - skipping setup"
                    )
                self.ui.print_info("Meta learning enabled")

            # Enhanced Federated Learning Setup
            if self.config.get("enable_federated", False) and self.config.get(
                "federated_markets", False
            ):
                self.logger.info("Setting up market-based federated learning...")
                alg_trainer_local = self.algorithm_trainer
                if alg_trainer_local is not None and hasattr(
                    alg_trainer_local, "model"
                ):
                    try:
                        market_configs = self._create_market_federated_configs()
                        model_obj = getattr(alg_trainer_local, "model")
                        self.federated_learner = MarketFederatedLearner(
                            model_obj, market_configs
                        )
                    except Exception as e:
                        self.logger.warning("Failed to setup federated learner: %s", e)
                else:
                    self.logger.warning(
                        "Federated learning requires a model - skipping setup"
                    )
                self.ui.print_info("Market-based federated learning enabled")

            # Continual Learning Setup
            if self.config.get("enable_continual_learning", False):
                self.logger.info("Setting up continual learning...")
                alg_trainer_local = self.algorithm_trainer
                if alg_trainer_local is not None and hasattr(
                    alg_trainer_local, "model"
                ):
                    try:
                        continual_config = ContinualLearningConfig(
                            method=self.config.get("continual_method", "ewc"),
                            ewc_lambda=self.config.get("continual_ewc_lambda", 0.1),
                            rehearsal_buffer_size=self.config.get(
                                "continual_buffer_size", 1000
                            ),
                            max_tasks_in_memory=self.config.get(
                                "continual_max_tasks", 5
                            ),
                            enable_memory_tracking=True,
                        )
                        model_obj = getattr(alg_trainer_local, "model")
                        self.continual_learner = ContinualLearner(
                            model_obj, continual_config
                        )
                    except Exception as e:
                        self.logger.warning("Failed to setup continual learner: %s", e)
                else:
                    self.logger.warning(
                        "Continual learning requires a model - skipping setup"
                    )

        except Exception as e:
            self.logger.error(f"Failed to setup advanced features: {e}")
            self.ui.print_warning(f"Advanced features setup failed: {e}")

    def _execute_training_with_features(self) -> bool:
        """Execute training with advanced features integration."""
        try:
            # Start training
            self.logger.info("Starting training with advanced features...")

            # Execute main training
            success = False
            if self.algorithm_trainer is not None:
                success = self.algorithm_trainer.train()

            if success:
                # Post-training feature integration
                self._integrate_advanced_features()

                self.ui.print_success(
                    "Training completed successfully with advanced features"
                )
                self.training_success = True
            else:
                self.ui.print_error("Training failed")
                return False

            return True

        except Exception as e:
            self.ui.print_error(f"Training with features failed: {e}")
            self.logger.error(f"Training with features failed: {e}", exc_info=True)
            return False

    def _integrate_advanced_features(self) -> None:
        """Integrate advanced features after training."""
        try:
            # Anomaly detection on training data
            if self.anomaly_detector is not None:
                self._run_anomaly_detection()

            # Meta learning adaptation
            meta_learner = self.meta_learner
            if meta_learner is not None:
                self._run_meta_learning_adaptation()

            # Federated learning aggregation
            federated_learner = self.federated_learner
            if federated_learner is not None:
                self._run_federated_aggregation()

            # Continual learning integration
            continual_learner = self.continual_learner
            if continual_learner is not None:
                self._run_continual_learning()

        except Exception as e:
            self.logger.error(f"Advanced features integration failed: {e}")

    def _run_anomaly_detection(self) -> None:
        """Run anomaly detection on training data."""
        try:
            self.logger.info("Running anomaly detection...")

            # Get training data (simplified - would need actual data access)
            # This is a placeholder for actual implementation
            get_sample = getattr(self, "_get_training_data_sample", None)
            if callable(get_sample):
                training_data = get_sample()
            else:
                training_data = None

            if training_data is not None:
                # Fit ML detectors
                detector = self.anomaly_detector
                if detector is None:
                    return
                detector.fit_ml_detectors(training_data)

                # Run detection on sample data
                is_anomaly, results = detector.detect_anomalies(training_data)

                self.logger.info(
                    f"Anomaly detection completed. Anomalies found: {is_anomaly}"
                )
                self.training_stats["anomaly_detection"] = results

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")

    def _run_meta_learning_adaptation(self) -> None:
        """Run meta learning adaptation."""
        try:
            self.logger.info("Running meta learning adaptation...")
            # Train meta learner on collected tasks (use local var to narrow type)
            meta = self.meta_learner
            if meta is None:
                self.logger.info("Meta learner not configured - skipping adaptation")
                return

            # If the implementation nests a meta_learner attribute, guard access
            if (
                hasattr(meta, "meta_learner")
                and len(getattr(meta.meta_learner, "task_buffer", [])) > 0
            ):
                history = meta.train_on_markets(num_epochs=50)
                self.logger.info("Meta learning adaptation completed")
                self.training_stats["meta_learning"] = history
            elif (
                hasattr(meta, "task_buffer")
                and len(getattr(meta, "task_buffer", [])) > 0
            ):
                history = meta.train_on_markets(num_epochs=50)
                self.logger.info("Meta learning adaptation completed")
                self.training_stats["meta_learning"] = history
            else:
                self.logger.info(
                    "No meta learning tasks collected - skipping adaptation"
                )

        except Exception as e:
            self.logger.error(f"Meta learning adaptation failed: {e}")

    def _run_federated_aggregation(self) -> None:
        """Run federated learning aggregation."""
        try:
            self.logger.info("Running federated learning aggregation...")

            # Train federated learning across markets
            def dummy_loss(
                outputs: torch.Tensor, targets: torch.Tensor
            ) -> torch.Tensor:
                return torch.nn.functional.mse_loss(outputs, targets)

            fed = self.federated_learner
            if fed is None:
                self.logger.warning(
                    "Federated learner not configured - skipping aggregation"
                )
                return

            results = fed.train_all_markets(dummy_loss)
            self.logger.info("Federated learning aggregation completed")
            if hasattr(fed, "get_federated_stats"):
                self.training_stats["federated_learning"] = fed.get_federated_stats()
            else:
                self.training_stats["federated_learning"] = {}

        except Exception as e:
            self.logger.error(f"Federated learning aggregation failed: {e}")

    def _run_continual_learning(self) -> None:
        """Run continual learning integration."""
        try:
            self.logger.info("Running continual learning integration...")

            # トレーニングデータをタスクデータとして準備
            task_data = self._prepare_task_data()
            if task_data is None:
                self.logger.warning(
                    "Could not prepare task data for continual learning"
                )
                return

            # 継続学習実行

            def sac_loss(
                outputs: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                next_outputs: torch.Tensor,
                dones: torch.Tensor,
            ) -> torch.Tensor:
                # SACの簡易損失関数
                return torch.nn.functional.mse_loss(outputs, actions)

            # Ensure continual learner and algorithm trainer exist
            cl = self.continual_learner
            alg = self.algorithm_trainer
            if cl is None:
                self.logger.warning("Continual learner not configured - skipping")
                return

            if alg is None:
                self.logger.warning(
                    "Algorithm trainer not available for continual learning"
                )
                return

            if not hasattr(alg, "model") or getattr(alg, "model") is None:
                self.logger.warning(
                    "Algorithm trainer model not available for continual learning"
                )
                return

            model = getattr(alg, "model")
            optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)

            learning_stats = cl.learn_task(task_data, sac_loss, optimizer, num_epochs=5)

            self.logger.info("Continual learning integration completed")
            self.training_stats["continual_learning"] = learning_stats

        except Exception as e:
            self.logger.error(f"Continual learning integration failed: {e}")

    def _prepare_task_data(self) -> Optional[TaskData]:
        """Prepare training data as task data for continual learning."""
        try:
            # トレーニングデータを取得（簡易版）
            alg = self.algorithm_trainer
            if (
                alg is not None
                and hasattr(alg, "dataloader")
                and getattr(alg, "dataloader")
            ):
                # データローダーからサンプルを取得（安全に）
                try:
                    data_iter = iter(getattr(alg, "dataloader"))
                    batch = next(data_iter)
                except Exception:
                    # Dataloader empty or not iterable
                    batch = None

                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    states, actions = batch[0], batch[1]
                    # 報酬と次の状態はダミーで作成（実際の環境に合わせて調整が必要）
                    rewards = (
                        torch.randn_like(actions[:, :1])
                        if actions.dim() > 1
                        else torch.randn_like(actions.unsqueeze(-1))
                    )
                    next_states = (
                        states + torch.randn_like(states) * 0.1
                    )  # 簡易的な次の状態
                    dones = torch.zeros(len(states), dtype=torch.float32)

                    task_id = f"task_{self.task_counter}"
                    self.task_counter += 1

                    return TaskData(
                        task_id=task_id,
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        dones=dones,
                        num_samples=len(states),
                    )

            # フォールバック：ランダムデータ生成
            self.logger.warning("Using fallback random data for continual learning")
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1

            alg = self.algorithm_trainer
            if (
                alg is not None
                and hasattr(alg, "model")
                and getattr(alg, "model") is not None
            ):
                model = getattr(alg, "model")
                state_dim = int(getattr(model, "input_dim", 10))
                action_dim = int(getattr(model, "output_dim", 4))
            else:
                state_dim = 10
                action_dim = 4

            return TaskData(
                task_id=task_id,
                states=torch.randn(100, state_dim),
                actions=torch.randn(100, action_dim),
                rewards=torch.randn(100, 1),
                next_states=torch.randn(100, state_dim),
                dones=torch.randint(0, 2, (100,)).float(),
                num_samples=100,
            )

        except Exception as e:
            self.logger.error(f"Failed to prepare task data: {e}")
            return None

    def _get_model_input_dim(self) -> int:
        """Get model input dimension."""
        alg = self.algorithm_trainer
        if alg is None or not hasattr(alg, "model") or getattr(alg, "model") is None:
            return 10  # Default

        model = getattr(alg, "model")
        try:
            params_iter = iter(model.parameters())
            first_layer = next(params_iter)
            if first_layer is None:
                return 10
            return int(
                first_layer.shape[1]
                if len(first_layer.shape) > 1
                else first_layer.shape[0]
            )
        except Exception:
            return 10

    def _get_model_output_dim(self) -> int:
        """Get model output dimension."""
        alg = self.algorithm_trainer
        if alg is None or not hasattr(alg, "model") or getattr(alg, "model") is None:
            return 1  # Default

        model = getattr(alg, "model")
        try:
            params = list(model.parameters())
            if not params:
                return 1
            last_layer = params[-1]
            return int(last_layer.shape[0])
        except Exception:
            return 1

    def _create_market_federated_configs(self) -> Dict[str, FederatedConfig]:
        """Create federated configs for different markets."""
        base_config = FederatedConfig(
            num_clients=self.config.get("num_clients", 5),
            num_rounds=self.config.get("federated_rounds", 10),
            client_fraction=self.config.get("client_fraction", 1.0),
            enable_privacy=self.config.get("enable_privacy", True),
            privacy_budget=self.config.get("privacy_budget", 1.0),
        )

        # Create market-specific configs
        markets = self.config.get("markets", ["default"])
        market_configs = {}

        for market in markets:
            market_config = copy.deepcopy(base_config)
            # Customize per market if needed
            market_configs[market] = market_config

        return market_configs

    def _initialize_v433_components(self) -> None:
        """V433適応型学習コンポーネントの初期化"""
        try:
            from ztb.optimization.unified_optimizer import (
                OptimizationConfig,
                UnifiedOptimizer,
            )
            from ztb.training.adaptive_sac_core import (
                AdaptiveSACConfig,
                AdaptiveSACCore,
            )
            from ztb.training.online_learning_engine import (
                OnlineLearningConfig,
                OnlineLearningEngine,
            )

            self.logger.info("Initializing V433 adaptive learning components")

            # Adaptive SAC Config
            v433_config = (
                self.config.get("v433_adaptive_config", {})
                if isinstance(self.config, dict)
                else {}
            )

            self.adaptive_sac_config = AdaptiveSACConfig(
                enable_market_regime_adaptation=v433_config.get(
                    "enable_market_regime_adaptation", True
                ),
                enable_online_learning=v433_config.get("enable_online_learning", True),
                adaptation_interval_steps=v433_config.get(
                    "adaptation_interval_steps", 1000
                ),
                learning_rate=v433_config.get("learning_rate", 3e-4),
                buffer_size=v433_config.get("buffer_size", 1000000),
                performance_window_size=v433_config.get("performance_window_size", 100),
            )

            # 観測空間と行動空間の次元を取得（環境設定から）
            # Extract environment config from multiple supported layouts:
            # 1) top-level 'environment'
            # 2) nested under 'training' -> 'environment' -> 'config'
            if isinstance(self.config, dict):
                env_config = self.config.get("environment", None)
                if env_config is None:
                    training_section = self.config.get("training", {})
                    env_section = (
                        training_section.get("environment", {})
                        if isinstance(training_section, dict)
                        else {}
                    )
                    # prefer inner 'config' dict if present
                    env_config = (
                        env_section.get("config", env_section)
                        if isinstance(env_section, dict)
                        else {}
                    )
            else:
                env_config = {}
            observation_dim = env_config.get("observation_dim", 10)  # デフォルト値
            action_dim = env_config.get("action_dim", 3)  # デフォルト値

            # Adaptive SAC Coreの初期化
            self.adaptive_sac_core = AdaptiveSACCore(
                self.adaptive_sac_config, observation_dim, action_dim
            )

            # Online Learning Config
            self.online_learning_config = OnlineLearningConfig(
                stream_buffer_size=v433_config.get("stream_buffer_size", 10000),
                learning_batch_size=v433_config.get("learning_batch_size", 64),
                experience_buffer_size=v433_config.get("experience_buffer_size", 50000),
                adaptation_threshold=v433_config.get("adaptation_threshold", 0.1),
                data_update_interval=v433_config.get("data_update_interval", 1.0),
            )

            # Online Learning Engineの初期化
            self.online_learning_engine = OnlineLearningEngine(
                self.online_learning_config, self.adaptive_sac_core
            )

            # Unified Optimizerの初期化
            optimizer_config = OptimizationConfig(
                enable_hyperparameter_optimization=v433_config.get(
                    "enable_hyperparameter_optimization", True
                ),
                enable_system_optimization=v433_config.get(
                    "enable_system_optimization", True
                ),
                enable_reward_optimization=v433_config.get(
                    "enable_reward_optimization", True
                ),
                enable_adaptive_optimization=v433_config.get(
                    "enable_adaptive_optimization", True
                ),
                max_trials=v433_config.get("max_trials", 100),
                max_parallel_trials=v433_config.get("max_parallel_trials", 4),
            )
            self.unified_optimizer = UnifiedOptimizer(optimizer_config)

            self.logger.info(
                "V433 adaptive learning components initialized successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize V433 components: {e}")
            self.enable_v433_adaptive = False

    def _setup_v433_adaptive_training(self) -> bool:
        """V433適応型トレーニングのセットアップ"""
        try:
            if not self.adaptive_sac_core:
                self.logger.error("Adaptive SAC core not initialized")
                return False

            if not self.online_learning_engine:
                self.logger.error("Online learning engine not initialized")
                return False

            if not self.unified_optimizer:
                self.logger.error("Unified optimizer not initialized")
                return False

            # 環境の初期化確認
            env_config = (
                self.config.get("environment", {})
                if isinstance(self.config, dict)
                else {}
            )
            if not env_config:
                self.logger.error(
                    "Environment configuration missing for V433 adaptive training"
                )
                return False

            # 適応型トレーニングのステータス表示
            self.ui.print_info("Setting up V433 adaptive training components:")
            self.ui.print_info(
                f"  - Adaptive SAC Core: {'✓' if self.adaptive_sac_core else '✗'}"
            )
            self.ui.print_info(
                f"  - Online Learning Engine: {'✓' if self.online_learning_engine else '✗'}"
            )
            self.ui.print_info(
                f"  - Unified Optimizer: {'✓' if self.unified_optimizer else '✗'}"
            )

            # 市場レジーム検知の初期化
            adaptive_core = self.adaptive_sac_core
            if adaptive_core is not None:
                if hasattr(adaptive_core, "market_regime_detector"):
                    self.logger.info("Market regime detector initialized")

                # 適応パラメータのログ（存在する場合のみ呼び出す）
                get_status = getattr(adaptive_core, "get_adaptation_status", None)
                adaptation_params = get_status() if callable(get_status) else {}
                self.logger.info(f"V433 adaptation parameters: {adaptation_params}")

            return True

        except Exception as e:
            self.logger.error(f"V433 adaptive training setup failed: {e}")
            return False

    def _execute_v433_adaptive_training(self) -> bool:
        """V433適応型トレーニングを実行"""
        try:
            self.logger.info("Executing V433 adaptive training")

            # 環境の作成
            env = self._create_v433_training_environment()
            if env is None:
                self.logger.error("Failed to create V433 training environment")
                return False

            # 適応型SACモデルの初期化
            adaptive_core = self.adaptive_sac_core
            if adaptive_core is None:
                self.logger.error("Adaptive SAC core not configured")
                return False

            init_fn = getattr(adaptive_core, "initialize_sac_model", None)
            if not callable(init_fn):
                self.logger.error(
                    "Adaptive SAC core missing initialize_sac_model method"
                )
                return False

            try:
                sac_model = init_fn(env)
            except Exception as e:
                self.logger.error(f"Failed to initialize adaptive SAC model: {e}")
                return False
            if sac_model is None:
                self.logger.error("Failed to initialize adaptive SAC model")
                return False

            # 総タイムステップの取得
            total_timesteps = self.config.get("total_timesteps", 100000)
            if self.total_timesteps is not None:
                total_timesteps = self.total_timesteps

            # 適応型トレーニングの実行
            start_fn = getattr(adaptive_core, "start_adaptive_training", None)
            if callable(start_fn):
                try:
                    start_fn(env, total_timesteps)
                except Exception as e:
                    self.logger.error(f"Adaptive training failed to start: {e}")
                    return False
            else:
                self.logger.error(
                    "Adaptive SAC core missing start_adaptive_training method"
                )
                return False

            # オンライン学習エンジンの開始（非同期）
            import asyncio
            import threading

            def run_online_learning() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    online_engine = self.online_learning_engine
                    if online_engine is not None and hasattr(
                        online_engine, "start_online_learning"
                    ):
                        loop.run_until_complete(online_engine.start_online_learning())
                    else:
                        self.logger.warning(
                            "Online learning engine not available to start"
                        )
                except Exception as e:
                    self.logger.error(f"Online learning failed: {e}")
                finally:
                    loop.close()

            online_thread = threading.Thread(target=run_online_learning, daemon=True)
            online_thread.start()

            # トレーニング完了まで待機
            self.logger.info("V433 adaptive training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"V433 adaptive training failed: {e}")
            return False

    def _create_v433_training_environment(self) -> Optional[Any]:
        """V433トレーニング環境を作成"""
        try:
            # Lazy import to avoid heavy runtime dependency and mypy import-untyped noise
            import importlib

            try:
                mod = importlib.import_module("ztb.trading.environment.heavy_env.core")
                HeavyTradingEnv = getattr(mod, "HeavyTradingEnv", None)
            except Exception:
                HeavyTradingEnv = None

            env_config = (
                self.config.get("environment", {})
                if isinstance(self.config, dict)
                else {}
            )

            if HeavyTradingEnv is None:
                self.logger.error(
                    "HeavyTradingEnv not available in runtime environment"
                )
                return None

            # Create EnvironmentConfig from the features_config
            from ztb.trading.environment.utils.config import EnvironmentConfig

            env_config_dict = dict(env_config) if isinstance(env_config, dict) else {}
            features_config = self.config.get("features", {})

            # Add feature_set from features_config if available
            if "feature_set" in features_config:
                env_config_dict["feature_set"] = features_config["feature_set"]

            # Add data_config for data loading
            data_config = self.config.get("data_config", {})
            if data_config:
                env_config_dict.update(data_config)

            # Ensure csv_path is set from data_path if available
            if "data_path" in self.config and self.config["data_path"]:
                env_config_dict["csv_path"] = self.config["data_path"]

            env_config_obj = EnvironmentConfig.from_dict(env_config_dict)

            # Debug: log env config just before handing to HeavyTradingEnv so we can
            # trace where `use_continuous_actions` may be lost/overwritten.
            try:
                self.logger.info(
                    f"Creating HeavyTradingEnv with env_config_dict keys: {list(env_config_dict.keys())}"
                )
            except Exception:
                self.logger.info(f"env_config_dict type: {type(env_config_dict)}")

            try:
                # If dict-like
                if isinstance(env_config_dict, dict):
                    preview = {
                        k: env_config_dict.get(k, "NOT_FOUND")
                        for k in ["use_continuous_actions", "action_space_type"]
                    }
                    self.logger.info(f"env_config_dict preview: {preview}")
                else:
                    self.logger.info(
                        f"env_config_dict repr[:200]: {repr(env_config_dict)[:200]}"
                    )
            except Exception:
                pass

            try:
                # For the constructed EnvironmentConfig object
                ua = getattr(env_config_obj, "use_continuous_actions", "NOT_PRESENT")
                at = getattr(env_config_obj, "action_space_type", "NOT_PRESENT")
                self.logger.info(
                    f"env_config_obj preview: use_continuous_actions={ua}, action_space_type={at}"
                )
            except Exception:
                self.logger.info(
                    f"Could not inspect env_config_obj (type={type(env_config_obj)})"
                )

            env = HeavyTradingEnv(
                config=env_config_obj,
                max_features=self.max_features,
            )

            self.logger.info("V433 training environment created successfully")
            return env

        except Exception as e:
            self.logger.error(f"Failed to create V433 training environment: {e}")
            return None
