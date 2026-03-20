#!/usr/bin/env python3
# ruff: noqa: E402
"""
Refactored Unified Trainer implementation with enhanced UI and modularity.
"""

import copy
import os
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, cast

import pandas as pd

# Avoid importing torch at module import time to prevent DLL initialization
# errors on machines without GPU drivers / CUDA available. Import on-demand
# to keep the module importable for tests that do not need Torch.
try:
    import torch
except Exception:
    torch = None

# Try to detect optional dependencies without importing heavy packages at
# module import time. Use importlib.util.find_spec to check availability which
# avoids side-effects or expensive imports during test collection.
import importlib.util

from ztb.analysis.common.types import TrainerProtocol
from ztb.training.constants import DEFAULT_LEARNING_RATE
from ztb.training.trainers.base_trainer import BaseTrainer
from ztb.types.common import (
    AnomalyDetectorProtocol,
    ConfigDict,
    ContinualLearnerProtocol,
    FederatedLearnerProtocol,
    MetaLearnerProtocol,
    TrainingStats,
)
from ztb.io.data_loader import DataLoader
from ztb.cache.parquet_io import read_parquet
from ztb.utils.error_utils import safe_execute
from ztb.utils.exceptions.custom_exceptions import TrainingError
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.memory_utils import cleanup_training_memory
from ztb.utils.performance_profiler import MemoryProfiler
from ztb.metrics.metrics import sharpe_ratio
from ztb.features.generators.multi_timeframe.datetime_utils import safe_to_datetime_series

OPACUS_AVAILABLE = importlib.util.find_spec("opacus") is not None

AMP_AVAILABLE = False
if torch is not None:
    def _try_amp_import():
        try:
            from torch.amp import GradScaler
            return True
        except Exception:
            return False
    
    def _try_cuda_amp_import():
        try:
            from torch.cuda.amp import GradScaler
            return True
        except Exception:
            return False
    
    AMP_AVAILABLE = _try_amp_import() or _try_cuda_amp_import()

from ztb.adaptation.continual_learning import (
    ContinualLearner,
    ContinualLearningConfig,
    TaskData,
)
from ztb.adaptation.meta_learning import MarketMetaLearner
from ztb.training.checkpoint.checkpoint_manager import TrainingCheckpointManager

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
from ztb.training.unified_trainer.runtime_flags import (
    resolve_ensemble_enabled,
    resolve_trainer_runtime_flags,
)
from ztb.training.unified_trainer.ui import TrainingUI
from ztb.utils.cache_utils import TTLCache
from ztb.utils.logging_utils import get_logger

# Import optimization utilities
from ztb.utils.memory_utils import OperationMemoryTracker
from ztb.utils.performance_profiler import PerformanceProfiler
from ztb.metrics.metrics import win_rate as calculate_win_rate

if TYPE_CHECKING:
    # Import types for static checking only. Runtime imports are guarded.
    from ztb.training.unified_trainer.base.base_trainer import BaseAlgorithmTrainer

    # Type-only import to avoid name collision with runtime EnsemblePredictor

class UnifiedTrainer(BaseTrainer, TrainerProtocol):
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
        max_features: int | None = None,
        total_timesteps: int | None = None,
        episodes: int | None = None,
        gradient_accumulation_steps: int = 1,
        enable_distributed: bool = False,
        world_size: int = 1,
        distributed_backend: str = "gloo",
        resume: bool = False,
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
        # Initialize base class
        super().__init__(config)

        # Initialize components first
        self.logger = get_logger(__name__)
        sig_policy = os.getenv("ZTB_SIGINT_POLICY")
        if sig_policy:
            from ztb.utils.signal_utils import configure_signal_handling

            configure_signal_handling(sig_policy, self.logger)
        self.config_manager = TrainingConfigManager()
        self.ui_manager = TrainingUIManager(self.logger)
        self.reporter = reporting.TrainingReporter(self.logger)
        self.resume = resume

        # Process configuration using TrainingConfigManager
        try:
            self.config = self.config_manager.process_config(config)
            self.global_config = None  # Not used in current implementation
        except Exception as e:
            self.logger.error(f"Configuration processing failed: {e}")
            raise

        # Expose convenient top-level attributes for tests/legacy code
        self.algorithm = (
            self.config.get("training", {}).get("algorithm")
            if isinstance(self.config, dict)
            else None
        )

        # Initialize legacy UI for backward compatibility
        self.ui = TrainingUI(self.logger)
        self.ui_manager.initialize_ui(self.ui)
        # Keep the reporting.TrainingReporter (don't overwrite with components.TrainingReporter)
        # self.reporter = TrainingReporter(self.logger)
        # Algorithm trainer (created during run)
        self.algorithm_trainer: BaseAlgorithmTrainer | None = None
        # Exposed model handle (if algorithm produces one during training)
        self._model: object | None = None
        # Optional explicit env override (if set externally)
        self._env: object | None = None

        # Anomaly Detection components
        self.anomaly_detector: AnomalyDetectorProtocol | None = None

        # Meta Learning components
        self.meta_learner: MetaLearnerProtocol | None = None

        # Federated Learning components (enhanced)
        self.federated_learner: FederatedLearnerProtocol | None = None

        # Continual Learning components
        self.continual_learner: ContinualLearnerProtocol | None = None
        self.task_counter = 0

        # Ensemble System components (enhanced for SAC v428 Phase 3)
        self.ensemble_system: EnsemblePredictor | None = None
        self.ensemble_config: EnsembleConfig | None = None
        self.ensemble_enabled = resolve_ensemble_enabled(self.config)

        if self.ensemble_enabled:
            self._initialize_ensemble_system(self.config)

        # Mixed Precision Training components
        self.grad_scaler: GradScaler | None = None
        if AMP_AVAILABLE and self.config.get("enable_mixed_precision", False):
            try:
                self.grad_scaler = GradScaler()
            except Exception as e:
                self.logger.warning(f"Failed to initialize GradScaler: {e}")
                self.grad_scaler = None

        # Initialize optimization utilities
        self.memory_tracker = OperationMemoryTracker()
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
            memory_threshold_mb=self.config.get("memory_threshold_mb", 1500.0),
            cache_ttl_seconds=self.config.get("cache_ttl_seconds", 300),
            gc_interval_steps=self.config.get("gc_interval_steps", 1000),
        )

        # Parallel config will be initialized when needed for parallel experiments
        self.parallel_config: ConfigDict | None = None

        # Training results
        self.training_success: bool = False
        self.training_stats: TrainingStats = {}
        self.training_report: dict[str, object] = {}
        self._feature_consistency_checked = False
        self._feature_consistency_ok = True

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

        # V433 Adaptive Learning Components: archived in 030#, dead code removed in 063#

    @property
    def model(self) -> object | None:
        """Return the underlying model from the algorithm trainer when available."""
        alg_trainer = self.algorithm_trainer
        if alg_trainer is not None and hasattr(alg_trainer, "model"):
            model = getattr(alg_trainer, "model")
            if model is not None:
                return model
        return self._model

    @model.setter
    def model(self, value: object | None) -> None:
        """set the model and propagate to the algorithm trainer if present."""
        self._model = value
        alg_trainer = self.algorithm_trainer
        if alg_trainer is not None and hasattr(alg_trainer, "model"):
            try:
                setattr(alg_trainer, "model", value)
            except Exception as e:
                self.logger.debug("model propagation to alg_trainer failed: %s", e)

    @property
    def env(self) -> object | None:
        """Return the training environment if available."""
        if self._env is not None:
            return self._env

        model = self.model
        if model is not None:
            if hasattr(model, "env") and getattr(model, "env") is not None:
                return getattr(model, "env")
            if hasattr(model, "get_env"):
                try:
                    return model.get_env()
                except Exception as e:
                    self.logger.debug("model.get_env() failed: %s", e)
                    return None

        alg_trainer = self.algorithm_trainer
        if alg_trainer is not None and hasattr(alg_trainer, "env"):
            try:
                return getattr(alg_trainer, "env")
            except Exception as e:
                self.logger.debug("alg_trainer.env access failed: %s", e)
                return None
        return None

    @env.setter
    def env(self, value: object | None) -> None:
        """set the training environment and propagate when possible."""
        self._env = value
        alg_trainer = self.algorithm_trainer
        if alg_trainer is not None and hasattr(alg_trainer, "env"):
            try:
                setattr(alg_trainer, "env", value)
            except Exception as e:
                self.logger.debug("env propagation to alg_trainer failed: %s", e)

        model = self.model
        if model is not None and hasattr(model, "set_env"):
            try:
                model.set_env(value)
            except Exception as e:
                self.logger.debug("model.set_env() failed: %s", e)

    def get_environment_metrics(self) -> dict[str, object]:
        """Extract environment metrics such as balance and trade count."""
        from ztb.training.utils.env_metrics import extract_trainer_env_metrics

        return extract_trainer_env_metrics(self, include_optional=False)

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

    def get_ensemble_stats(self) -> dict[str, object]:
        """Get current ensemble statistics for monitoring."""
        if self.ensemble_system is None:
            return {"error": "ensemble_not_initialized"}
        # ensemble_system may return a non-typed mapping; cast to expected return type
        return cast(dict[str, object], self.ensemble_system.get_ensemble_stats())

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

    def run_training(self) -> None:
        """
        Run training with error handling.
        """
        success = self.train()
        if not success:
            raise RuntimeError("Training failed")

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
            self.config.get("training", {}).get("total_timesteps", 0)
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

                # Additional feature consistency validation
                self._feature_consistency_ok = self._validate_feature_consistency()
                self._feature_consistency_checked = True
                if not self._feature_consistency_ok:
                    return False

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
        feature_cache = getattr(self, "feature_cache", None)
        if feature_cache is not None and hasattr(feature_cache, "clear"):
            try:
                feature_cache.clear()
            except Exception as e:
                self.logger.debug("Feature cache cleanup skipped due to error: %s", e)

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
        if total_steps <= 0:
            return

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
            memory_percent = float(memory_stats.get("memory_percent", 0))
            if memory_percent > 95:
                self.logger.error(
                    f"Critical memory usage: {memory_percent:.1f}%"
                )
            elif memory_percent > 90:
                self.logger.warning(
                    f"High memory usage detected: {memory_percent:.1f}%"
                )

    def _start_memory_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        if (
            self.memory_monitor_thread is not None
            and self.memory_monitor_thread.is_alive()
        ):
            return
        self.memory_monitor_thread = None

        self.memory_monitor_stop_event.clear()

        raw_interval = self.config.get("memory_monitor_interval_seconds", 60)
        try:
            monitor_interval = max(5, int(raw_interval))
        except (TypeError, ValueError):
            monitor_interval = 60

        def memory_monitor_worker():
            """Background worker for memory monitoring."""
            while not self.memory_monitor_stop_event.is_set():
                try:
                    memory_stats = self.memory_profiler.get_memory_stats()
                    self.logger.debug(f"Background memory check: {memory_stats}")

                    # Alert on high memory usage
                    memory_percent = float(memory_stats.get("memory_percent", 0))
                    if memory_percent > 95:
                        self.logger.error(
                            f"Critical memory usage in background monitor: {memory_percent:.1f}%"
                        )
                    elif memory_percent > 90:
                        self.logger.warning(
                            f"High memory usage in background monitor: {memory_percent:.1f}%"
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
        if self.memory_monitor_thread.is_alive():
            self.memory_monitor_thread.join(timeout=5.0)
            if self.memory_monitor_thread.is_alive():
                self.logger.warning("Memory monitoring thread did not stop gracefully")
            else:
                self.logger.info("Stopped background memory monitoring")

        self.memory_monitor_thread = None

    @contextmanager
    def _safe_memory_tracking(self):
        """Ensure memory tracker enter/exit symmetry even on exceptions."""
        tracker_entered = False
        try:
            self.memory_tracker.__enter__()
            tracker_entered = True
        except Exception as e:
            self.logger.warning(
                "Memory tracker initialization failed; continuing without tracker: %s",
                e,
            )

        try:
            yield
        finally:
            if tracker_entered:
                try:
                    self.memory_tracker.__exit__(None, None, None)
                except Exception as e:
                    self.logger.warning("Memory tracker cleanup failed: %s", e)

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
            data_path = data_config.get("data_path") or self.config.get("data_path")

            if not data_path:
                self.logger.warning(
                    "No data path specified in config - skipping feature validation"
                )
                return True

            from pathlib import Path

            data_file = Path(data_path)
            if not data_file.exists():
                self.logger.error(f"Data file not found: {data_path}")
                self.ui.print_error(f"Data file not found: {data_path}")
                return False

            # Read data file to get actual feature count
            try:
                cache_key = (
                    f"feature_header_columns:{data_file}:{data_file.stat().st_mtime_ns}"
                )
                cached_columns = self.feature_cache.get(cache_key)
                if isinstance(cached_columns, list):
                    header_columns = [str(col) for col in cached_columns]
                else:
                    # Read only header to get column count efficiently
                    df_header = DataLoader.load_csv(data_file, nrows=0)
                    header_columns = [str(col) for col in df_header.columns]
                    self.feature_cache.set(cache_key, header_columns)
                actual_features = header_columns[1:] if len(header_columns) > 1 else []
                actual_feature_count = len(actual_features)  # Exclude timestamp/index
            except Exception as e:
                self.logger.error(f"Failed to read data file header: {e}")
                self.ui.print_error(f"Failed to read data file: {e}")
                return False

            # Get configured feature count from config
            configured_feature_count_raw = (
                self.config.get("max_features", actual_feature_count)
                or actual_feature_count
            )
            try:
                configured_feature_count = int(configured_feature_count_raw)
            except (TypeError, ValueError):
                configured_feature_count = actual_feature_count

            # Compare feature counts
            if configured_feature_count == actual_feature_count:
                self.logger.info(
                    f"✅ Feature consistency validated: {configured_feature_count} features match data file"
                )
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
                self.logger.info(
                    "Attempting to update config features to match data file"
                )

                try:
                    # Update config with actual features (simplified mapping)
                    updated_features = {
                        "basic_features": actual_features[:7]
                        if len(actual_features) > 7
                        else actual_features,
                        "technical_indicators": actual_features[7:10]
                        if len(actual_features) > 10
                        else [],
                        "regime_features": actual_features[10:20]
                        if len(actual_features) > 20
                        else [],
                        "correlation_features": actual_features[20:30]
                        if len(actual_features) > 30
                        else [],
                        "ensemble_features": actual_features[30:40]
                        if len(actual_features) > 40
                        else [],
                        "risk_adjusted_features": actual_features[40:80]
                        if len(actual_features) > 80
                        else [],
                        "market_features": actual_features[80:90]
                        if len(actual_features) > 90
                        else [],
                        "padding_features": actual_features[90:]
                        if len(actual_features) > 90
                        else [],
                    }

                    # Remove empty categories
                    updated_features = {k: v for k, v in updated_features.items() if v}

                    # Update config
                    self.config["features"] = updated_features

                    self.logger.info(
                        f"Config updated with {len(updated_features)} feature categories"
                    )
                    self.ui.print_success(
                        f"✅ Config updated to match data: {sum(len(v) for v in updated_features.values())} features"
                    )

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
                    "Data file has fewer features than configured. "
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
            if not self._feature_consistency_checked:
                self._feature_consistency_ok = self._validate_feature_consistency()
                self._feature_consistency_checked = True
            if not self._feature_consistency_ok:
                self.logger.warning(
                    "Feature consistency validation failed - proceeding with caution"
                )
                self.ui.print_warning(
                    "Feature consistency validation failed - proceeding with caution"
                )

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

            runtime_flags = resolve_trainer_runtime_flags(
                self.config,
                enable_distributed=self.enable_distributed,
                world_size=self.world_size,
                ensemble_enabled=self.ensemble_enabled,
            )

            # Check for distributed training
            if runtime_flags.distributed_training_enabled:
                self.logger.info(
                    f"Distributed training enabled with {self.world_size} processes"
                )
                if not self._setup_distributed_training():
                    self.ui.print_error("Failed to setup distributed training")
                    return False

            # Check for federated learning
            if runtime_flags.federated_learning_enabled:
                self.logger.info("Federated learning enabled")
                if not self._setup_federated_learning():
                    self.ui.print_error("Failed to setup federated learning")
                    return False

            # Check for ensemble system (SAC v428 Phase 3)
            if runtime_flags.ensemble_enabled:
                self.logger.info("Ensemble system enabled for SAC v428 Phase 3")
                if not self._setup_ensemble_training():
                    self.ui.print_error("Failed to setup ensemble training")
                    return False

            # Check for mixed precision training
            if runtime_flags.mixed_precision_enabled:
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

            # Resume from checkpoint if requested
            if self.resume:
                checkpoint_dir = self.config.get("checkpoint_dir", "models/checkpoints")
                checkpoint_manager = TrainingCheckpointManager(save_dir=checkpoint_dir)
                snapshot = checkpoint_manager.load_latest()
                if snapshot:
                    self.logger.info(
                        f"Resuming from checkpoint at step {snapshot.step}"
                    )
                    if self.algorithm_trainer and hasattr(
                        self.algorithm_trainer, "model"
                    ):
                        # Restore model state
                        model_state = snapshot.payload.get("model_state")
                        if model_state:
                            self.algorithm_trainer.model.load_state_dict(model_state)
                            self.logger.info("Model state restored from checkpoint")
                        else:
                            self.logger.warning("No model state found in checkpoint")
                    else:
                        self.logger.warning(
                            "Algorithm trainer has no model attribute for checkpoint restore"
                        )
                else:
                    self.logger.warning("No checkpoint found for resume")

            # Apply system optimizations before training
            self.logger.info("Applying system-level optimizations...")
            self._apply_system_optimizations()

            # Start training UI
            self.ui.start_training()

            # Start background memory monitoring
            self._start_memory_monitoring()

            # Initialize optimization tracking
            self.logger.info("Initializing performance optimization tracking...")
            start_time = time.perf_counter()
            success = False
            with self._safe_memory_tracking():
                # Execute training (federated or regular)
                self.logger.info(f"Starting {algorithm.upper()} training...")
                if self.config.get("enable_federated", False):
                    success = self._execute_federated_training()
                elif self.algorithm_trainer is not None:
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
            training_time = time.perf_counter() - start_time
            memory_stats = f"Training completed in {training_time:.2f} seconds"
            perf_report = f"Total training time: {training_time:.2f}s"

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
                            dict[str, object], ensemble.get_ensemble_stats()
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
        finally:
            self._stop_memory_monitoring()

    def get_training_stats(self) -> TrainingStats:
        """Get training statistics."""
        return self.training_stats.copy()

    def get_training_report(self) -> dict[str, object]:
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
        self, client_updates: list[dict[str, object]]
    ) -> dict[str, object]:
        """
        Perform federated averaging of client model updates.

        Args:
            client_updates: list of client model states

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

    def _step_optimizer(self, optimizer: "torch.optim.Optimizer") -> None:
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
            runtime_flags = resolve_trainer_runtime_flags(
                self.config,
                enable_distributed=self.enable_distributed,
                world_size=self.world_size,
                ensemble_enabled=self.ensemble_enabled,
            )
            # Anomaly Detection Setup
            if self.config.get("enable_anomaly_detection", False):
                self.logger.info("Setting up anomaly detection...")
                try:
                    from ztb.data.anomaly_detection import ComprehensiveAnomalyDetector
                except Exception as e:
                    self.logger.warning("Anomaly detection unavailable: %s", e)
                else:
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
            if runtime_flags.market_federated_learning_enabled:
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
            if runtime_flags.continual_learning_enabled:
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

            fed.train_all_markets(dummy_loss)
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

    def _prepare_task_data(self) -> TaskData | None:
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
                except Exception as e:
                    # Dataloader empty or not iterable
                    self.logger.debug("Dataloader iteration failed: %s", e)
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
        except Exception as e:
            self.logger.debug("_get_model_input_dim fallback to 10: %s", e)
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
        except Exception as e:
            self.logger.debug("_get_model_output_dim fallback to 1: %s", e)
            return 1

    def _create_market_federated_configs(self) -> dict[str, FederatedConfig]:
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

    # --- V433 adaptive methods removed in 063# (dead code, enable_v433_adaptive=False) ---
    # Removed: _initialize_v433_components, _setup_v433_adaptive_training,
    #          _execute_v433_adaptive_training  (~246 lines)
    # Source modules archived: adaptive_sac_core.py, online_learning_engine.py

    def _load_data_with_format_detection(self, data_path: str) -> pd.DataFrame:
        """
        データファイルを拡張子に応じて読み込む（CSV/Parquet対応）
        
        Args:
            data_path: データファイルパス
            
        Returns:
            pd.DataFrame: 読み込んだデータ
            
        Note:
            v459最適化: Parquet形式の特徴生成済みデータに対応
        """
        from pathlib import Path
        
        path = Path(data_path)
        
        # Parquet形式の場合
        if path.suffix.lower() == '.parquet':
            self.logger.info(f"📦 Loading pre-computed features from Parquet: {data_path}")
            try:
                df = read_parquet(path)
                self.logger.info(f"✅ Parquet loaded: {df.shape}, {len(df.columns)} features")
                return df
            except Exception as e:
                self.logger.error(f"❌ Parquet loading failed: {e}")
                raise
        
        # CSV形式の場合（従来の処理）
        else:
            self.logger.info(f"📄 Loading CSV data: {data_path}")
            df = DataLoader.load_csv_optimized(data_path)
            if "timestamp" in df.columns:
                df["timestamp"] = safe_to_datetime_series(df["timestamp"])
            self.logger.info(f"✅ CSV loaded: {df.shape}")
            return df
    
    def _validate_v454_columns(self, df: pd.DataFrame, data_path: str) -> None:
        """
        Validate that the DataFrame contains v454 specific columns.
        Logs a warning if missing.
        """
        v454_cols = ["vol_ema_14", "trend_dev_100", "noise_index"]
        missing = [col for col in v454_cols if col not in df.columns]

        if missing:
            self.logger.warning(
                f"⚠️  MISSING v454 FEATURES in {data_path}: {missing}. "
                "If you are training a v454 model, performance may be degraded. "
                "Please regenerate data using scripts/generate_v454_data.py."
            )
        else:
            self.logger.info(f"✅  v454 features validation passed for {data_path}")

    def _create_training_environment(self) -> object | None:
        """トレーニング環境を作成 (renamed from _create_v433_training_environment in 063#)"""
        try:
            # Lazy import to avoid heavy runtime dependency and mypy import-untyped noise
            import importlib

            try:
                # Use full-featured HeavyTradingEnv that respects action_type configuration
                mod = importlib.import_module(
                    "ztb.trading.environment.heavy_env.core"
                )
                HeavyTradingEnv = getattr(mod, "HeavyTradingEnv", None)
            except Exception as e:
                self.logger.debug("HeavyTradingEnv import failed: %s", e)
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

            # Load data from csv_path
            import pandas as pd

            data_path = env_config_obj.csv_path
            if not data_path:
                self.logger.error("No csv_path specified in environment config")
                return None

            try:
                data = self._load_data_with_format_detection(data_path)
                self.logger.info(f"Loaded data from {data_path}, shape: {data.shape}")
                # Validate v454 columns
                self._validate_v454_columns(data, str(data_path))
            except Exception as e:
                self.logger.error(f"Failed to load data from {data_path}: {e}")
                return None

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
            except Exception as e:
                self.logger.debug("env_config_dict preview failed: %s", e)

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
                data=data,
                config=env_config_obj,
            )

            # Initialize and attach Market Regime Classifier
            try:
                from ztb.analysis.regime.market_regime_classifier import MarketRegimeClassifier

                classifier = MarketRegimeClassifier()
                env.enable_market_regime_adaptation(regime_classifier=classifier)
                self.logger.info("Attached MarketRegimeClassifier to HeavyTradingEnv")
            except ImportError:
                self.logger.warning("Could not import MarketRegimeClassifier")
            except Exception as e:
                self.logger.warning(f"Failed to attach MarketRegimeClassifier: {e}")

            self.logger.info("Training environment created successfully")
            return env

        except Exception as e:
            self.logger.error(f"Failed to create V433 training environment: {e}")
            return None

    def run_multi_period_backtest_v433(
        self,
        model_path: str,
        data_path: str | None = None,
        window_sizes: list[int] | None = None,
        overlap_ratio: float = 0.5,
        output_path: str | None = None,
    ) -> dict[str, object]:
        """
        Run multi-period backtest analysis for SAC v445.3 model.

        This method integrates the multi_period_analysis_sac_v445_3.py functionality
        into the unified trainer framework.

        Args:
            model_path: Path to the trained model
            data_path: Path to custom data file (optional)
            window_sizes: list of window sizes in hours to test
            overlap_ratio: Overlap ratio between consecutive windows
            output_path: Path to save results (optional)

        Returns:
            dict containing backtest results
        """
        try:
            import json
            from pathlib import Path

            import pandas as pd

            from stable_baselines3 import PPO
            from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
            from ztb.trading.environment.utils.config import EnvironmentConfig
        except ImportError as e:
            self.logger.error(f"Failed to import required modules: {e}")
            return {"error": f"Import error: {e}"}

        self.logger.info("Starting multi-period backtest analysis...")

        # set default window sizes
        if window_sizes is None:
            window_sizes = [24]  # Default to 24 hours

        try:
            # Load model
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = PPO.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")

            # Load data
            if data_path and Path(data_path).exists():
                custom_df = self._load_data_with_format_detection(data_path)
                custom_df["timestamp"] = safe_to_datetime_series(custom_df["timestamp"])
                self.logger.info(
                    f"Loaded custom data from {data_path}, shape: {custom_df.shape}"
                )
                df = custom_df
            else:
                # Load default training data
                default_path = "data/btc_jpy_real_dataset.csv"
                if not Path(default_path).exists():
                    raise FileNotFoundError(
                        f"Default data file not found: {default_path}"
                    )
                df = DataLoader.load_csv_optimized(default_path)
                df["timestamp"] = safe_to_datetime_series(df["timestamp"])
                self.logger.info(f"Loaded default data, shape: {df.shape}")

            # Create environment config (simplified version)
            env_config_dict = {
                "initial_portfolio_value": 10000.0,
                "transaction_fee": 0.001,
                "use_continuous_actions": True,
                "adaptive_feature_selection": {"enabled": False},
                "target_feature_count": 140,
            }
            env_config = EnvironmentConfig(**env_config_dict)

            # Create environment
            env = HeavyTradingEnv(df=df, config=env_config, use_continuous_actions=True)
            self.logger.info(
                f"Environment created with observation space: {env.observation_space}"
            )

            # Verify observation space matches model expectations
            expected_obs_dim = model.observation_space.shape[0]
            actual_obs_dim = env.observation_space.shape[0]
            if actual_obs_dim != expected_obs_dim:
                self.logger.warning(
                    f"Observation space mismatch: model expects {expected_obs_dim}, environment provides {actual_obs_dim}"
                )

            results = {}

            for window_size in window_sizes:
                self.logger.info(f"Testing window size: {window_size} hours")

                # Identify market periods
                periods = self._identify_market_periods(df, window_size, overlap_ratio)

                window_results = []
                for period in periods:
                    period_result = self._test_period_with_model(
                        model,
                        env,
                        df,
                        period["start_idx"],
                        period["end_idx"],
                        period["period_name"],
                    )
                    window_results.append(period_result)

                # Analyze results by trend type
                trend_analysis = self._analyze_results_by_trend(window_results)

                results[f"{window_size}h_windows"] = {
                    "period_results": window_results,
                    "summary": trend_analysis,
                }

            # Save results if output path provided
            if output_path:
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                safe_json_dump(results, output_path, indent=2, default=str)
                self.logger.info(f"Results saved to {output_path}")

            self.logger.info("Multi-period backtest analysis completed")
            return results

        except Exception as e:
            self.logger.error(f"Multi-period backtest failed: {e}")
            return {"error": str(e)}

    def _identify_market_periods(
        self, df, window_size_hours: int = 24, overlap_ratio: float = 0.5
    ) -> list[dict[str, object]]:
        """Identify different market periods (uptrend, downtrend, sideways)."""
        periods = []

        # Calculate moving averages and trends
        df_copy = df.copy()
        df_copy["MA20"] = df_copy["close"].rolling(window=20).mean()
        df_copy["MA50"] = df_copy["close"].rolling(window=50).mean()
        df_copy["trend"] = (df_copy["MA20"] - df_copy["MA50"]) / df_copy["MA50"] * 100

        # Identify periods with configurable window size and overlap
        step_size = int(window_size_hours * (1 - overlap_ratio))
        window_size = window_size_hours

        for i in range(0, len(df_copy) - window_size, step_size):
            start_idx = i
            end_idx = min(i + window_size, len(df_copy))

            period_data = df_copy.iloc[start_idx:end_idx]
            start_price = period_data["close"].iloc[0]
            end_price = period_data["close"].iloc[-1]
            price_change = (end_price - start_price) / start_price * 100

            # Classify trend
            if price_change < -2:  # Downtrend
                trend_type = "downtrend"
            elif price_change > 2:  # Uptrend
                trend_type = "uptrend"
            else:  # Sideways
                trend_type = "sideways"

            periods.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_date": period_data["timestamp"].iloc[0],
                    "end_date": period_data["timestamp"].iloc[-1],
                    "start_price": start_price,
                    "end_price": end_price,
                    "price_change_pct": price_change,
                    "trend_type": trend_type,
                    "period_name": f"{trend_type}_{len(periods)+1}_{period_data['timestamp'].iloc[0].strftime('%Y%m%d')}_{window_size_hours}h",
                }
            )

        return periods

    def _test_period_with_model(
        self, model, env, df, start_idx: int, end_idx: int, period_name: str
    ) -> dict[str, object]:
        """Test the model on a specific period."""
        # Reset environment and advance to start_idx
        obs, _ = env.reset()

        # Advance to start_idx by taking dummy actions
        for step in range(start_idx):
            action = [0.0]  # Neutral action
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                self.logger.warning(
                    f"Environment terminated before reaching start_idx {start_idx}"
                )
                break

        # Test the period
        done = False
        total_steps = 0
        actions_taken = []
        rewards_received = []
        max_test_steps = min(end_idx - start_idx, 1000)

        initial_portfolio_value = env.portfolio_value

        while not done and total_steps < max_test_steps:
            try:
                action, _ = model.predict(obs, deterministic=True)
                actions_taken.append(action[0])

                obs, reward, terminated, truncated, info = env.step(action)
                rewards_received.append(reward)

                done = terminated or truncated
                total_steps += 1
            except IndexError:
                break

        # Calculate results
        final_portfolio_value = env.portfolio_value
        total_profit = final_portfolio_value - initial_portfolio_value
        total_return_pct = (
            (total_profit / initial_portfolio_value) * 100
            if initial_portfolio_value > 0
            else 0
        )

        # Action statistics
        sell_actions = sum(1 for a in actions_taken if a < -0.3)
        buy_actions = sum(1 for a in actions_taken if a > 0.3)
        hold_actions = len(actions_taken) - sell_actions - buy_actions

        return {
            "period_name": period_name,
            "start_date": str(df.iloc[start_idx]["timestamp"]),
            "end_date": str(df.iloc[min(end_idx, len(df) - 1)]["timestamp"]),
            "duration_hours": total_steps,
            "initial_portfolio": initial_portfolio_value,
            "final_portfolio": final_portfolio_value,
            "total_profit": total_profit,
            "total_return_pct": total_return_pct,
            "total_actions": len(actions_taken),
            "sell_actions": sell_actions,
            "buy_actions": buy_actions,
            "hold_actions": hold_actions,
            "sell_percentage": (sell_actions / len(actions_taken)) * 100
            if actions_taken
            else 0,
            "buy_percentage": (buy_actions / len(actions_taken)) * 100
            if actions_taken
            else 0,
            "hold_percentage": (hold_actions / len(actions_taken)) * 100
            if actions_taken
            else 0,
            "total_reward": sum(rewards_received),
            "average_reward": sum(rewards_received) / len(rewards_received)
            if rewards_received
            else 0,
        }

    def _analyze_results_by_trend(
        self, results: list[dict[str, object]]
    ) -> dict[str, object]:
        """Analyze results by trend type."""
        trend_groups = {}
        for result in results:
            trend_type = result.get("trend_type", "unknown")
            if trend_type not in trend_groups:
                trend_groups[trend_type] = []
            trend_groups[trend_type].append(result)

        analysis = {"overall": {}, "by_trend_type": {}}

        # Overall analysis
        if results:
            returns = [r["total_return_pct"] for r in results]
            analysis["overall"] = {
                "total_periods": len(results),
                "avg_return": sum(returns) / len(returns),
                "win_rate": sum(1 for r in results if r["total_return_pct"] > 0)
                / len(results)
                * 100,
                "sharpe_ratio": sharpe_ratio(returns, period_per_year=1),
            }

        for trend_type, trend_results in trend_groups.items():
            if trend_results:
                returns = [r["total_return_pct"] for r in trend_results]
                analysis["by_trend_type"][trend_type] = {
                    "count": len(trend_results),
                    "avg_return": sum(returns) / len(returns),
                    "win_rate": sum(
                        1 for r in trend_results if r["total_return_pct"] > 0
                    )
                    / len(trend_results)
                    * 100,
                    "sharpe_ratio": sharpe_ratio(returns, period_per_year=1),
                }

        return analysis

    def run_multi_period_backtest(
        self,
        periods: list[dict[str, object]],
        model_path: str | None = None,
        config_path: str | None = None,
    ) -> dict[str, object]:
        """
        Run multi-period backtest analysis.

        Args:
            periods: list of period definitions with start/end dates
            model_path: Path to trained model (optional)
            config_path: Path to config file (optional)

        Returns:
            Multi-period backtest results
        """
        results = {
            "period_results": [],
            "overall_metrics": {},
            "regime_performance": {},
            "recommendations": [],
        }

        try:
            # Load model if path provided
            model = None
            if model_path:
                try:
                    from stable_baselines3 import PPO

                    model = PPO.load(model_path)
                    self.logger.info(f"Loaded model from {model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load model: {e}")
                    return results

            # Create environment for testing
            env = self._create_backtest_environment(config_path)
            if env is None:
                self.logger.error("Failed to create backtest environment")
                return results

            # Load data
            df = self._load_backtest_data(config_path)
            if df is None or df.empty:
                self.logger.error("Failed to load backtest data")
                return results

            # Run backtest for each period
            for period in periods:
                period_result = self._run_single_period_backtest(model, env, df, period)
                results["period_results"].append(period_result)

            # Calculate overall metrics
            results["overall_metrics"] = self._calculate_overall_backtest_metrics(
                results["period_results"]
            )

            # Analyze regime performance
            results["regime_performance"] = self._analyze_backtest_regime_performance(
                results["period_results"]
            )

            # Generate recommendations
            results["recommendations"] = self._generate_backtest_recommendations(
                results
            )

        except Exception as e:
            self.logger.error(f"Multi-period backtest failed: {e}")
            results["error"] = str(e)

        return results

    def _create_backtest_environment(self, config_path: str | None = None):
        """Create environment for backtesting."""
        try:
            # Use existing environment creation logic
            if hasattr(self, "_create_training_environment"):
                return self._create_training_environment()
            else:
                # Fallback environment creation
                from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
                from ztb.trading.environment.utils.config import EnvironmentConfig

                env_config = EnvironmentConfig(
                    initial_portfolio_value=100000,
                    transaction_cost=0.001,
                    max_position_size=1.0,
                )
                return HeavyTradingEnv(env_config)
        except Exception as e:
            self.logger.error(f"Failed to create backtest environment: {e}")
            return None

    def _load_backtest_data(self, config_path: str | None = None):
        """Load data for backtesting."""
        try:
            # Use config to determine data path
            data_config = self.config.get("training", {}).get("data_config", {})
            csv_path = data_config.get("data_path", "data/btc_jpy_real_dataset.csv")

            import pandas as pd

            df = DataLoader.load_csv_optimized(csv_path)
            df["timestamp"] = safe_to_datetime_series(df["timestamp"])
            return df
        except Exception as e:
            self.logger.error(f"Failed to load backtest data: {e}")
            return None

    def _run_single_period_backtest(
        self, model, env, df, period: dict[str, object]
    ) -> dict[str, object]:
        """Run backtest for a single period."""
        try:
            start_date = period.get("start_date")
            end_date = period.get("end_date")
            period_name = period.get("name", "unknown")

            # Find indices for the period
            start_idx = 0
            end_idx = len(df) - 1

            if start_date:
                # Use pd.Timestamp directly to avoid C extension
                start_ts = pd.Timestamp(start_date)
                start_mask = df["timestamp"] >= start_ts
                start_idx = start_mask.idxmax() if start_mask.any() else 0

            if end_date:
                end_ts = pd.Timestamp(end_date)
                end_mask = df["timestamp"] <= end_ts
                end_idx = end_mask.idxmax() if end_mask.any() else len(df) - 1

            # Run the period test
            return self._test_period_with_model(
                model, env, df, start_idx, end_idx, period_name
            )

        except Exception as e:
            self.logger.error(f"Single period backtest failed: {e}")
            return {
                "period_name": period.get("name", "unknown"),
                "error": str(e),
                "total_return_pct": 0.0,
            }

    def _calculate_overall_backtest_metrics(
        self, period_results: list[dict[str, object]]
    ) -> dict[str, object]:
        """Calculate overall metrics from period results."""
        if not period_results:
            return {}

        valid_results = [r for r in period_results if "error" not in r]

        if not valid_results:
            return {"error": "No valid period results"}

        returns = [r.get("total_return_pct", 0) for r in valid_results]
        total_trades = sum(r.get("total_actions", 0) for r in valid_results)

        return {
            "total_periods": len(valid_results),
            "average_return": sum(returns) / len(returns) if returns else 0.0,
            "total_trades": total_trades,
            "win_rate": calculate_win_rate(returns) * 100,
        }

    def _analyze_backtest_regime_performance(
        self, period_results: list[dict[str, object]]
    ) -> dict[str, object]:
        """Analyze performance by market regime."""
        # Placeholder - would integrate with regime detection
        return {
            "bull_market_performance": {"average_return": 0.0, "win_rate": 0.0},
            "bear_market_performance": {"average_return": 0.0, "win_rate": 0.0},
            "sideways_performance": {"average_return": 0.0, "win_rate": 0.0},
        }

    def _generate_backtest_recommendations(self, results: dict[str, object]) -> list[str]:
        """Generate recommendations based on backtest results."""
        recommendations = []
        overall = results.get("overall_metrics", {})

        if overall.get("win_rate", 0) > 60:
            recommendations.append(
                "Strong overall performance - maintain current strategy"
            )
        elif overall.get("win_rate", 0) < 40:
            recommendations.append(
                "Performance needs improvement - consider strategy adjustments"
            )

        return recommendations
