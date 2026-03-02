#!/usr/bin/env python3
"""
Base trainer classes for unified training system.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from ztb.training.constants import DEFAULT_CHECK_FREQ, DEFAULT_MAX_MEMORY_GB
from ztb.training.gradient_accumulation import GradientAccumulator
from ztb.training.optimization.early_stopping import EarlyStopping
from ztb.training.optimization.memory_efficient_loader import MemoryEfficientLoader
from ztb.training.unified_trainer.base.lr_scheduler import DynamicLRScheduler
from ztb.utils.errors import ZTBError
from ztb.utils.logging_utils import get_logger

from .callbacks import TrainingProgressCallback
from .metrics_mixin import MetricsCollectionMixin

class TrainingError(ZTBError):
    """Base exception for training-related errors."""

    pass

class ConfigurationError(TrainingError):
    """Raised when training configuration is invalid."""

    pass

class DataError(TrainingError):
    """Raised when training data is invalid or missing."""

    pass

class ModelError(TrainingError):
    """Raised when model operations fail."""

    pass

class BaseAlgorithmTrainer(ABC, MetricsCollectionMixin):
    """Base class for algorithm-specific trainers with common training optimizations."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        MetricsCollectionMixin.__init__(self)
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)

        # Common training optimization components
        self.gradient_accumulation_steps: int = 1
        self.system_optimizer: Any | None = None
        self.gradient_accumulator: GradientAccumulator | None = None
        self.lr_scheduler: DynamicLRScheduler | None = None
        self.early_stopping: EarlyStopping | None = None
        self.memory_loader: MemoryEfficientLoader | None = None

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        統合的なデータ読み込みメソッド（CSV/Parquet自動検出）
        
        Phase 4 Week 1 Day 2-3実装: 
        - すべてのアルゴリズム（SAC/PPO/DQN/A2C）で共通経路を使用
        - 事前計算特徴の検出とスキップ処理
        - 特徴検出ロジックの厳密化（42番再レビュー対応）
        
        Args:
            data_path: データファイルのパス（CSV or Parquet）
            
        Returns:
            読み込まれたDataFrame
        """
        from ztb.io.data_loader import DataLoader

        path = Path(data_path)
        
        if path.suffix == '.parquet':
            self.logger.info(f"Parquet形式検出: {data_path}")
            df = pd.read_parquet(path)
            self.logger.info(f"Parquet読み込み完了: {len(df)}行、{len(df.columns)}列")
            
            # 事前計算特徴の厳密な検出（42番再レビュー対応）
            if self._has_precomputed_features(df):
                self.logger.info("事前計算特徴を検出、特徴生成をスキップします")
                self._apply_feature_skip()
            
            return df
        else:
            self.logger.info(f"CSV形式として読み込み: {data_path}")
            df = DataLoader.load_csv_strict(data_path)
            self.logger.info(f"CSV読み込み完了: {len(df)}行、{len(df.columns)}列")
            return df
    
    def _has_precomputed_features(self, df: pd.DataFrame) -> bool:
        """
        事前計算特徴の存在を厳密に検証（42番再レビュー対応）
        
        Phase 4実装: feature_cols >= 5 だけでは誤検出リスクがあるため、
        明示的な特徴列の存在確認を追加
        
        Args:
            df: 検証対象のDataFrame
            
        Returns:
            True: 事前計算特徴が存在
            False: OHLCV のみ
        """
        # 1. 必須列の存在確認
        required_ohlcv = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        if not required_ohlcv.issubset(df.columns):
            return False
        
        # 2. OHLCV以外の列をカウント
        feature_cols = [c for c in df.columns if c not in required_ohlcv]
        
        # 3. 想定している8特徴（またはそのサブセット）の存在確認
        expected_features = {
            'rsi', 'macd', 'bb_width', 'volatility', 
            'momentum', 'volume_ma_ratio', 'atr', 'obv'
        }
        detected_features = set(feature_cols) & expected_features
        
        # 4. 特徴数 >= 5 かつ 既知特徴 >= 3 で事前計算と判定
        has_features = len(feature_cols) >= 5 and len(detected_features) >= 3
        
        if has_features:
            self.logger.info(
                f"事前計算特徴検出: {len(feature_cols)}列 "
                f"({len(detected_features)}個の既知特徴: {detected_features})"
            )
        else:
            self.logger.debug(
                f"事前計算特徴なし: {len(feature_cols)}列、"
                f"既知特徴{len(detected_features)}個"
            )
        
        return has_features
    
    def _apply_feature_skip(self) -> None:
        """
        事前計算特徴検出時の設定適用
        
        feature_set を "minimal" に設定して特徴生成をスキップ
        """
        config_dict = self.config if isinstance(self.config, dict) else {}
        
        # training.features.feature_set に設定
        if "training" not in config_dict:
            config_dict["training"] = {}
        if "features" not in config_dict["training"]:
            config_dict["training"]["features"] = {}
        
        config_dict["training"]["features"]["feature_set"] = "minimal"
        self.logger.info("feature_set='minimal' を設定しました")

    def safe_training_operation(
        self, operation: Callable, operation_name: str, **kwargs
    ) -> Any:
        """Execute a training operation safely with standardized error handling.

        Args:
            operation: The operation to execute
            operation_name: Descriptive name for logging
            **kwargs: Additional context for error reporting

        Returns:
            Result of the operation

        Raises:
            TrainingError: If operation fails
        """
        try:
            self.logger.debug(f"🔧 Executing: {operation_name}")
            result = operation()
            self.logger.debug(f"✅ Completed: {operation_name}")
            return result
        except KeyboardInterrupt:
            self.logger.warning(f"⚠️  Operation interrupted by user: {operation_name}")
            raise
        except ConfigurationError:
            self.logger.error(f"❌ Configuration error in {operation_name}: {kwargs}")
            raise
        except DataError:
            self.logger.error(f"❌ Data error in {operation_name}: {kwargs}")
            raise
        except ModelError:
            self.logger.error(f"❌ Model error in {operation_name}: {kwargs}")
            raise
        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error in {operation_name}: {e}", exc_info=True
            )
            raise TrainingError(f"Operation '{operation_name}' failed: {e}") from e

    def log_structured_event(
        self, event_type: str, phase: str, details: dict[str, Any], level: str = "info"
    ) -> None:
        """Log structured training events for better analysis.

        Args:
            event_type: Type of event (start, progress, completion, error)
            phase: Training phase (initialization, training, evaluation, cleanup)
            details: Structured details about the event
            level: Log level (debug, info, warning, error)
        """
        message = f"[{event_type.upper()}] {phase}: {details}"

        # Add structured context for log analysis
        structured_details = {
            "event_type": event_type,
            "phase": phase,
            "timestamp": time.time(),
            **details,
        }

        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message, extra={"structured_data": structured_details})

    def initialize_training_optimizations(self) -> None:
        """Initialize common training optimization components."""
        optimization_config = self.config.get("training", {}).get("optimization", {})

        # Initialize gradient accumulator
        if self.gradient_accumulation_steps > 1:
            self.gradient_accumulator = GradientAccumulator(
                accumulation_steps=self.gradient_accumulation_steps
            )
            self.logger.info(
                f"Initialized gradient accumulator with {self.gradient_accumulation_steps} steps"
            )

        # Initialize dynamic LR scheduler
        if optimization_config.get("use_dynamic_lr", False):
            lr_patience = optimization_config.get("lr_patience", 10)
            lr_factor = optimization_config.get("lr_factor", 0.5)
            min_lr = optimization_config.get("min_lr", 1e-6)
            self.lr_scheduler = DynamicLRScheduler(
                optimizer=None,  # Will be set after model creation
                patience=lr_patience,
                factor=lr_factor,
                min_lr=min_lr,
            )
            self.logger.info(
                f"Initialized dynamic LR scheduler (patience={lr_patience}, factor={lr_factor})"
            )

        # Initialize early stopping
        if optimization_config.get("use_early_stopping", False):
            es_patience = optimization_config.get("early_stopping_patience", 20)
            es_min_delta = optimization_config.get("early_stopping_min_delta", 0.001)
            self.early_stopping = EarlyStopping(
                patience=es_patience, min_delta=es_min_delta, restore_best_weights=True
            )
            self.logger.info(
                f"Initialized early stopping (patience={es_patience}, min_delta={es_min_delta})"
            )

        # Initialize memory efficient loader
        if optimization_config.get("use_memory_efficient_loading", False):
            batch_size = optimization_config.get("memory_batch_size", 64)
            max_memory_gb = optimization_config.get(
                "max_memory_gb", DEFAULT_MAX_MEMORY_GB
            )
            self.memory_loader = MemoryEfficientLoader(
                batch_size=batch_size, max_memory_gb=max_memory_gb
            )
            self.logger.info(
                f"Initialized memory efficient loader (batch_size={batch_size}, max_memory={max_memory_gb}GB)"
            )

    def create_training_callback(
        self, check_freq: int = DEFAULT_CHECK_FREQ
    ) -> TrainingProgressCallback:
        """Create a standardized training progress callback.

        Args:
            check_freq: Frequency of progress checks

        Returns:
            Configured TrainingProgressCallback instance
        """
        callback = TrainingProgressCallback(
            check_freq=check_freq,
            system_optimizer=self.system_optimizer,
            metrics_csv_writer=self.metrics_csv_writer,
            lr_scheduler=self.lr_scheduler,
            early_stopping=self.early_stopping,
            trainer_ref=self,
        )
        return callback

    def setup_training_environment(self) -> None:
        """Setup training environment (memory management, etc.)."""
        import gc

        gc.disable()  # Disable automatic GC during training for performance
        self.logger.debug("Training environment setup completed")

    def cleanup_training_environment(self) -> None:
        """Cleanup training environment."""
        import gc

        gc.enable()  # Re-enable GC after training
        gc.collect()  # Force garbage collection
        self.logger.debug("Training environment cleanup completed")

    def save_model(self, model: Any, model_name: str, extension: str = ".zip") -> str:
        """Save model with standardized naming and error handling.

        Args:
            model: Model instance to save
            model_name: Base name for the model
            extension: File extension (.zip for SB3 models, .pth for PyTorch models)

        Returns:
            Path where model was saved
        """
        model_path = f"models/{model_name}{extension}"
        os.makedirs("models", exist_ok=True)

        self.logger.info(f"💾 Saving model to {model_path}")
        try:
            # Use central save helper to ensure consistent error handling
            from ztb.utils.training_utils import save_model as _save_model

            saved_ok = _save_model(model, model_path)

            if not saved_ok:
                self.logger.error(
                    f"Model save helper reported failure for {model_path}"
                )

            # 🔧 Fix: Explicitly save VecNormalize stats if present
            # SB3 does not automatically save VecNormalize stats in the model zip
            if hasattr(model, "get_env"):
                try:
                    import zipfile

                    from stable_baselines3.common.vec_env import VecNormalize

                    env = model.get_env()
                    current_env = env
                    vec_norm_env = None

                    # Traverse wrappers to find VecNormalize
                    # Limit depth to prevent infinite loops
                    for _ in range(10):
                        if current_env is None:
                            break
                        if isinstance(current_env, VecNormalize):
                            vec_norm_env = current_env
                            break

                        if hasattr(current_env, "venv"):
                            current_env = current_env.venv
                        elif hasattr(current_env, "env"):
                            current_env = current_env.env
                        elif hasattr(current_env, "envs") and len(current_env.envs) > 0:
                            current_env = current_env.envs[0]
                        else:
                            break

                    # After traversing wrappers, if we found VecNormalize save its stats
                    if vec_norm_env is not None:
                        self.logger.info(
                            "Found VecNormalize, appending stats to zip..."
                        )
                        temp_stats_path = f"models/vec_normalize_{model_name}.pkl"
                        vec_norm_env.save(temp_stats_path)

                        with zipfile.ZipFile(model_path, "a") as zipf:
                            zipf.write(temp_stats_path, arcname="vec_normalize.pkl")

                        if os.path.exists(temp_stats_path):
                            os.remove(temp_stats_path)
                        self.logger.info("✅ VecNormalize stats appended to model zip")
                except Exception as e:
                    self.logger.warning(f"Failed to save VecNormalize stats: {e}")

            elif hasattr(model, "save_checkpoint"):
                model.save_checkpoint(model_path)
            else:
                raise AttributeError(
                    "Model does not have save or save_checkpoint method"
                )
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

        return model_path

    def collect_training_stats(
        self,
        training_time: float,
        total_timesteps: int,
        model_path: str,
        **additional_stats,
    ) -> dict[str, Any]:
        """Collect standardized training statistics.

        Args:
            training_time: Total training time in seconds
            total_timesteps: Total number of training steps
            model_path: Path where model was saved
            **additional_stats: Additional algorithm-specific statistics

        Returns:
            Dictionary of training statistics
        """
        stats = {
            "total_timesteps": total_timesteps,
            "training_time": training_time,
            "model_path": model_path,
            "status": "completed",
        }
        stats.update(additional_stats)
        return stats

    def log_training_start(
        self,
        algorithm_name: str,
        total_timesteps: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Log standardized training start message.

        Args:
            algorithm_name: Name of the algorithm (SAC, PPO, etc.)
            total_timesteps: Total number of training timesteps (optional)
            epochs: Number of epochs for self-supervised learning (optional)
            batch_size: Batch size for self-supervised learning (optional)
        """
        details = {"algorithm": algorithm_name}
        if total_timesteps is not None:
            details["total_timesteps"] = total_timesteps
        if epochs is not None:
            details["epochs"] = epochs
        if batch_size is not None:
            details["batch_size"] = batch_size

        self.log_structured_event("start", "initialization", details)
        self.logger.info(f"🚀 Starting {algorithm_name} training...")

        if total_timesteps is not None:
            self.logger.info(f"🎯 Training for {total_timesteps:,} timesteps")
        elif epochs is not None and batch_size is not None:
            self.logger.info(
                f"🎯 Training for {epochs} epochs with batch size {batch_size}"
            )

    def log_training_progress(
        self, message: str = "Training started...", phase: str = "training"
    ) -> None:
        """Log standardized training progress message.

        Args:
            message: Progress message to log
            phase: Current training phase
        """
        self.log_structured_event("progress", phase, {"message": message})
        self.logger.info(f"🏃 {message}")

    def log_training_completion(
        self, training_time: float, stats: dict[str, Any] | None = None
    ) -> None:
        """Log standardized training completion message.

        Args:
            training_time: Total training time in seconds
            stats: Optional training statistics to log
        """
        details = {
            "training_time_seconds": training_time,
            "training_time_formatted": f"{training_time:.1f}s",
        }
        if stats:
            details["statistics"] = stats

        self.log_structured_event("completion", "finalization", details)
        self.logger.info(f"✅ Training completed in {training_time:.1f} seconds")

        if stats:
            self.logger.info(f"📈 Training stats: {stats}")

    def execute_training_pipeline(
        self, algorithm_name: str, training_function: Callable, **context
    ) -> bool:
        """Execute a complete training pipeline with standardized error handling and logging.

        Args:
            algorithm_name: Name of the algorithm
            training_function: Function that performs the actual training
            **context: Additional context for error reporting

        Returns:
            True if training completed successfully
        """
        try:
            # Phase 1: Initialization
            self.log_structured_event(
                "start", "pipeline", {"algorithm": algorithm_name}
            )

            # Create training callback
            callback = self.create_training_callback()
            context["callback"] = callback
            context["start_time"] = time.time()

            # Execute training function with error handling
            result = self.safe_training_operation(
                lambda: training_function(
                    total_timesteps=self.config.get("training", {}).get(
                        "total_timesteps", 100000
                    ),
                    callback=callback,
                    start_time=context["start_time"],
                ),
                f"{algorithm_name} training pipeline",
                algorithm=algorithm_name,
                **context,
            )

            # Phase 2: Completion
            self.log_structured_event(
                "completion", "pipeline", {"algorithm": algorithm_name, "success": True}
            )

            return result

        except KeyboardInterrupt as kb_int:
            # Only catch true KeyboardInterrupt - let other exceptions fall through
            self.logger.warning(f"⚠️  {algorithm_name} training interrupted by user (KeyboardInterrupt)")
            self.log_structured_event(
                "interruption",
                "pipeline",
                {"algorithm": algorithm_name, "reason": "user_interrupt", "exception": str(kb_int)},
            )
            raise  # Re-raise to prevent masking real issues

        except TrainingError as e:
            self.logger.error(f"❌ {algorithm_name} training failed: {e}")
            self.log_structured_event(
                "error", "pipeline", {"algorithm": algorithm_name, "error": str(e)}
            )
            return False

        except Exception as e:
            self.logger.error(
                f"❌ Unexpected error in {algorithm_name} training: {e}", exc_info=True
            )
            self.log_structured_event(
                "error",
                "pipeline",
                {"algorithm": algorithm_name, "error": str(e), "unexpected": True},
            )
            return False

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration for this algorithm."""
        pass

    @abstractmethod
    def train(self, total_timesteps: int | None = None) -> bool:
        """Execute training for this algorithm.

        Args:
            total_timesteps: Total number of timesteps to train for
        """
        pass

    @abstractmethod
    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        pass

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger | None) -> None:
        """set logger instance."""
        self._logger = value or get_logger(self.__class__.__name__)
