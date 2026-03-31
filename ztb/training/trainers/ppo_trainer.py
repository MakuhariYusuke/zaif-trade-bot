#!/usr/bin/env python3
"""
PPO Algorithm Trainer for Unified Training.

Handles PPO-specific training logic including:
- Standard PPO training
- SELL bias mitigation training
- Memory optimization and CUDA setup
"""

import gc
import os
from pathlib import Path
from typing import Any

from ztb.training.constants import SAVE_INTERVAL
from ztb.training.unified_trainer.components.config_manager import TrainingConfigManager
from ztb.training.unified_trainer.reporting import TrainingReporter
from ztb.training.unified_trainer.components.ui_manager import TrainingUIManager
from ztb.training.unified_trainer.ensemble_mixin import EnsembleMixin
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PPOAlgorithmTrainer(EnsembleMixin):
    """
    Handles PPO algorithm training with various configurations.
    """

    def __init__(
        self, config_manager: TrainingConfigManager, progress_bar_enabled: bool = False
    ) -> None:
        """
        Initialize PPO trainer.

        Args:
            config_manager: ConfigManager instance
            progress_bar_enabled: Whether progress bar is enabled
        """
        super().__init__()
        self.config_manager = config_manager
        self.progress_bar_enabled = progress_bar_enabled
        self.logger = get_logger(__name__)
        self.ui_manager = TrainingUIManager(self.logger)
        self.reporter = TrainingReporter(self.logger)

    def _setup_memory_optimization(self, unified_config: dict[str, Any]) -> None:
        """
        set up memory optimization environment variables.

        Args:
            unified_config: Unified configuration dict
        """
        # Only set CUDA-related env vars if CUDA is available and requested
        if unified_config.get("enable_cuda_optimizations", True):
            os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Memory optimization: set garbage collection thresholds
        if unified_config.get("aggressive_memory_management", False):
            gc.set_threshold(700, 10, 10)  # More aggressive GC

    def _get_lagrange_params(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract Lagrange constraint parameters.

        Args:
            config: Configuration dict

        Returns:
            Lagrange parameters dict or None
        """
        from ztb.training.config.lagrange_defaults import LAGRANGE_DEFAULTS

        # 🔧 FIX: lagrange_constraintキーもチェック（v392等で使用）
        lagrange_config = config.get("lagrange_constraint", {})

        def get_lagrange_param(key: str, default: Any = None) -> Any:
            """トップレベル（lagrange_プレフィックス）とlagrange_constraintの両方をチェック"""
            # lagrange_プレフィックス付きキーを優先、次にlagrange_constraint内、最後にデフォルト
            prefixed_key = f"lagrange_{key}"
            return config.get(
                prefixed_key,
                lagrange_config.get(key, LAGRANGE_DEFAULTS.get(key, default)),
            )

        # enable_lagrangeは特別扱い（プレフィックスなしとlagrange_constraint.enabledの両方をチェック）
        enable_lagrange = config.get(
            "enable_lagrange", lagrange_config.get("enabled", True)
        )

        if not enable_lagrange:
            return None

        return {
            "r_target": get_lagrange_param("r_target"),
            "tolerance": get_lagrange_param("tolerance"),
            "eta": get_lagrange_param("eta"),
            "lambda_max": get_lagrange_param("lambda_max"),
            "warmup_steps": get_lagrange_param("warmup_steps"),
        }

    def _create_ppo_trainer(
        self, unified_config: dict[str, Any], enable_sell_mitigation: bool = False
    ) -> Any:
        """
        Create appropriate PPO trainer instance.

        Args:
            unified_config: Unified configuration
            enable_sell_mitigation: Whether to use SELL bias mitigation

        Returns:
            Trainer instance
        """
        cfg = unified_config if isinstance(unified_config, dict) else {}
        checkpoint_interval = cfg.get("checkpoint_interval", SAVE_INTERVAL)

        if enable_sell_mitigation:
            # Import SELLMitigationParams
            from ztb.training.config.trainer_params import SELLMitigationParams

            # Build mitigation params with unified config
            lagrange_params = self._get_lagrange_params(unified_config)

            mitigation_params = SELLMitigationParams(
                data_path=unified_config.get("data_path"),
                config=unified_config,
                checkpoint_dir=unified_config.get("checkpoint_dir", "checkpoints"),
                checkpoint_interval=checkpoint_interval,
                progress_bar=self.progress_bar_enabled,
                enable_lagrange=unified_config.get("enable_lagrange", True),
                enable_probes=unified_config.get("enable_probes", False),
                enable_weights=unified_config.get("enable_weights", False),
                enable_pan=unified_config.get("enable_pan", True),
                enable_target_entropy=unified_config.get(
                    "enable_target_entropy", False
                ),
                enable_stratified_sampling=unified_config.get(
                    "enable_stratified_sampling", False
                ),
                allow_reverse=unified_config.get("allow_reverse", False),
                probe_csv_path=unified_config.get("probe_csv_path"),
                lagrange_params=lagrange_params,
            )

            from ztb.training.sell_mitigation_ppo_trainer import (
                SELLBiasMitigationPPOTrainer,
            )

            return SELLBiasMitigationPPOTrainer(mitigation_params)
        else:
            # Import TrainerParams for standard PPO
            from ztb.training.config.trainer_params import TrainerParams

            trainer_params = TrainerParams(
                data_path=unified_config.get("data_path"),
                config=unified_config,
                checkpoint_dir=unified_config.get("checkpoint_dir", "checkpoints"),
                checkpoint_interval=checkpoint_interval,
                progress_bar=self.progress_bar_enabled,
            )

            from ztb.training.core.ppo_trainer import PPOTrainerAutoHalt

            return PPOTrainerAutoHalt(trainer_params)

    def _save_model_and_schema(
        self, model: Any, unified_config: dict[str, Any]
    ) -> None:
        """
        Save trained model and schema.

        Args:
            model: Trained model
            unified_config: Unified configuration
        """
        if model is None:
            return

        # Save model
        model_dir = Path(unified_config.get("model_dir", "models"))
        model_dir.mkdir(exist_ok=True)
        session_id = unified_config.get("session_id", "ppo_session")
        model_path = model_dir / f"{session_id}.zip"

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

        # Save model schema
        self._save_model_schema(session_id, model_dir)

    def _save_model_schema(
        self, session_id: str, model_dir: Path, df: Any | None = None
    ) -> None:
        """
        Save model schema using FeatureSchemaManager.

        Args:
            session_id: Model session ID
            model_dir: Directory where model is saved
            df: Optional DataFrame
        """
        try:
            import numpy as np
            import pandas as pd

            from ztb.training.core.feature_schema_manager import FeatureSchemaManager
            from ztb.io.data_loader import DataLoader

            # Load DataFrame if not provided
            if df is None:
                data_path = self.config_manager.config.get("data_path")
                if not data_path:
                    self.logger.warning("No data_path in config, skipping schema save")
                    return
                df = DataLoader.load_csv_optimized(data_path)

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
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
            ]

            # Calculate scaler data
            feature_data = df[feature_columns].values
            scaler_data = {
                "mean": np.mean(feature_data, axis=0),
                "std": np.std(feature_data, axis=0),
            }

            # Save schema using FeatureSchemaManager
            schema_manager = FeatureSchemaManager(
                model_name=session_id, models_dir=Path(model_dir)
            )

            schema_hash = schema_manager.save_schema(
                features=feature_columns,
                config=self.config_manager.config,
                scaler_data=scaler_data,
            )

            self.logger.info(
                f"✅ Model schema saved: {len(feature_columns)} features, "
                f"hash: {schema_hash[:16]}..."
            )
            self.logger.info(f"   Schema directory: {model_dir}/schemas/{session_id}/")

        except Exception as e:
            # Non-fatal: Log warning but don't fail training
            self.logger.warning(
                f"Failed to save model schema (non-fatal): {e}", exc_info=True
            )

    def train(self, unified_config: dict[str, Any]) -> Any:
        """
        Execute PPO training.

        Args:
            unified_config: Unified configuration

        Returns:
            Trained model
        """
        # Initialize ensemble if enabled
        self.initialize_ensemble(unified_config)

        # set up memory optimization
        self._setup_memory_optimization(unified_config)

        # Check if SELL bias mitigation is enabled
        enable_sell_mitigation = unified_config.get("enable_sell_mitigation", False)

        if enable_sell_mitigation:
            self.logger.info("SELL bias mitigation enabled - using enhanced trainer")
            try:
                from ztb.training.experiments.sell_mitigation_ppo_trainer import (  # noqa: F401
                    SELLBiasMitigationPPOTrainer,
                )
            except ImportError as e:
                self.logger.warning(
                    f"SELL mitigation trainer not available: {e}. Falling back to standard PPO."
                )
                enable_sell_mitigation = False

        # Create trainer
        trainer = self._create_ppo_trainer(unified_config, enable_sell_mitigation)

        # Memory optimization: Periodic cleanup during training
        if unified_config.get("aggressive_memory_management", False):
            import atexit

            atexit.register(gc.collect)

        try:
            # Log training start with structured info
            self.logger.info(
                "Starting PPO training",
                extra={
                    "algorithm": "ppo",
                    "session_id": unified_config.get("session_id", "ppo_session"),
                    "total_timesteps": unified_config["training"]["total_timesteps"],
                    "enable_sell_mitigation": enable_sell_mitigation,
                    "memory_optimization": self.config_manager.get_memory_optimization_config(),
                },
            )

            model = trainer.train(
                session_id=unified_config.get("session_id", "ppo_session")
            )

            # Generate ensemble report if enabled
            if self.ensemble_enabled:
                try:
                    ensemble_report_path = self.generate_ensemble_report(self.reporter, self.ui_manager)
                    if ensemble_report_path:
                        self.logger.info(
                            f"Ensemble report generated: {ensemble_report_path}"
                        )
                        self.print_ensemble_status(self.ui_manager)
                except Exception as e:
                    self.logger.error(f"Ensemble report generation failed: {e}")

            # Log training completion
            self.logger.info(
                "PPO training completed successfully",
                extra={
                    "session_id": unified_config.get("session_id", "ppo_session"),
                    "model_saved": model is not None,
                },
            )

        except Exception as e:
            self.logger.error(
                "PPO training failed",
                extra={
                    "session_id": unified_config.get("session_id", "ppo_session"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise
        finally:
            # Aggressive memory cleanup
            if unified_config.get("aggressive_memory_management", False):
                gc.collect()

        # Save model and schema
        self._save_model_and_schema(model, unified_config)

        return model
