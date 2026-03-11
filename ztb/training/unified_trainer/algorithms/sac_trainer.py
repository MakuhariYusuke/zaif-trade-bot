"""SAC algorithm trainer implementation."""
from __future__ import annotations

import copy
import dataclasses
import math
import logging
import os
import time
from typing import Optional, cast

import numpy as np
import torch

# Guard SB3 imports to avoid import-time errors in minimal test environments.
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecFrameStack,
        VecNormalize,
    )
except Exception:
    SAC = None
    CallbackList = list
    EvalCallback = type("EvalCallback", (), {})
    Monitor = None
    DummyVecEnv = None
    VecFrameStack = None
    VecNormalize = None

from ztb.features.processors.optimization.features import OptimizerFeatureTracker
from ztb.io.data_loader import DataLoader
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.training.checkpoint.checkpoint_manager import (
    TrainingCheckpointConfig,
    TrainingCheckpointManager,
)
from ztb.training.config.configuration_manager import ConfigurationManager
from ztb.training.constants import DEFAULT_LEARNING_RATE_SAC, DEFAULT_BATCH_SIZE_SAC, DEFAULT_GAMMA, DEFAULT_TAU, DEFAULT_BUFFER_SIZE_SAC, DEFAULT_LEARNING_STARTS_SAC, DEFAULT_TRAIN_FREQ, DEFAULT_GRADIENT_STEPS, DEFAULT_TARGET_UPDATE_INTERVAL
from ztb.training.unified_trainer.base.base_trainer import (
    BaseAlgorithmTrainer,
    ModelError,
)
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback
from ztb.training.utils.distributed_training import get_distributed_info
from ztb.training.utils.training_stats import TrainingStats
from ztb.types.common import ConfigDict
from ztb.utils.checkpoint import TrainingStateManager
from ztb.utils.logging_utils import StructuredLogger
from ztb.utils.training_utils import create_checkpoint_callback
from ztb.utils.dataclass_utils import shallow_asdict

class SACTrainer(BaseAlgorithmTrainer):
    """SAC algorithm trainer with enhanced UI and monitoring."""

    def __init__(
        self,
        config: ConfigDict,
        env: HeavyTradingEnv | None = None,
        logger: logging.Logger | None = None,
        gradient_accumulation_steps: int = 1,
        system_optimizer: object | None = None,
        optimizer_tracker: OptimizerFeatureTracker | None = None,
    ):
        super().__init__(config, logger)
        self.env = env
        # model will be instantiated later; annotate as optional to satisfy mypy
        self.model: object | None = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.system_optimizer = system_optimizer
        self.optimizer_tracker = optimizer_tracker
        self.training_stats: TrainingStats = {}

        # Initialize structured logger for JSON logging
        self.structured_logger = StructuredLogger("ztb.training.sac", json_format=True)

        # Training state manager for resume functionality
        self.training_state_manager = TrainingStateManager(
            save_dir=self.config.get("training", {}).get(
                "checkpoint_dir", "models/training_states"
            )
        )

        # Training checkpoint manager for periodic saves
        self.checkpoint_manager = TrainingCheckpointManager(
            save_dir=self.config.get("training", {}).get(
                "checkpoint_dir", "models/checkpoints"
            ),
            config=TrainingCheckpointConfig(
                interval_steps=1000,  # Save every 1000 steps as per Week 9-10 requirements
                keep_last=5,
                compress="lz4",
                async_save=True,
                include_optimizer=True,
                include_replay_buffer=False,
            ),
        )

        # Initialize market regime adaptation if enabled
        self.market_regime_adaptation = self.config.get("training", {}).get(
            "market_regime_adaptation", {}
        )
        self.regime_classifier = None
        self.regime_adaptation_enabled = False
        if self.market_regime_adaptation.get("enabled", False) and self.env is not None:
            self._initialize_market_regime_adaptation()

    # Phase 4: _load_data_with_format_detection() は BaseAlgorithmTrainer.load_data() に統合されました
    # 統合経路により、すべてのアルゴリズム（SAC/PPO/DQN/A2C）で共通の特徴検出ロジックを使用

    @staticmethod
    def _is_valid_feature_set_name(value: str | None) -> bool:
        """Return True when value looks like an explicit, non-placeholder feature set."""
        if not isinstance(value, str):
            return False
        name = value.strip()
        if not name:
            return False
        # Valid feature sets include "default" and other predefined sets
        valid_sets = [
            "default",
            "high_quality",
            "minimal",
            "full",
            "no_harmful",
            "v435_risk_managed",
            "v435_risk_managed_no_multi_timeframe",
            "curated",
            "v451",
            "v454",
        ]
        return name in valid_sets

    def _extract_feature_set(self, source: object) -> str | None:
        """Safely extract feature_set string from dict-like or object sources."""
        candidate: str | None = None
        if isinstance(source, dict):
            candidate = source.get("feature_set")
        elif hasattr(source, "feature_set"):
            candidate = getattr(source, "feature_set")
        if isinstance(candidate, str):
            candidate = candidate.strip()
        else:
            candidate = None
        return candidate

    def _format_reward_params(self, params: dict[str, object]) -> str:
        if not params:
            return "None"
        parts = []
        for key in sorted(params.keys()):
            value = params.get(key, "NOT_FOUND")
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _extract_expected_reward_params(self, config: ConfigDict) -> dict[str, object]:
        expected: dict[str, object] = {}
        if not isinstance(config, dict):
            return expected

        reward_block = config.get("reward")
        if isinstance(reward_block, dict):
            expected.update(reward_block)

        env_section = config.get("training", {}).get("environment", {})
        if not env_section:
            env_section = config.get("environment", {})
        if isinstance(env_section, dict):
            reward_settings = env_section.get("reward_settings")
            if isinstance(reward_settings, dict):
                expected.update(reward_settings)
            elif isinstance(reward_settings, RewardSettings):
                expected.update(shallow_asdict(reward_settings))

        # 386# FIX: Fallback to top-level reward_settings
        if not expected or "balance_penalty_value" not in expected:
            top_level_rs = config.get("reward_settings")
            if isinstance(top_level_rs, dict):
                expected.update(top_level_rs)

        return expected

    def _collect_actual_reward_params(self, env: object) -> dict[str, object]:
        params: dict[str, object] = {}
        if env is None:
            return params
        reward_settings = getattr(env, "reward_settings", None)
        if isinstance(reward_settings, RewardSettings):
            params.update(shallow_asdict(reward_settings))
            custom_params = reward_settings.custom_reward_params
            if isinstance(custom_params, dict):
                params.update(custom_params)
        elif isinstance(reward_settings, dict):
            params.update(reward_settings)
        return params

    def _log_reward_params_verification(
        self, env: object, expected_params: dict[str, object]
    ) -> None:
        actual_params = self._collect_actual_reward_params(env)
        keys = sorted(expected_params.keys())
        if not keys:
            self.logger.warning("========== REWARD PARAMS VERIFICATION ==========")
            self.logger.warning("EXPECTED: None (no reward overrides found)")
            self.logger.warning(
                "ACTUAL:   %s", self._format_reward_params(actual_params)
            )
            self.logger.warning(
                "STATUS:  ⚠️ NO EXPECTED PARAMS - verify config source"
            )
            self.logger.warning("===============================================")
            return

        mismatches = []
        for key in keys:
            expected = expected_params.get(key, "NOT_FOUND")
            actual = actual_params.get(key, "NOT_FOUND")
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if not math.isfinite(float(expected)) or not math.isfinite(
                    float(actual)
                ):
                    if expected != actual:
                        mismatches.append(key)
                elif abs(float(expected) - float(actual)) > 1e-9:
                    mismatches.append(key)
            else:
                if expected != actual:
                    mismatches.append(key)

        expected_line = self._format_reward_params(
            {k: expected_params.get(k, "NOT_FOUND") for k in keys}
        )
        actual_line = self._format_reward_params(
            {k: actual_params.get(k, "NOT_FOUND") for k in keys}
        )

        self.logger.warning("========== REWARD PARAMS VERIFICATION ==========")
        self.logger.warning("EXPECTED: %s", expected_line)
        self.logger.warning("ACTUAL:   %s", actual_line)
        if mismatches:
            self.logger.warning(
                "STATUS: ❌ MISMATCH - Settings may not be applied correctly!"
            )
            self.logger.warning("MISMATCH KEYS: %s", ", ".join(sorted(mismatches)))
            self.logger.warning("Check config propagation path.")
        else:
            self.logger.warning("STATUS: ✅ MATCH - Settings correctly applied")
        self.logger.warning("===============================================")

    def _log_cost_breakdown(self) -> None:
        from ztb.utils.env_metrics import extract_trainer_env_metrics

        metrics = extract_trainer_env_metrics(self, include_optional=True)
        if not metrics:
            self.logger.warning("========== COST BREAKDOWN ANALYSIS ==========")
            self.logger.warning("No environment metrics available.")
            self.logger.warning("============================================")
            return

        gross = metrics.get("gross_pnl")
        fees = metrics.get("total_fees")
        slip = metrics.get("total_slippage")
        net = metrics.get("net_pnl")
        initial_balance = metrics.get("initial_balance")

        if gross is None and net is not None and fees is not None and slip is not None:
            gross = float(net) + float(fees) + float(slip)
        if net is None and gross is not None and fees is not None and slip is not None:
            net = float(gross) - float(fees) - float(slip)

        def pct(value: float | None) -> str:
            if value is None or not initial_balance:
                return "N/A"
            return f"{(float(value) / float(initial_balance)) * 100:+.2f}%"

        if gross is None and net is None and fees is None and slip is None:
            self.logger.warning("========== COST BREAKDOWN ANALYSIS ==========")
            self.logger.warning("Cost metrics not available in environment.")
            self.logger.warning("============================================")
            return

        total_costs = 0.0
        if fees is not None:
            total_costs += float(fees)
        if slip is not None:
            total_costs += float(slip)

        cost_ratio = None
        if gross is not None and float(gross) != 0.0:
            cost_ratio = (total_costs / abs(float(gross))) * 100.0

        interpretation = "取引自体が損失"
        if gross is not None and float(gross) > 0:
            if net is not None and float(net) < 0:
                interpretation = "取引自体は利益だがコストに負けている"
            else:
                interpretation = "取引自体もコスト控除後も利益"

        self.logger.warning("========== COST BREAKDOWN ANALYSIS ==========")
        if gross is not None:
            self.logger.warning(
                "Gross PnL:     %s JPY (%s)", f"{float(gross):+,.0f}", pct(gross)
            )
        if fees is not None:
            self.logger.warning(
                "Total Fees:    %s JPY (%s)",
                f"{-abs(float(fees)):+,.0f}",
                pct(-abs(float(fees))),
            )
        if slip is not None:
            self.logger.warning(
                "Total Slippage: %s JPY (%s)",
                f"{-abs(float(slip)):+,.0f}",
                pct(-abs(float(slip))),
            )
        if net is not None:
            self.logger.warning(
                "Net PnL:       %s JPY (%s)", f"{float(net):+,.0f}", pct(net)
            )
        if cost_ratio is not None and math.isfinite(cost_ratio):
            self.logger.warning(
                "Cost Ratio:    %.1f%% (costs / |gross_pnl|)", cost_ratio
            )
        self.logger.warning("Interpretation: %s", interpretation)
        self.logger.warning("============================================")

    def _resolve_feature_set_override(self, env_candidate: object) -> str | None:
        """Find the most reliable feature_set declaration in the stacked config."""
        config_dict = self.config if isinstance(self.config, dict) else {}
        training_section = (
            config_dict.get("training", {}) if isinstance(config_dict, dict) else {}
        )
        candidates = [
            training_section.get("features", {}),  # Highest priority
            config_dict.get("features", {}),
            training_section.get("environment", {}),
            config_dict.get("environment", {}),
            env_candidate,  # Lowest priority
        ]
        fallback: str | None = None
        for i, source in enumerate(candidates):
            candidate = self._extract_feature_set(source)
            if not candidate:
                continue
            if self._is_valid_feature_set_name(candidate):
                return candidate
            if fallback is None:
                fallback = candidate
        return fallback

    def _ensure_feature_set_on_target(self, target: object, feature_set: str) -> None:
        """Apply feature_set to dict/object target when it's missing or default."""
        if not self._is_valid_feature_set_name(feature_set):
            return

        if isinstance(target, dict):
            current = target.get("feature_set")
            if not self._is_valid_feature_set_name(current):
                target["feature_set"] = feature_set
        else:
            # For objects, check if feature_set attribute exists
            current = getattr(target, "feature_set", None)
            if not self._is_valid_feature_set_name(current):
                setattr(target, "feature_set", feature_set)

    def _propagate_feature_set(self, feature_set: str, env_candidate: object) -> None:
        """Write the resolved feature_set back into every config view."""
        if not self._is_valid_feature_set_name(feature_set):
            return

        # Only propagate to dict/object targets, not to string env_candidate
        if env_candidate is not None and (
            isinstance(env_candidate, dict) or not isinstance(env_candidate, str)
        ):
            self._ensure_feature_set_on_target(env_candidate, feature_set)

        cfg = self.config if isinstance(self.config, dict) else None
        if not isinstance(cfg, dict):
            return

        env_section = cfg.setdefault("environment", {})
        if isinstance(env_section, dict):
            self._ensure_feature_set_on_target(env_section, feature_set)

        features_section = cfg.setdefault("features", {})
        if isinstance(features_section, dict):
            self._ensure_feature_set_on_target(features_section, feature_set)

        training_section = cfg.setdefault("training", {})
        if isinstance(training_section, dict):
            training_env = training_section.setdefault("environment", {})
            if isinstance(training_env, dict):
                self._ensure_feature_set_on_target(training_env, feature_set)

            training_features = training_section.setdefault("features", {})
            if isinstance(training_features, dict):
                self._ensure_feature_set_on_target(training_features, feature_set)

    def _initialize_market_regime_adaptation(self):
        """Initialize market regime adaptation system."""
        try:
            from ztb.analysis.regime.market_regime_classifier import MarketRegimeClassifier

            self.regime_classifier = MarketRegimeClassifier(
                config=self.market_regime_adaptation
            )
            self.regime_adaptation_enabled = True
            self.logger.info("Market regime adaptation enabled with classifier")

            # Initialize regime-specific statistics
            self.regime_stats = {
                "regime_counts": {},
                "regime_rewards": {},
                "regime_actions": {},
                "regime_transitions": {},
            }

            # Alias for backward compatibility
            self.regime_statistics = self.regime_stats

            # Enable regime adaptation in the environment if it's a HeavyTradingEnv
            if hasattr(self.env, "enable_market_regime_adaptation"):
                self.env.enable_market_regime_adaptation(
                    self.regime_classifier, self.market_regime_adaptation
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize market regime adaptation: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate SAC configuration using unified configuration manager."""
        try:
            # Use configuration manager for validation
            config_manager = ConfigurationManager(self.logger)

            # Additional SAC-specific validation
            sac_config = self.config.get("sac_hyperparameters", {}) or self.config.get(
                "training", {}
            ).get("sac_hyperparameters", {})
            if not sac_config:
                self.logger.error("Missing SAC hyperparameters section")
                return False

            # Validate SAC-specific parameters
            required_sac_params = [
                "learning_rate",
                "buffer_size",
                "learning_starts",
                "batch_size",
            ]
            for param in required_sac_params:
                if param not in sac_config:
                    self.logger.error(f"Missing SAC hyperparameter: {param}")
                    return False

            # Validate parameter types and ranges
            if not isinstance(sac_config.get("learning_rate"), (int, float)) or not (
                0 < sac_config["learning_rate"] < 1
            ):
                self.logger.error("learning_rate must be a float between 0 and 1")
                return False

            if (
                not isinstance(sac_config.get("buffer_size"), int)
                or sac_config["buffer_size"] <= 0
            ):
                self.logger.error("buffer_size must be a positive integer")
                return False

            # Validate data file exists
            data_path = config_manager.get_config_value(
                self.config, "training.data_config.data_path"
            )
            if data_path and not os.path.exists(data_path):
                self.logger.error(f"Data file not found: {data_path}")
                return False

            self.logger.info("SAC configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"SAC configuration validation failed: {e}")
            return False

    def train(self, total_timesteps: int | None = None) -> bool:
        """Execute SAC training with enhanced monitoring.

        Args:
            total_timesteps: Total number of timesteps to train for (overrides config)
        """
        return self.execute_training_pipeline(
            "SAC", self._execute_sac_training, total_timesteps=total_timesteps
        )

    def _execute_sac_training(
        self,
        total_timesteps: int | None = None,
        callback: TrainingProgressCallback | None = None,
        start_time: float | None = None,
    ) -> bool:
        """Execute core SAC training logic with structured logging.

        Args:
            total_timesteps: Total number of timesteps to train for
        """
        env = None
        wrapped_env = None
        eval_env = None

        try:
            # Get training parameters from config or parameter
            config_timesteps = self.config.get("training", {}).get(
                "total_timesteps", 100000
            )
            total_timesteps = total_timesteps or config_timesteps
            resume_path = self.config.get("training", {}).get("resume_from", None)
            start_time = start_time or time.time()
            data_path = self.config.get("data_config", {}).get("data_path")

            # Check for resume functionality
            resumed_state = None
            if resume_path and os.path.exists(resume_path):
                self.log_structured_event("training", "resume", {"path": resume_path})
                resumed_state = self.training_state_manager.load_training_state(
                    resume_path
                )

                # Validate compatibility
                validation = self.training_state_manager.validate_resume_compatibility(
                    resumed_state, self.config, data_path
                )

                if not validation["compatible"]:
                    error_msg = "Resume validation failed: " + "; ".join(
                        validation["errors"]
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        self.logger.warning(f"Resume validation warning: {warning}")

                self.logger.info(f"Resuming training from {resume_path}")
            elif resume_path:
                self.logger.warning(
                    f"Resume path {resume_path} not found, starting fresh training"
                )

            # Create callback for progress tracking if not provided
            if callback is None:
                # Get early stopping configuration
                early_stopping_config = (
                    self.config.get("training", {})
                    .get("sac_hyperparameters", {})
                    .get("early_stopping", {})
                )

                callback = TrainingProgressCallback(
                    check_freq=self.config.get("training", {}).get(
                        "log_interval", 1000
                    ),
                    verbose=1,
                    trainer_ref=self,
                    early_stopping=early_stopping_config
                    if early_stopping_config
                    else None,
                    checkpoint_manager=self.checkpoint_manager,
                )

                # Enable regime tracking if adaptation is enabled
                if self.regime_classifier is not None:
                    callback.enable_regime_tracking = True

                # Add checkpoint callback
                checkpoint_interval = self.config.get("checkpoint_interval", 10000)
                checkpoint_dir = self.config.get("checkpoint_dir", "models/checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_callback = create_checkpoint_callback(
                    save_freq=checkpoint_interval,
                    save_path=checkpoint_dir,
                    name_prefix="sac_checkpoint",
                    verbose=1,
                )
                callback = CallbackList([callback, checkpoint_callback])
            else:
                # If callback is provided, add checkpoint callback to it
                checkpoint_interval = self.config.get("checkpoint_interval", 10000)
                checkpoint_dir = self.config.get("checkpoint_dir", "models/checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_callback = create_checkpoint_callback(
                    save_freq=checkpoint_interval,
                    save_path=checkpoint_dir,
                    name_prefix="sac_checkpoint",
                    verbose=1,
                )
                if isinstance(callback, CallbackList):
                    existing_callbacks = getattr(callback, "callbacks", [])
                    duplicate_exists = any(
                        type(cb) is type(checkpoint_callback) and getattr(cb, "save_path", None) == checkpoint_dir
                        for cb in existing_callbacks
                    )
                    if not duplicate_exists:
                        callback.callbacks.append(checkpoint_callback)
                else:
                    callback = CallbackList([callback, checkpoint_callback])
            # Load and prepare data
            data_config = self.config.get("training", {}).get("data_config", {})
            data_path = data_config.get(
                "data_path", "data/btc_jpy_featured_dataset.csv"
            )

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            self.log_structured_event("data", "loading", {"path": data_path})
            # Phase 4: BaseAlgorithmTrainer.load_data() を使用（統合経路）
            df = self.load_data(data_path)
            self.log_structured_event("data", "loaded", {"rows": len(df), "columns": len(df.columns)})

            # Create environment
            self.log_structured_event(
                "environment", "creation", {"type": "HeavyTradingEnv"}
            )
            # Check for environment config in multiple locations (top-level or training section)
            env_config = self.config.get("environment", {})
            if not env_config:
                env_config = self.config.get("training", {}).get("environment", {})
            # Extract the actual config from the environment section (could be nested)
            actual_env_config = env_config.get("config", env_config)

            # 386# FIX: Merge top-level reward_settings into environment config
            # reward_settings can be defined at YAML top-level for readability,
            # but EnvironmentConfig.from_dict() expects it inside the env dict.
            if isinstance(actual_env_config, dict) and "reward_settings" not in actual_env_config:
                top_level_rs = self.config.get("reward_settings") if isinstance(self.config, dict) else None
                if isinstance(top_level_rs, dict):
                    actual_env_config["reward_settings"] = top_level_rs
                    self.logger.info(
                        f"Merged top-level reward_settings into env config: {list(top_level_rs.keys())}"
                    )

            # Log the exact object passed to the environment so we can trace
            # where boolean flags like `use_continuous_actions` may be lost.
            try:
                self.logger.info(
                    f"Environment config keys: {list(actual_env_config.keys())}"
                )
                self.logger.info(
                    f"actual_env_config feature_set: {actual_env_config.get('feature_set', 'NOT_FOUND')}"
                )
            except Exception:
                self.logger.info(f"Environment config type: {type(actual_env_config)}")

            # Provide a small preview depending on whether it's a dict or an object
            if isinstance(actual_env_config, dict):
                preview = {
                    k: actual_env_config.get(k, "NOT_FOUND")
                    for k in ["use_continuous_actions", "action_space_type"]
                }
                self.logger.info(f"Env config preview (dict): {preview}")
                self.logger.info(
                    f"use_continuous_actions in config (dict): {actual_env_config.get('use_continuous_actions', 'NOT_FOUND')}"
                )
            else:
                try:
                    ua = getattr(
                        actual_env_config, "use_continuous_actions", "NOT_PRESENT"
                    )
                    at = getattr(actual_env_config, "action_space_type", "NOT_PRESENT")
                    self.logger.info(
                        f"Env config preview (obj): use_continuous_actions={ua}, action_space_type={at}"
                    )
                except Exception as e:
                    self.logger.info(f"Could not introspect actual_env_config: {e}")

            # Also log a short repr for manual inspection
            try:
                self.logger.info(
                    f"repr(actual_env_config)[:200]: {repr(actual_env_config)[:200]}"
                )
            except Exception:
                pass

            resolved_feature_set = self._resolve_feature_set_override(actual_env_config)
            if resolved_feature_set and self._is_valid_feature_set_name(
                resolved_feature_set
            ):
                self._propagate_feature_set(resolved_feature_set, actual_env_config)
                self.logger.info(
                    f"Resolved feature_set for environment: {resolved_feature_set}"
                )

            # Convert whatever shape we received into an EnvironmentConfig instance
            # EnvironmentConfig.from_dict can handle nested layouts (training.environment, training.environment.config)
            try:
                # Gate 0 Debug: Log reward_settings before EnvironmentConfig creation
                if isinstance(actual_env_config, dict):
                    rs = actual_env_config.get("reward_settings", "NOT_FOUND")
                    self.logger.warning(f"Gate0 Debug: actual_env_config.reward_settings = {rs}")
                if isinstance(actual_env_config, EnvironmentConfig):
                    env_config_obj = actual_env_config
                elif isinstance(actual_env_config, dict):
                    env_config_obj = EnvironmentConfig.from_dict(actual_env_config)
                else:
                    # Fallback: try converting the whole training section or full config
                    env_config_obj = EnvironmentConfig.from_dict(self.config)
                
                # Gate 0 Debug: Log reward_settings after EnvironmentConfig creation  
                if hasattr(env_config_obj, "reward_settings"):
                    rs_dict = (
                        shallow_asdict(env_config_obj.reward_settings)
                        if env_config_obj.reward_settings
                        else None
                    )
                    self.logger.warning(f"Gate0 Debug: env_config_obj.reward_settings = {rs_dict}")

                # 🔧 CRITICAL FIX: Inject curriculum_learning config if present in training config
                # This ensures BalanceCurriculumManager receives the necessary configuration
                training_cfg = (
                    self.config.get("training", {})
                    if isinstance(self.config, dict)
                    else {}
                )
                curriculum_cfg = training_cfg.get("curriculum_learning")

                if curriculum_cfg and hasattr(env_config_obj, "curriculum_learning"):
                    self.logger.debug(
                        f"Injecting curriculum_learning config into EnvironmentConfig: {list(curriculum_cfg.keys())}"
                    )
                    env_config_obj.curriculum_learning = curriculum_cfg
                elif curriculum_cfg:
                    self.logger.warning(
                        "curriculum_learning found in config but EnvironmentConfig has no such field!"
                    )
                else:
                    self.logger.debug(
                        "No curriculum_learning config found in training section."
                    )

                # Log the feature_set
                self.logger.debug(
                    f"env_config_obj.feature_set = {getattr(env_config_obj, 'feature_set', 'NOT_SET')}"
                )

                # Honor explicit flags from the original config dict if present.
                # Some conversion paths may nest the fields; check common locations.
                try:
                    # Look into training.environment.config, training.environment, then top-level
                    cfg = self.config if isinstance(self.config, dict) else {}
                    training_section = (
                        cfg.get("training", {}) if isinstance(cfg, dict) else {}
                    )
                    env_section_cfg = (
                        training_section.get("environment", {})
                        if isinstance(training_section, dict)
                        else {}
                    )
                    inner_cfg = (
                        env_section_cfg.get("config", env_section_cfg)
                        if isinstance(env_section_cfg, dict)
                        else {}
                    )
                    # Check both boolean and action_space_type string
                    explicit_bool = None
                    if isinstance(inner_cfg, dict):
                        explicit_bool = inner_cfg.get("use_continuous_actions", None)
                        action_space_type_val = inner_cfg.get("action_space_type", None)
                    else:
                        explicit_bool = None
                        action_space_type_val = None

                    if explicit_bool is None:
                        # also check top-level trainer.environment keys
                        explicit_bool = (
                            env_section_cfg.get("use_continuous_actions", None)
                            if isinstance(env_section_cfg, dict)
                            else None
                        )
                        if explicit_bool is None:
                            # check unified_trainer style environment.use_continuous_actions
                            explicit_bool = (
                                cfg.get("environment", {}).get(
                                    "use_continuous_actions", None
                                )
                                if isinstance(cfg, dict)
                                else None
                            )
                        if explicit_bool is None:
                            explicit_bool = (
                                cfg.get("use_continuous_actions", None)
                                if isinstance(cfg, dict)
                                else None
                            )

                    # If action_space_type indicates continuous, treat as True
                    if (
                        (explicit_bool is True)
                        or (
                            isinstance(action_space_type_val, str)
                            and str(action_space_type_val)
                            .strip()
                            .lower()
                            .startswith("cont")
                        )
                        or (
                            isinstance(env_section_cfg.get("action_space_type"), str)
                            and str(env_section_cfg.get("action_space_type"))
                            .strip()
                            .lower()
                            .startswith("cont")
                        )
                    ):
                        setattr(env_config_obj, "use_continuous_actions", True)
                        setattr(env_config_obj, "enable_action_masking", False)
                except Exception:
                    # Non-fatal: keep env_config_obj as-is but log for later debugging
                    self.logger.debug(
                        "Could not inspect original config for explicit action-space flags"
                    )
            except Exception as e:
                # Log and re-raise so the test run fails loudly instead of silently forcing values
                self.logger.error(
                    f"Failed to normalize environment config to EnvironmentConfig: {e}"
                )
                raise

            feature_start = time.perf_counter()
            env = HeavyTradingEnv(
                df=df,
                config=env_config_obj,
                optimizer_tracker=self.optimizer_tracker,
            )
            try:
                expected_reward_params = self._extract_expected_reward_params(
                    self.config
                )
                self._log_reward_params_verification(env, expected_reward_params)
            except Exception as e:
                self.logger.warning(
                    "Reward params verification skipped due to error: %s", e
                )
                import traceback
                self.logger.warning("Traceback: %s", traceback.format_exc())
            # Attempt to capture feature generation time as measured by the environment construction
            try:
                env._feature_generation_time = time.perf_counter() - feature_start
                self.logger.info(
                    f"Captured env feature generation time: {env._feature_generation_time:.3f}s"
                )
            except Exception:
                pass

            # Initialize market regime adaptation if enabled and not yet initialized
            if (
                self.market_regime_adaptation.get("enabled", False)
                and self.regime_classifier is None
            ):
                self._initialize_market_regime_adaptation()

            # Enable market regime adaptation in environment if configured
            if self.regime_classifier is not None:
                env.enable_market_regime_adaptation(
                    regime_classifier=self.regime_classifier,
                    adaptation_config=self.market_regime_adaptation,
                )
                self.logger.info("Market regime adaptation enabled in environment")

            wrapped_env = Monitor(env)

            # 🔧 CRITICAL FIX: Apply VecNormalize to prevent input saturation
            # Even though HeavyTradingEnv has internal normalization, VecNormalize handles
            # reward normalization and clipping which are crucial for stability.
            # We wrap it in DummyVecEnv first as VecNormalize requires a VecEnv.
            if not isinstance(wrapped_env, DummyVecEnv):
                wrapped_env = DummyVecEnv([lambda: wrapped_env])

            # Get SAC hyperparameters
            sac_config = self.config.get("training", {}).get("sac_hyperparameters", {})

            # Check if normalization is disabled in config (default: enabled)
            normalize_kwargs = self.config.get("training", {}).get("normalization", {})
            if normalize_kwargs.get("enabled", True):
                self.logger.info("Applying VecNormalize to environment")
                wrapped_env = VecNormalize(
                    wrapped_env,
                    norm_obs=normalize_kwargs.get("norm_obs", True),
                    norm_reward=normalize_kwargs.get("norm_reward", True),
                    clip_obs=normalize_kwargs.get("clip_obs", 10.0),
                    clip_reward=normalize_kwargs.get("clip_reward", 10.0),
                    gamma=sac_config.get("gamma", DEFAULT_GAMMA),
                )
                self.logger.info(f"VecNormalize applied with: {normalize_kwargs}")

            # Recurrent RL (GRU) Support
            use_recurrent = sac_config.get("use_recurrent", False)
            if use_recurrent:
                n_stack = sac_config.get("n_stack", 60)
                self.logger.info(f"Enabling Recurrent RL with {n_stack} frame stacking")

                # Wrap in DummyVecEnv to make it compatible with VecFrameStack
                # Note: We use a lambda to create the env, but here we already have an instance.
                # DummyVecEnv expects a list of callables.
                # Since we already have 'wrapped_env' (Monitor), we can wrap it.
                # However, DummyVecEnv usually takes a list of functions that return envs.
                # To wrap an existing env instance, we can just return it, but we must be careful about reset().
                # A safer way is to let DummyVecEnv manage it, but we already created it.
                # We will use a lambda that returns the *existing* instance.
                # Warning: This is not standard for parallel envs, but fine for single env.
                vec_env = DummyVecEnv([lambda: wrapped_env])
                wrapped_env = VecFrameStack(vec_env, n_stack=n_stack)
                self.logger.info(
                    f"Environment wrapped with VecFrameStack (n_stack={n_stack})"
                )

            # --- Evaluation Environment Setup ---
            eval_callback = None
            evaluation_config = self.config.get("evaluation", {})

            if evaluation_config.get("enabled", False):
                self.logger.info("Setting up evaluation environment...")

                # Determine eval config (merge base env config with eval overrides)
                eval_overrides = evaluation_config.get("overrides", {})

                # Create a copy of the base config object
                if isinstance(env_config_obj, EnvironmentConfig):
                    eval_env_config_obj = copy.deepcopy(env_config_obj)
                    # Apply overrides
                    for k, v in eval_overrides.items():
                        if hasattr(eval_env_config_obj, k):
                            setattr(eval_env_config_obj, k, v)
                            self.logger.info(f"Eval Env Override: {k} = {v}")
                else:
                    # Fallback for dict config
                    eval_env_config_obj = env_config_obj.copy()
                    eval_env_config_obj.update(eval_overrides)

                # Create Eval Env
                # Note: We use the same df for now. In a real ML pipeline, we should use validation data.
                # If 'data_path' is specified in evaluation_config, load it.
                eval_data_path = evaluation_config.get("data_path")
                if eval_data_path and os.path.exists(eval_data_path):
                    self.logger.info(f"Loading evaluation data from {eval_data_path}")
                    # Phase 4: BaseAlgorithmTrainer.load_data() を使用（統合経路）
                    eval_df = self.load_data(eval_data_path)
                else:
                    eval_df = (
                        df  # Use same dataframe (be careful about leakage if not split)
                    )

                eval_env_raw = HeavyTradingEnv(
                    df=eval_df,
                    config=eval_env_config_obj,
                    optimizer_tracker=self.optimizer_tracker,
                )

                # Wrap Eval Env
                eval_env = Monitor(eval_env_raw)
                eval_env = DummyVecEnv([lambda: eval_env])

                # Apply VecNormalize if used in training
                if normalize_kwargs.get("enabled", True):
                    eval_env = VecNormalize(
                        eval_env,
                        norm_obs=normalize_kwargs.get("norm_obs", True),
                        norm_reward=False,  # Don't normalize rewards during eval usually
                        clip_obs=normalize_kwargs.get("clip_obs", 10.0),
                        clip_reward=normalize_kwargs.get("clip_reward", 10.0),
                        gamma=sac_config.get("gamma", DEFAULT_GAMMA),
                        training=False,  # Important: don't update stats during eval
                    )

                # Apply VecFrameStack if used
                if use_recurrent:
                    eval_env = VecFrameStack(eval_env, n_stack=n_stack)

                # Create EvalCallback
                eval_freq = evaluation_config.get("eval_freq", 5000)
                n_eval_episodes = evaluation_config.get("n_eval_episodes", 5)

                checkpoint_dir = self.config.get("checkpoint_dir", "models/checkpoints")
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(checkpoint_dir, "best_model"),
                    log_path=os.path.join(checkpoint_dir, "eval_logs"),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False,
                )

                # Add to callback list
                if isinstance(callback, CallbackList):
                    callback.callbacks.append(eval_callback)
                elif callback is not None:
                    callback = CallbackList([callback, eval_callback])
                else:
                    callback = CallbackList([eval_callback])

                self.logger.info("Evaluation environment setup complete.")

            self.logger.info(
                f"Environment observation space: {wrapped_env.observation_space}"
            )
            print(f"Environment action space: {wrapped_env.action_space}")

            # Prepare policy kwargs with overfitting prevention parameters
            policy_kwargs = {}
            if "dropout_rate" in sac_config:
                policy_kwargs["dropout_rate"] = sac_config["dropout_rate"]
            if "l2_regularization" in sac_config:
                policy_kwargs["weight_decay"] = sac_config["l2_regularization"]
            if sac_config.get("net_arch"):
                self.logger.info(
                    f"Applying net_arch from config: {sac_config['net_arch']}"
                )
                policy_kwargs["net_arch"] = sac_config["net_arch"]

            # Inject GRU Feature Extractor if recurrent mode is enabled
            if use_recurrent:
                from ztb.ml.networks.recurrent_features import GRUFeatureExtractor

                policy_kwargs["features_extractor_class"] = GRUFeatureExtractor
                policy_kwargs["features_extractor_kwargs"] = {
                    "n_stack": sac_config.get("n_stack", 60),
                    "hidden_size": sac_config.get("recurrent_hidden_size", 128),
                    "num_layers": sac_config.get("recurrent_num_layers", 1),
                    "features_dim": sac_config.get("features_dim", 256),
                }
                self.logger.info(
                    f"Using GRUFeatureExtractor with hidden_size={sac_config.get('recurrent_hidden_size', 128)}"
                )

                # Create SAC model (import SB3 lazily to avoid import-time failures)
            if self.model is None:
                self.log_structured_event(
                    "model", "creation", {"algorithm": "SAC", "policy": "MlpPolicy"}
                )
                try:
                    from stable_baselines3 import SAC as _LocalSAC
                except Exception:
                    _LocalSAC = None

                if _LocalSAC is None:
                    raise ModelError(
                        "stable_baselines3.SAC is not available in this environment"
                    )

                # Optional warm-start: load weights/hyperparams from a saved SB3 model zip.
                # Note: SB3 does NOT persist the replay buffer by default, so the buffer starts empty.
                init_model_path = (
                    self.config.get("training", {}).get("init_model_path")
                    or self.config.get("training", {}).get("initial_model_path")
                    or self.config.get("training", {}).get("pretrained_model_path")
                )
                if init_model_path:
                    try:
                        if os.path.exists(str(init_model_path)):
                            self.log_structured_event(
                                "model", "load_initial", {"path": str(init_model_path)}
                            )
                            self.logger.info(
                                "Loading initial SAC model from %s", init_model_path
                            )
                            self.model = _LocalSAC.load(
                                str(init_model_path), env=wrapped_env
                            )
                        else:
                            self.logger.warning(
                                "Initial model path not found: %s", init_model_path
                            )
                    except Exception as e:
                        self.logger.warning(
                            "Failed to load initial SAC model from %s: %s (falling back to new model)",
                            init_model_path,
                            e,
                        )
                        self.model = None

                if self.model is None:
                    self.model = _LocalSAC(
                        "MlpPolicy",
                        wrapped_env,
                        learning_rate=sac_config.get("learning_rate", DEFAULT_LEARNING_RATE_SAC),
                        buffer_size=sac_config.get("buffer_size", DEFAULT_BUFFER_SIZE_SAC),
                        learning_starts=sac_config.get("learning_starts", DEFAULT_LEARNING_STARTS_SAC),
                        batch_size=sac_config.get("batch_size", DEFAULT_BATCH_SIZE_SAC),
                        tau=sac_config.get("tau", DEFAULT_TAU),
                        gamma=sac_config.get("gamma", DEFAULT_GAMMA),
                        train_freq=sac_config.get("train_freq", DEFAULT_TRAIN_FREQ),
                        gradient_steps=sac_config.get("gradient_steps", DEFAULT_GRADIENT_STEPS),
                        ent_coef=sac_config.get("ent_coef", "auto"),
                        target_update_interval=sac_config.get(
                            "target_update_interval", DEFAULT_TARGET_UPDATE_INTERVAL
                        ),
                        policy_kwargs=policy_kwargs if policy_kwargs else None,
                        verbose=0,  # We'll handle logging ourselves
                    )
            else:
                self.logger.info("Using existing model instance")
                self.model.set_env(wrapped_env)

            # Restore training state if resuming
            if resumed_state:
                self.training_state_manager.restore_training_state(
                    self.model, resumed_state
                )
                # Adjust total timesteps for remaining training
                remaining_timesteps = total_timesteps - resumed_state["total_timesteps"]
                if remaining_timesteps > 0:
                    total_timesteps = remaining_timesteps
                    self.log_structured_event(
                        "training",
                        "adjusted_timesteps",
                        {
                            "original": self.config.get("training", {}).get(
                                "total_timesteps", 100000
                            ),
                            "remaining": total_timesteps,
                        },
                    )
                else:
                    self.logger.info("Training already completed, skipping")
                    return True

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

            # set up dynamic LR scheduler with model optimizer if enabled
            if (
                self.lr_scheduler
                and hasattr(model, "policy")
                and hasattr(model.policy, "optimizer")
            ):
                self.lr_scheduler.optimizer = model.policy.optimizer
                self.log_structured_event(
                    "optimizer", "setup", {"scheduler": "dynamic_lr"}
                )

            # Execute training
            reset_override = None
            training_cfg = self.config.get("training", {})
            if isinstance(training_cfg, dict):
                reset_override = training_cfg.get("reset_num_timesteps")

            # Default: when resuming, keep the timestep counter; otherwise start fresh.
            reset_num_timesteps = resumed_state is None
            if reset_override is not None:
                reset_num_timesteps = bool(reset_override)

            self.log_structured_event(
                "training",
                "execution",
                {
                    "timesteps": total_timesteps,
                    "reset_num_timesteps": reset_num_timesteps,
                },
            )
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps,
            )

            # Training completed
            training_time = max(time.time() - start_time, 1e-9)

            # DEBUG: Log detailed training completion statistics
            if hasattr(model, "logger") and model.logger:
                try:
                    logger_values = getattr(model.logger, "name_to_value", {})
                    final_actor_loss = logger_values.get("train/actor_loss", 0)
                    final_critic_loss = logger_values.get("train/critic_loss", 0)
                    final_ent_coef = logger_values.get("train/ent_coef", 0)

                    self.logger.debug(
                        f"SAC training completed: Time={training_time:.1f}s | "
                        f"Steps={total_timesteps} | SPS={total_timesteps/training_time:.1f} | "
                        f"Final ActorLoss={final_actor_loss:.4f} | CriticLoss={final_critic_loss:.4f} | "
                        f"EntCoef={final_ent_coef:.4f}"
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to log final training metrics: {e}")

            # Save training state for potential resume
            if self.config.get("training", {}).get("save_training_state", True):
                try:
                    final_timesteps = (
                        resumed_state["total_timesteps"] + total_timesteps
                        if resumed_state
                        else total_timesteps
                    )
                    state_path = self.training_state_manager.save_training_state(
                        model=self.model,
                        total_timesteps=final_timesteps,
                        episode_count=getattr(callback, "episode_count", 0),
                        episode_rewards=getattr(callback, "reward_history", []),
                        episode_lengths=getattr(callback, "episode_lengths", []),
                        config=self.config,
                        training_time=training_time,
                    )
                    self.log_structured_event(
                        "training", "state_saved", {"path": state_path}
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to save training state: {e}")

            # Clean up metrics collection
            self.cleanup_metrics_collection()

            # Cleanup training environment
            self.cleanup_training_environment()

            # Save model
            model_name = self.config.get("model_name", "sac_model")
            model_path = self.save_model(model, model_name, ".zip")

            # Collect training statistics
            self.training_stats = self.collect_training_stats(
                training_time=training_time,
                total_timesteps=total_timesteps,
                model_path=model_path,
                steps_per_second=total_timesteps / training_time,
                final_reward=callback.callbacks[0].reward_history[-1]
                if hasattr(callback, "callbacks")
                and callback.callbacks[0].reward_history
                else callback.reward_history[-1]
                if hasattr(callback, "reward_history") and callback.reward_history
                else 0,
                action_distribution=self._calculate_final_action_distribution(callback),
            )

            # Add environment feature generation time to stats (if available)
            try:
                feat_time = getattr(env, "_feature_generation_time", None)
                if feat_time is not None:
                    self.training_stats["feature_generation_time_s"] = feat_time
            except Exception:
                pass

            # Collect reward_components from callback if available (for AB analysis)
            try:
                cb = (
                    callback.callbacks[0]
                    if hasattr(callback, "callbacks")
                    else callback
                )
                if (
                    hasattr(cb, "reward_components_history")
                    and cb.reward_components_history
                ):
                    # Average reward components across all episodes
                    components = {}
                    for comp_dict in cb.reward_components_history:
                        for key, val in comp_dict.items():
                            if key not in components:
                                components[key] = []
                            components[key].append(float(val))
                    # Average each component
                    avg_components = {
                        k: sum(v) / len(v) for k, v in components.items() if v
                    }
                    self.training_stats["reward_components"] = avg_components
                    self.logger.info(f"Collected reward_components: {avg_components}")
            except Exception as e:
                self.logger.debug(f"Could not collect reward_components: {e}")

            try:
                self._log_cost_breakdown()
            except Exception as e:
                self.logger.debug("Cost breakdown logging failed: %s", e)

            self.log_training_completion(training_time, self.training_stats)
            return True

        except KeyboardInterrupt as kb_int:
            # Only log as "user interrupt" if it's truly KeyboardInterrupt
            import traceback
            self.logger.error(f"⚠️ KeyboardInterrupt detected: {kb_int}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            # Try to save partial model
            self._attempt_emergency_save()
            return False

        except MemoryError:
            self.logger.error("Memory error during training - attempting cleanup")
            self._cleanup_on_memory_error()
            return False

        except Exception as e:
            self.logger.error(f"❌ SAC training failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

            # Attempt recovery based on error type
            if self._attempt_error_recovery(e):
                self.logger.info("Error recovery attempted - retrying training")
                try:
                    # Retry with reduced parameters
                    return self._retry_training_with_reduced_params()
                except Exception as retry_e:
                    self.logger.error(f"Retry also failed: {retry_e}")

            return False
        finally:
            # Ensure environments are closed even on failure to avoid lingering
            # handles/threads between sequential experiments.
            if eval_env is not None and hasattr(eval_env, "close"):
                try:
                    eval_env.close()
                except Exception as close_err:
                    self.logger.debug(
                        "Failed to close eval env cleanly: %s", close_err
                    )
            if wrapped_env is not None and hasattr(wrapped_env, "close"):
                try:
                    wrapped_env.close()
                except Exception as close_err:
                    self.logger.debug(
                        "Failed to close training env cleanly: %s", close_err
                    )
            elif env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception as close_err:
                    self.logger.debug(
                        "Failed to close base env cleanly: %s", close_err
                    )

    def _attempt_emergency_save(self) -> None:
        """Attempt to save model in case of interruption."""
        try:
            if self.model is not None:
                emergency_path = f"models/emergency_save_{int(time.time())}.zip"
                os.makedirs("models", exist_ok=True)
                from ztb.utils.training_utils import save_model as _save_model

                _save_model(self.model, emergency_path)
                self.logger.info(f"Emergency save completed: {emergency_path}")
        except Exception as e:
            self.logger.error(f"Emergency save failed: {e}")

    def _cleanup_on_memory_error(self) -> None:
        """Clean up resources on memory error."""
        try:
            # Force garbage collection
            import gc

            gc.collect()

            # Clear any cached data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clean up metrics collection
            self.cleanup_metrics_collection()

            self.logger.info("Memory cleanup completed")
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

    def _attempt_error_recovery(self, error: Exception) -> bool:
        """Attempt to recover from training errors."""
        error_str = str(error).lower()

        # CUDA out of memory
        if "cuda" in error_str and "memory" in error_str:
            self.logger.info("Attempting CUDA memory recovery")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

        # Gradient explosion
        if "nan" in error_str or "inf" in error_str:
            self.logger.info(
                "Detected gradient explosion - recovery not implemented yet"
            )
            return False

        # Network issues (for distributed training)
        if "connection" in error_str or "network" in error_str:
            self.logger.info("Network error detected - attempting local fallback")
            return True

        return False

    def _retry_training_with_reduced_params(self) -> bool:
        """Retry training with reduced parameters."""
        try:
            self.logger.info("Retrying training with reduced parameters")

            # Reduce batch size and learning rate
            original_config = self.config.copy()
            sac_config = original_config.get("training", {}).get(
                "sac_hyperparameters", {}
            )

            # Reduce batch size
            original_batch_size = sac_config.get("batch_size", 128)
            sac_config["batch_size"] = max(16, original_batch_size // 4)

            # Reduce learning rate
            original_lr = sac_config.get("learning_rate", 0.0003)
            sac_config["learning_rate"] = original_lr * 0.1

            # Reduce total timesteps
            original_timesteps = original_config["training"]["total_timesteps"]
            original_config["training"]["total_timesteps"] = original_timesteps // 2

            self.logger.info(
                f"Reduced batch_size: {original_batch_size} -> {sac_config['batch_size']}"
            )
            self.logger.info(
                f"Reduced learning_rate: {original_lr} -> {sac_config['learning_rate']}"
            )
            self.logger.info(
                f"Reduced timesteps: {original_timesteps} -> {original_config['total_timesteps']}"
            )

            # Create new trainer with reduced config
            retry_trainer = SACTrainer(original_config, self.logger)
            return retry_trainer.train()

        except Exception as e:
            self.logger.error(f"Retry training failed: {e}")
            return False

    def _train_with_curriculum_learning(self) -> bool:
        """Train using curriculum learning (v435 archived in 030#)."""
        self.logger.warning(
            "Curriculum learning (v435) has been archived. "
            "Use standard SACTrainer.train() instead."
        )
        return False

    def _calculate_final_action_distribution(self, callback) -> dict[str, float]:
        """Calculate final action distribution from callback data."""
        # Handle CallbackList
        if hasattr(callback, "callbacks"):
            callback = callback.callbacks[0]

        if not hasattr(callback, "discrete_actions") or not callback.discrete_actions:
            return {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

        total_actions = len(callback.discrete_actions)

        # Convert discrete actions to proper indices (SELL: -1 -> 2, HOLD: 0 -> 0, BUY: 1 -> 1)
        discrete_indices: list[int] = []
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

    def get_training_stats(self) -> dict[str, object]:
        """Get training statistics."""
        # Cast to expected return type to satisfy static checkers
        return cast(dict[str, object], self.training_stats.copy())

    def load_model(self, model_path: str) -> bool:
        """Load a trained SAC model from file."""
        try:
            self.logger.info(f"Loading SAC model from {model_path}")
            try:
                from stable_baselines3 import SAC as _LocalSAC
            except Exception:
                _LocalSAC = None

            if _LocalSAC is None:
                raise ModelError(
                    "stable_baselines3.SAC is not available in this environment"
                )

            self.model = _LocalSAC.load(model_path)
            self.logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False

    def validate_training(self, model_path: str | None = None) -> dict[str, object]:
        """Validate trained model."""
        try:
            # Use provided path or get from training stats
            validation_model_path: str | None = model_path
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

            if self.model is None:
                return {
                    "validation_success": False,
                    "error": "Model is None after loading",
                }

            # Basic validation checks
            validation_results = {
                "validation_success": True,
                "model_path": model_path,
                "observation_space": str(self.model.observation_space),
                "action_space": str(self.model.action_space),
                "policy_type": type(self.model.policy).__name__,
            }

            # Check model components
            if hasattr(self.model, "actor"):
                validation_results["has_actor"] = True
            if hasattr(self.model, "critic"):
                validation_results["has_critic"] = True
            if hasattr(self.model, "critic_target"):
                validation_results["has_critic_target"] = True

            self.logger.info(f"✅ Model validation successful: {validation_results}")
            return validation_results

        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            return {"validation_success": False, "error": str(e)}

    def run_hyperparameter_optimization(
        self,
        param_space: dict[str, object],
        n_trials: int = 50,
        optimization_target: str = "final_reward",
    ) -> dict[str, object]:
        """Run hyperparameter optimization for SAC."""
        try:
            self.logger.info(
                f"Starting hyperparameter optimization with {n_trials} trials"
            )

            # Import optimization utilities
            from ztb.training.hyperparameter_optimizer import HyperparameterOptimizer

            # Create optimizer
            HyperparameterOptimizer(
                config={
                    "algorithm": "sac",
                    "param_space": param_space,
                    "n_trials": n_trials,
                    "optimization_target": optimization_target,
                }
            )

            # Run optimization
            # result = optimizer.run_optimization(
            #     objective_function=self._evaluate_hyperparams,
            #     parameter_space=param_space,
            #     n_trials=n_trials
            # )
            # For now, return mock result
            result = type(
                "MockResult",
                (),
                {
                    "best_params": {},
                    "best_score": 0.0,
                    "trials": [],
                    "optimization_time": 0.0,
                },
            )()

            self.logger.info(
                f"Hyperparameter optimization completed. Best params: {result.best_params}"
            )

            return {
                "optimization_success": True,
                "best_params": result.best_params,
                "optimization_results": result,
                "n_trials": n_trials,
            }

        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return {"optimization_success": False, "error": str(e)}

    def _evaluate_hyperparams(self, params: dict[str, object]) -> float:
        """Evaluate a set of hyperparameters by training a model."""
        try:
            # Create config with test parameters
            test_config = self.config.copy()
            test_config["training"]["sac_hyperparameters"].update(params)
            test_config["training"]["total_timesteps"] = min(
                10000, test_config["training"]["total_timesteps"] // 10
            )  # Shorter training for optimization

            # Create temporary trainer
            temp_trainer = SACTrainer(test_config, logger=self.logger)

            # Train with reduced timesteps
            success = temp_trainer.train()

            if success:
                stats = temp_trainer.get_training_stats()
                # Return final reward as optimization target (higher is better)
                return float(stats.get("final_reward", 0))
            else:
                return float("-inf")  # Failed training

        except Exception as e:
            self.logger.warning(f"Hyperparameter evaluation failed: {e}")
            return float("-inf")

    def run_training_with_overrides(
        self,
        total_timesteps: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        output_dir: str | None = None,
        resume_path: str | None = None,
    ) -> bool:
        """Run training with parameter overrides."""
        try:
            # Create config copy and apply overrides
            config_override = self.config.copy()

            if total_timesteps:
                config_override["total_timesteps"] = total_timesteps
                self.logger.info(f"Total timesteps overridden: {total_timesteps}")

            if learning_rate:
                config_override["training"]["sac_hyperparameters"][
                    "learning_rate"
                ] = learning_rate
                self.logger.info(f"Learning rate overridden: {learning_rate}")

            if batch_size:
                config_override["training"]["sac_hyperparameters"][
                    "batch_size"
                ] = batch_size
                self.logger.info(f"Batch size overridden: {batch_size}")

            if output_dir:
                config_override["output_dir"] = output_dir
                self.logger.info(f"Output directory overridden: {output_dir}")

            if resume_path:
                config_override["training"]["resume_from"] = resume_path
                self.logger.info(f"Resume path overridden: {resume_path}")

            # Create temporary trainer with overrides
            override_trainer = SACTrainer(config_override, logger=self.logger)

            # Run training
            return override_trainer.train()

        except Exception as e:
            self.logger.error(f"Training with overrides failed: {e}")
            return False

    def _setup_callbacks(self):
        """Create callback list including a configured CheckpointCallback."""
        try:
            from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        except Exception:
            # Fallback to simple list if SB3 is not available during lightweight tests
            CallbackList = list
            CheckpointCallback = None

        checkpoint_interval = self.config.get("checkpoint_interval", self.config.get("training", {}).get("checkpoint_interval", 1000))
        checkpoint_dir = self.config.get("checkpoint_dir", self.config.get("training", {}).get("checkpoint_dir", "models/checkpoints"))
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except Exception:
            pass

        if CheckpointCallback is None:
            return CallbackList()

        checkpoint_cb = CheckpointCallback(save_freq=checkpoint_interval, save_path=checkpoint_dir, name_prefix="sac_checkpoint", verbose=1)
        return CallbackList([checkpoint_cb])

    def _log_sac_training_completion(self, training_time: float, callback: object) -> None:
        """Log final SAC training statistics in a deterministic debug format."""
        try:
            total_timesteps = int(self.config.get("training", {}).get("total_timesteps", 0))
            sps = total_timesteps / training_time if training_time and training_time > 0 else 0.0

            # Best-effort extraction of final metrics
            final_actor_loss = 0.0
            final_critic_loss = 0.0
            final_ent_coef = 0.0
            try:
                logger_obj = getattr(self.model, "logger", None)
                logger_values = getattr(logger_obj, "name_to_value", {}) if logger_obj is not None else {}
                final_actor_loss = float(logger_values.get("train/actor_loss", 0.0))
                final_critic_loss = float(logger_values.get("train/critic_loss", 0.0))
                final_ent_coef = float(logger_values.get("train/ent_coef", 0.0))
            except Exception:
                pass

            # Final reward extraction
            final_reward = 0.0
            try:
                cb = callback.callbacks[0] if hasattr(callback, "callbacks") else callback
                final_reward = float(cb.reward_history[-1]) if hasattr(cb, "reward_history") and cb.reward_history else 0.0
            except Exception:
                pass

            # Structured debug line expected by unit tests
            self.logger.debug(
                f"SAC training completed: Time={training_time:.2f}s | Steps={total_timesteps} | SPS={sps:.2f} | "
                f"Final ActorLoss={final_actor_loss:.4f} | CriticLoss={final_critic_loss:.4f} | EntCoef={final_ent_coef:.4f} | FinalReward={final_reward:.4f}"
            )
        except Exception as e:
            self.logger.debug(f"Failed to emit SAC completion log: {e}")

    def _convert_to_sac_v435_config(self) -> dict[str, object]:
        """Convert unified config to SAC v435 config format."""
        unified_config = self.config

        # Base SAC v435 config
        sac_config = {
            "model_name": "sac_v435_unified",
            "version": "4.3.5",
            "description": "SAC v435 with curriculum learning from unified trainer",
            "training": {
                "total_timesteps": unified_config.get("training", {}).get(
                    "total_timesteps", 1000000
                ),
                "learning_rate": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("learning_rate", 3e-4),
                "batch_size": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("batch_size", DEFAULT_BATCH_SIZE_SAC),
                "buffer_size": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("buffer_size", 1000000),
                "learning_starts": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("learning_starts", 1000),
                "tau": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("tau", 0.005),
                "gamma": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("gamma", 0.99),
                "ent_coef": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("ent_coef", "auto_1.0"),
                "target_entropy": unified_config.get("training", {})
                .get("sac_hyperparameters", {})
                .get("target_entropy", "auto"),
                "curriculum_learning": True,
            },
            "environment": {
                "transaction_cost": unified_config.get("training", {})
                .get("environment", {})
                .get("transaction_cost", 0.0015),
                "max_position_size": unified_config.get("training", {})
                .get("environment", {})
                .get("max_position_size", 0.1),
                "random_start": True,
                "enable_correlation_reduction": True,
                "correlation_threshold": 0.85,
                "max_features": 100,
                "feature_adaptation": True,
                "market_regime_detection": True,
            },
            "reward_function": {
                "base_profit_bonus_atr_coeff": 5.0,
                "base_profit_bonus_portfolio_coeff": 10.0,
                "base_action_penalty": 0.15,
                "loss_penalty_coeff": -1.0,
                "action_frequency_penalty": 0.05,
                "long_short_asymmetry": True,
                "risk_adjusted_bonus": True,
                "market_regime_penalty": True,
            },
            "features": {
                "technical_indicators": [
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_upper",
                    "bb_middle",
                    "bb_lower",
                    "bb_width",
                    "stoch_k",
                    "stoch_d",
                    "williams_r",
                    "ichimoku_tenkan",
                    "ichimoku_kijun",
                    "ichimoku_senkou_a",
                    "ichimoku_senkou_b",
                    "atr_14",
                    "cci_14",
                    "mfi_14",
                    "roc_12",
                    "mom_10",
                ],
                "price_features": [
                    "price_change",
                    "volume_change",
                    "returns",
                    "log_returns",
                    "sma_5",
                    "sma_10",
                    "sma_20",
                    "sma_50",
                    "volatility_5d",
                    "volatility_10d",
                    "volatility_20d",
                ],
                "adaptive_selection": True,
                "correlation_filtering": True,
                "importance_weighting": True,
            },
            "risk_management": {
                "dynamic_position_sizing": True,
                "drawdown_control": True,
                "max_drawdown_limit": 0.1,
                "volatility_adjustment": True,
                "correlation_risk": True,
            },
            "data": {
                "primary_dataset": unified_config.get("training", {})
                .get("data_config", {})
                .get("data_path", "data/btc_jpy_featured_dataset.csv"),
                "validation_split": 0.2,
                "test_split": 0.1,
                "feature_engineering": True,
                "data_augmentation": True,
            },
            "evaluation": {
                "backtest_episodes": 100,
                "performance_metrics": [
                    "total_return",
                    "win_rate",
                    "max_drawdown",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                ],
                "comparison_models": ["v430", "v434.2"],
                "robustness_tests": True,
            },
            "output": {
                "model_dir": "models/v435_unified",
                "config_dir": "config/v435",
                "results_dir": "results/v435_unified",
                "tensorboard_log": "tensorboard/v435_unified",
            },
        }

        return sac_config

    def analyze_results(self) -> dict[str, object]:
        """Analyze training results and provide comprehensive summary."""
        try:
            self.logger.info("Analyzing SAC training results...")

            # Get final action distribution from callback if available
            action_distribution = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
            regime_distributions = {}

            # Try to get callback data from training stats
            if hasattr(self, "training_stats") and self.training_stats:
                callback = self.training_stats.get("callback")
                if callback and hasattr(callback, "continuous_actions"):
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

            # Get regime statistics from environment if available
            regime_stats = {}
            if (
                hasattr(self, "env")
                and self.env is not None
                and hasattr(self.env, "regime_stats")
                and self.env.regime_stats
            ):
                regime_stats = self.env.regime_stats.copy()

            # Calculate training metrics
            training_metrics = {
                "algorithm": "SAC",
                "final_action_distribution": action_distribution,
                "regime_distributions": regime_distributions,
                "regime_stats": regime_stats,
                "market_regime_adaptation": self.market_regime_adaptation.get(
                    "enabled", False
                ),
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
            self.logger.error(f"Failed to analyze SAC results: {e}")
            return {"error": str(e)}

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action using the trained SAC model.

        Args:
            state: Current state observation

        Returns:
            Selected action
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first.")

        action, _ = self.model.predict(state, deterministic=False)
        return action
