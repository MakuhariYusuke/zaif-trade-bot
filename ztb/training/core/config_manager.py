#!/usr/bin/env python3
"""
Configuration Manager for Unified Training.

Handles all configuration-related operations including:
- Parameter extraction from config dict
- Unified config building
- Memory optimization settings
- Feature configuration
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from ztb.training.algorithms.sac.sac_algorithm import DEFAULT_SAC_CONFIG
from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
from ztb.utils.config_manager import BaseConfigManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingConfigManager(BaseConfigManager):
    """
    Manages configuration extraction and building for unified training.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ConfigManager.

        Args:
            config: Raw configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)

    def load_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Return the in-memory config for compatibility."""
        return self.config

    def save_config(self, *args, **kwargs) -> None:
        """TrainingConfigManager does not persist configs."""
        raise NotImplementedError("TrainingConfigManager does not support save_config")

    def _get_config_value(
        self, key: str, sections: Optional[List[str]] = None, default: Any = None
    ) -> Any:
        """
        Get configuration value with priority order.

        Priority: top-level > sections (in order) > default
        """
        # Check top-level first
        if key in self.config:
            return self.config[key]

        # Check specified sections
        if sections:
            for section in sections:
                section_data = self.config.get(section, {})
                if key in section_data:
                    return section_data[key]

        return default

    def get_memory_optimization_config(self) -> Dict[str, Optional[int]]:
        """
        Extract memory optimization parameters from config.

        These parameters control memory usage during training:
        - data_rows_limit: Maximum number of data rows to load
        - max_features: Maximum number of features to use (variance-based selection)

        Returns:
            Dict containing memory optimization settings
        """
        return {
            "data_rows_limit": self._get_config_value("data_rows_limit"),
            "max_features": self._get_config_value("max_features"),
        }

    def get_environment_config(self) -> Dict[str, Any]:
        """
        Extract environment-specific parameters from config.

        Returns:
            Dict containing environment settings like max_position_size,
            initial_balance, transaction_cost, etc.
        """
        # Check for nested training.environment structure
        training_env = deepcopy(self.config.get("training", {}).get("environment", {}))
        nested_env = (
            training_env.get("config", {}) if isinstance(training_env, dict) else {}
        )
        environment: Dict[str, Any] = {}

        if isinstance(nested_env, dict):
            environment.update(nested_env)

        # Include direct fields defined alongside the nested config (e.g. feature_set)
        if isinstance(training_env, dict):
            for key in [
                "feature_set",
                "csv_path",
                "use_continuous_actions",
                "enable_action_masking",
            ]:
                if key in training_env and key not in environment:
                    environment[key] = training_env[key]

        # Also check training.features section for feature_set
        training_features = self.config.get("training", {}).get("features", {})
        if isinstance(training_features, dict) and "feature_set" in training_features:
            environment["feature_set"] = training_features["feature_set"]

        # Apply top-level overrides/fallbacks
        environment["max_position_size"] = self._get_config_value(
            "max_position_size",
            ["environment"],
            environment.get(
                "max_position_size", DEFAULT_PPO_CONFIG.get("max_position_size", 1.0)
            ),
        )
        environment["initial_balance"] = self._get_config_value(
            "initial_balance",
            ["environment"],
            environment.get(
                "initial_balance", DEFAULT_PPO_CONFIG.get("initial_balance", 1000000)
            ),
        )
        environment["transaction_cost"] = self._get_config_value(
            "transaction_cost",
            ["environment"],
            environment.get(
                "transaction_cost", DEFAULT_PPO_CONFIG.get("transaction_cost", 0.001)
            ),
        )
        environment["reward_scaling"] = self._get_config_value(
            "reward_scaling",
            ["environment"],
            environment.get(
                "reward_scaling", DEFAULT_PPO_CONFIG.get("reward_scaling", 1.0)
            ),
        )
        environment["reward_clip_value"] = self._get_config_value(
            "reward_clip_value",
            ["environment"],
            environment.get(
                "reward_clip_value", DEFAULT_PPO_CONFIG.get("reward_clip_value", 10.0)
            ),
        )
        environment["use_continuous_actions"] = self._get_config_value(
            "use_continuous_actions",
            ["environment"],
            environment.get(
                "use_continuous_actions",
                DEFAULT_PPO_CONFIG.get("use_continuous_actions", False),
            ),
        )

        # Preserve reward_settings and behavior optimization blocks if present
        if "reward_settings" in nested_env and "reward_settings" not in environment:
            environment["reward_settings"] = nested_env["reward_settings"]
        if (
            "behavior_optimization" in nested_env
            and "behavior_optimization" not in environment
        ):
            environment["behavior_optimization"] = nested_env["behavior_optimization"]
        if "action_bonuses" in nested_env and "action_bonuses" not in environment:
            environment["action_bonuses"] = nested_env["action_bonuses"]

        # Add curriculum stage from curriculum_learning or environment section
        curriculum_learning = self.config.get("curriculum_learning", {})
        if (
            isinstance(curriculum_learning, dict)
            and "curriculum_stage" in curriculum_learning
        ):
            environment["curriculum_stage"] = curriculum_learning["curriculum_stage"]
        # Also check for curriculum_stage directly in environment section
        elif self._get_config_value("curriculum_stage", ["environment"]):
            environment["curriculum_stage"] = self._get_config_value(
                "curriculum_stage", ["environment"]
            )
        else:
            # Check top-level curriculum_learning
            top_curriculum = self.config.get("training", {}).get(
                "curriculum_learning", {}
            )
            if (
                isinstance(top_curriculum, dict)
                and "curriculum_stage" in top_curriculum
            ):
                environment["curriculum_stage"] = top_curriculum["curriculum_stage"]

        return environment

    def get_ppo_core_config(self) -> Dict[str, Any]:
        """
        Extract PPO algorithm-specific parameters from config.

        Returns:
            Dict containing PPO hyperparameters (learning_rate, n_steps, etc.)
        """
        return {
            "learning_rate": self._get_config_value(
                "learning_rate",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("learning_rate", 3e-4),
            ),
            "n_steps": self._get_config_value(
                "n_steps",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("n_steps", 1024),
            ),
            "batch_size": self._get_config_value(
                "batch_size",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("batch_size", 32),
            ),
            "n_epochs": self._get_config_value(
                "n_epochs",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("n_epochs", 10),
            ),
            "gamma": self._get_config_value(
                "gamma",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("gamma", 0.99),
            ),
            "gae_lambda": self._get_config_value(
                "gae_lambda",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("gae_lambda", 0.95),
            ),
            "clip_range": self._get_config_value(
                "clip_range",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("clip_range", 0.2),
            ),
            "clip_range_vf": self._get_config_value(
                "clip_range_vf", ["ppo_hyperparameters", "ppo"]
            ),
            "normalize_advantage": self._get_config_value(
                "normalize_advantage",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("normalize_advantage", True),
            ),
            "ent_coef": self._get_config_value(
                "ent_coef",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("ent_coef", 0.0),
            ),
            "vf_coef": self._get_config_value(
                "vf_coef",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("vf_coef", 0.5),
            ),
            "max_grad_norm": self._get_config_value(
                "max_grad_norm",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("max_grad_norm", 0.5),
            ),
            "use_sde": self._get_config_value(
                "use_sde",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("use_sde", False),
            ),
            "sde_sample_freq": self._get_config_value(
                "sde_sample_freq",
                ["ppo_hyperparameters", "ppo"],
                DEFAULT_PPO_CONFIG.get("sde_sample_freq", -1),
            ),
            "target_kl": self._get_config_value(
                "target_kl", ["ppo_hyperparameters", "ppo"]
            ),
            "policy_kwargs": self._get_config_value(
                "policy_kwargs", ["ppo_hyperparameters", "ppo"]
            ),
        }

    def get_sac_core_config(self) -> Dict[str, Any]:
        """
        Extract SAC (Soft Actor-Critic) algorithm-specific parameters from config.

        Returns:
            Dict containing SAC hyperparameters
        """
        sections = ["sac_hyperparameters", "sac"]
        return {
            "learning_rate": self._get_config_value(
                "learning_rate", sections, DEFAULT_SAC_CONFIG.get("learning_rate", 3e-4)
            ),
            "buffer_size": self._get_config_value(
                "buffer_size",
                sections,
                DEFAULT_SAC_CONFIG.get("buffer_size", 1_000_000),
            ),
            "learning_starts": self._get_config_value(
                "learning_starts",
                sections,
                DEFAULT_SAC_CONFIG.get("learning_starts", 100),
            ),
            "batch_size": self._get_config_value(
                "batch_size", sections, DEFAULT_SAC_CONFIG.get("batch_size", 256)
            ),
            "tau": self._get_config_value(
                "tau", sections, DEFAULT_SAC_CONFIG.get("tau", 0.005)
            ),
            "gamma": self._get_config_value(
                "gamma", sections, DEFAULT_SAC_CONFIG.get("gamma", 0.99)
            ),
            "train_freq": self._get_config_value(
                "train_freq",
                sections,
                DEFAULT_SAC_CONFIG.get("train_freq", (1, "step")),
            ),
            "gradient_steps": self._get_config_value(
                "gradient_steps",
                sections,
                DEFAULT_SAC_CONFIG.get("gradient_steps", 1),
            ),
            "ent_coef": self._get_config_value(
                "ent_coef", sections, DEFAULT_SAC_CONFIG.get("ent_coef", "auto")
            ),
            "target_update_interval": self._get_config_value(
                "target_update_interval",
                sections,
                DEFAULT_SAC_CONFIG.get("target_update_interval", 1),
            ),
            "target_entropy": self._get_config_value(
                "target_entropy",
                sections,
                DEFAULT_SAC_CONFIG.get("target_entropy", "auto"),
            ),
            "use_sde": self._get_config_value(
                "use_sde", sections, DEFAULT_SAC_CONFIG.get("use_sde", False)
            ),
            "sde_sample_freq": self._get_config_value(
                "sde_sample_freq",
                sections,
                DEFAULT_SAC_CONFIG.get("sde_sample_freq", -1),
            ),
            "use_sde_at_warmup": self._get_config_value(
                "use_sde_at_warmup",
                sections,
                DEFAULT_SAC_CONFIG.get("use_sde_at_warmup", False),
            ),
            "net_arch": self._get_config_value("net_arch", sections, None),
            "policy_kwargs": self._get_config_value("policy_kwargs", sections, None),
        }

    def get_curriculum_config(self) -> Dict[str, Any]:
        sections = ["curriculum_learning", "curriculum"]
        return {
            "curriculum_stage": self._get_config_value("curriculum_stage", sections, 0),
            "curriculum_threshold": self._get_config_value(
                "curriculum_threshold", sections, 0.0
            ),
            "curriculum_speed": self._get_config_value(
                "curriculum_speed", sections, 1.0
            ),
            "curriculum_min_level": self._get_config_value(
                "curriculum_min_level", sections, 0
            ),
            "curriculum_max_level": self._get_config_value(
                "curriculum_max_level", sections, 100
            ),
            "curriculum_level": self._get_config_value("curriculum_level", sections, 0),
        }

    def get_feature_config(self) -> Dict[str, Any]:
        """
        Extract feature-related parameters from config.

        Returns:
            Dict containing feature settings like feature_set, custom_features, etc.
        """
        # Check for nested training.features structure
        training_features = self.config.get("training", {}).get("features", {})

        return {
            "feature_set": self._get_config_value(
                "feature_set",
                ["features"],
                training_features.get("feature_set", "full"),
            ),
            "custom_features": self.config.get("custom_features", None),
            "feature_config_path": self.config.get("feature_config_path", None),
            "max_features": self.config.get("max_features", None),
        }

    def build_unified_config(
        self,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        total_timesteps_override: Optional[int] = None,
        debug_internal_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a unified configuration dict with all settings properly organized.

        Args:
            enable_streaming: Whether streaming is enabled
            stream_batch_size: Batch size for streaming
            total_timesteps_override: Override for total_timesteps
            debug_internal_state: Enable debugging of internal model states

        Returns:
            Unified config dict with structure:
            {
                "ppo": {PPO core hyperparameters},
                "memory_optimization": {data_rows_limit, max_features},
                "environment": {env-specific settings},
                ... (all other top-level settings for backward compatibility)
            }
        """
        # Build structured config
        ppo_core = self.get_ppo_core_config()
        sac_core = self.get_sac_core_config()
        memory_opt = self.get_memory_optimization_config()
        environment = self.get_environment_config()
        features = self.get_feature_config()

        # Extract total_timesteps from multiple possible locations
        # Priority: override > training section > top-level > ppo section > default
        total_timesteps = (
            total_timesteps_override
            or (self.config.get("training", {}) or {}).get("total_timesteps")
            or self.config.get("total_timesteps")
            or (self.config.get("ppo", {}) or {}).get("total_timesteps")
            or DEFAULT_PPO_CONFIG.get("total_timesteps", 100000)
        )

        # Build base unified structure first
        unified_base = {
            # Structured sections
            "ppo": {
                **ppo_core,
                **environment,  # PPOConfig expects these fields
                "total_timesteps": total_timesteps,
            },
            "sac": {
                **sac_core,
                **environment,  # SAC may also need some env fields
                "total_timesteps": total_timesteps,
            },
            "memory_optimization": memory_opt,
            "environment": environment,
            "features": features,
            "training": self.config.get("training", {}),  # Include training config
            # Performance settings
            "enable_streaming": enable_streaming,
            "stream_batch_size": stream_batch_size,
            "aggressive_memory_management": self.config.get(
                "aggressive_memory_management", False
            ),
            "enable_cuda_optimizations": self.config.get(
                "enable_cuda_optimizations", True
            ),
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
        unified["sac"]["total_timesteps"] = total_timesteps

        # Add debug flag to the environment config
        unified["environment"]["debug_internal_state"] = debug_internal_state

        return unified
