#!/usr/bin/env python3
"""
Configuration converter for v4XX series compatibility.

Converts legacy configuration formats to unified trainer format.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class V4XXConfigConverter:
    """Configuration converter for v4XX series compatibility."""

    @staticmethod
    def convert_v427_to_unified(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v427-style configuration to unified trainer format.

        Args:
            config: v427 configuration dictionary

        Returns:
            Unified trainer configuration dictionary
        """
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "model_name": config.get("model_name", "sac_v427_converted"),
            "version": "4.2.7",
            "training": {
                "data_config": {
                    "data_path": config.get(
                        "data_path", "data/btc_jpy_real_dataset.csv"
                    ),
                    "validation_split": config.get("validation_split", 0.2),
                    "test_split": config.get("test_split", 0.1),
                },
                "total_timesteps": config.get("total_timesteps", 10000),
                "sac_hyperparameters": config.get("sac_hyperparameters", {}),
                "environment": config.get("environment", {}),
                "reward_function": config.get("reward_settings", {}),
                "checkpoint_dir": config.get(
                    "checkpoint_dir", "models/training_states"
                ),
            },
        }

        # Ensure required SAC hyperparameters
        sac_params = unified_config["training"]["sac_hyperparameters"]
        defaults = {
            "learning_rate": 0.0003,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "target_entropy": -2.0,
        }

        for key, default_value in defaults.items():
            if key not in sac_params:
                sac_params[key] = default_value

        # Ensure required environment parameters
        env_params = unified_config["training"]["environment"]
        env_defaults = {
            "initial_balance": 200000.0,
            "transaction_cost": 1e-05,
            "max_position_size": 1.0,
            "random_start": True,
        }

        for key, default_value in env_defaults.items():
            if key not in env_params:
                env_params[key] = default_value

        logger.info("Converted v427 configuration to unified format")
        return unified_config

    @staticmethod
    def convert_v440_to_unified(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v440-style configuration to unified trainer format.

        Args:
            config: v440 configuration dictionary

        Returns:
            Unified trainer configuration dictionary
        """
        unified_config = {
            "algorithm": "sac",
            "model_name": config.get("model_name", "sac_v440_converted"),
            "version": "4.4.0",
            "training": {
                "data_config": {
                    "data_path": config.get(
                        "data_path", "data/btc_jpy_real_dataset.csv"
                    ),
                    "validation_split": 0.2,
                    "test_split": 0.1,
                },
                "total_timesteps": config.get("total_timesteps", 50000),
                "sac_hyperparameters": config.get("sac_hyperparameters", {}),
                "environment": config.get("environment", {}),
                "reward_function": config.get("reward_function", {}),
                "checkpoint_dir": "models/training_states",
            },
        }

        # Set v440-specific defaults
        sac_params = unified_config["training"]["sac_hyperparameters"]
        if not sac_params:
            sac_params.update(
                {
                    "learning_rate": 3e-4,
                    "batch_size": 256,
                    "buffer_size": 1000000,
                    "learning_starts": 1000,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "ent_coef": "auto_1.0",
                    "target_entropy": "auto",
                }
            )

        env_params = unified_config["training"]["environment"]
        if not env_params:
            env_params.update(
                {
                    "initial_balance": 10000,
                    "transaction_cost": 0.0,
                    "max_position_size": 1.0,
                    "random_start": True,
                }
            )

        logger.info("Converted v440 configuration to unified format")
        return unified_config

    @staticmethod
    def convert_v444_to_unified(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v444-style configuration to unified trainer format.

        Args:
            config: v444 configuration dictionary

        Returns:
            Unified trainer configuration dictionary
        """
        # Start with the base config structure
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "model_name": config.get("model_name", "sac_v444_converted"),
            "version": "4.4.4",
            "training": {
                "data_config": {
                    "data_path": config.get("data_path", "data/btc_jpy_featured_dataset.csv"),
                    "validation_split": config.get("validation_split", 0.2),
                    "test_split": config.get("test_split", 0.1),
                },
                "total_timesteps": config.get("total_timesteps", 10000),
                "sac_hyperparameters": config.get("sac_hyperparameters", {}),
                "environment": {},
                "reward_function": config.get("reward_function", {}),
                "checkpoint_dir": config.get("checkpoint_dir", "models/training_states"),
            },
        }

        # Copy environment settings and expand behavior_optimization and action_bonuses
        env_config = config.get("environment", {})
        unified_env = unified_config["training"]["environment"]

        # Copy all environment settings
        for key, value in env_config.items():
            if key not in ["behavior_optimization", "action_bonuses"]:
                unified_env[key] = value

        # Expand behavior_optimization and action_bonuses to top level for environment access
        if "behavior_optimization" in env_config:
            unified_env.update(env_config["behavior_optimization"])
            logger.info("Expanded behavior_optimization parameters to environment config")

        if "action_bonuses" in env_config:
            unified_env.update(env_config["action_bonuses"])
            logger.info("Expanded action_bonuses parameters to environment config")

        # Handle regime_adaptation if present
        if "regime_adaptation" in config:
            if "config" not in unified_env:
                unified_env["config"] = {}
            unified_env["config"]["advanced_market_regime"] = config["regime_adaptation"]
            logger.info("Mapped regime_adaptation to training.environment.config.advanced_market_regime")

        # Extract curriculum_stage from training.curriculum_learning and add to environment config
        # This is CRITICAL for balance_penalty to work correctly
        training_config = config.get("training", {})
        curriculum_learning = training_config.get("curriculum_learning", {})
        if "curriculum_stage" in curriculum_learning:
            curriculum_stage = curriculum_learning["curriculum_stage"]
            unified_env["curriculum_stage"] = curriculum_stage
            logger.info(f"Mapped curriculum_stage '{curriculum_stage}' to training.environment config")
        
        # Also ensure curriculum_learning is preserved for other components that may need it
        if "curriculum_learning" in training_config:
            unified_config["training"]["curriculum_learning"] = training_config["curriculum_learning"]
            logger.info("Preserved curriculum_learning configuration in training section")

        # Ensure required SAC hyperparameters
        sac_params = unified_config["training"]["sac_hyperparameters"]
        defaults = {
            "learning_rate": 0.0003,
            "buffer_size": 1000000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "target_update_interval": 1,
        }

        for key, default_value in defaults.items():
            if key not in sac_params:
                sac_params[key] = default_value

        logger.info("Converted v444 configuration to unified format")
        return unified_config

    @staticmethod
    def detect_config_version(config: Dict[str, Any]) -> str:
        """
        Detect configuration version from structure.

        Args:
            config: Configuration dictionary

        Returns:
            Version string (e.g., "v427", "v435", "v440", "v444")
        """
        # Check for v444 structure (has environment section with behavior_optimization and action_bonuses)
        if ("environment" in config and
            isinstance(config["environment"], dict) and
            "behavior_optimization" in config["environment"] and
            "action_bonuses" in config["environment"]):
            return "v444"

        # Check for v435 structure (has training section)
        if "training" in config and "sac_hyperparameters" in config["training"]:
            return "v435"

        # Check for v427 structure (top-level sac_hyperparameters)
        if "sac_hyperparameters" in config and "environment" in config:
            return "v427"

        # Check for v440 structure (minimal config, often from results)
        if "config_version" in config and config["config_version"] == "4.4.0":
            return "v440"

        return "unknown"

    @classmethod
    def convert_to_unified(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-detect and convert configuration to unified format.

        Args:
            config: Input configuration dictionary

        Returns:
            Unified trainer configuration dictionary
        """
        version = cls.detect_config_version(config)

        if version == "v444":
            logger.info("Converting v444 configuration to unified format")
            unified_config = cls.convert_v444_to_unified(config)
            return unified_config
        elif version == "v435":
            logger.info("Configuration already in unified format (v435)")
            # Ensure regime_adaptation is mapped to advanced_market_regime for environment
            if "regime_adaptation" in config and "advanced_market_regime" not in config:
                # Add to training.environment.config section for environment access
                if "training" not in config:
                    config["training"] = {}
                if "environment" not in config["training"]:
                    config["training"]["environment"] = {}
                if "config" not in config["training"]["environment"]:
                    config["training"]["environment"]["config"] = {}
                config["training"]["environment"]["config"][
                    "advanced_market_regime"
                ] = config["regime_adaptation"]
                logger.info(
                    "Mapped regime_adaptation to training.environment.config.advanced_market_regime for environment compatibility"
                )
            return config
        elif version == "v427":
            return cls.convert_v427_to_unified(config)
        elif version == "v440":
            return cls.convert_v440_to_unified(config)
        else:
            logger.warning(
                f"Unknown configuration version, assuming v435 format: {version}"
            )
            # Ensure regime_adaptation is mapped to advanced_market_regime for environment
            if "regime_adaptation" in config and "advanced_market_regime" not in config:
                # Add to training.environment.config section for environment access
                if "training" not in config:
                    config["training"] = {}
                if "environment" not in config["training"]:
                    config["training"]["environment"] = {}
                if "config" not in config["training"]["environment"]:
                    config["training"]["environment"]["config"] = {}
                config["training"]["environment"]["config"][
                    "advanced_market_regime"
                ] = config["regime_adaptation"]
                logger.info(
                    "Mapped regime_adaptation to training.environment.config.advanced_market_regime for environment compatibility"
                )
            return config

    @classmethod
    def load_and_convert_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration file and convert to unified format.

        Args:
            config_path: Path to configuration file

        Returns:
            Unified trainer configuration dictionary
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            return cls.convert_to_unified(config)

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")


def convert_config_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert configuration file to unified format and save.

    Args:
        input_path: Input configuration file path
        output_path: Output file path (optional, defaults to input_path + "_unified")

    Returns:
        Path to converted configuration file
    """
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(
            input_path_obj.parent
            / f"{input_path_obj.stem}_unified{input_path_obj.suffix}"
        )

    unified_config = V4XXConfigConverter.load_and_convert_config(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unified_config, f, indent=2, ensure_ascii=False)

    logger.info(f"Converted configuration saved to: {output_path}")
    return output_path
