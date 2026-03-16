#!/usr/bin/env python3
"""
Configuration validation and management for Unified Trainer.
"""

import json
import os
import time
from typing import Any, Optional

from ztb.types.common import ConfigDict
from ztb.io.json_io import read_json
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.logging_utils import get_logger

class TrainingConfigValidator:
    """Enhanced configuration validator with detailed error reporting."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self, config: ConfigDict) -> tuple[bool, list[str], list[str]]:
        """
        Validate configuration comprehensively.

        Returns:
            tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Basic structure validation
        if not self._validate_basic_structure(config):
            return False, self.errors, self.warnings

        # Algorithm-specific validation
        algorithm = config.get("algorithm", "").lower()
        if algorithm == "sac":
            self._validate_sac_config(config)
        elif algorithm == "ppo":
            self._validate_ppo_config(config)
        else:
            self.errors.append(f"Unsupported algorithm: {algorithm}")

        # Common validations
        self._validate_data_config(config)
        self._validate_environment_config(config)
        self._validate_training_config(config)

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_basic_structure(self, config: ConfigDict) -> bool:
        """Validate basic configuration structure."""
        required_keys = ["algorithm"]
        for key in required_keys:
            if key not in config:
                self.errors.append(f"Missing required key: {key}")
                return False

        algorithm = config.get("algorithm", "").lower()
        if algorithm not in [
            "sac",
            "ppo",
            "base_ml",
            "iterative",
            "ensemble",
            "curriculum",
        ]:
            self.errors.append(f"Invalid algorithm: {algorithm}")
            return False

        return True

    def _validate_sac_config(self, config: ConfigDict):
        """Validate SAC-specific configuration."""
        sac_config = config.get("sac_hyperparameters", {})

        # Required hyperparameters
        required_params = {
            "learning_rate": (float, (0, 1)),
            "buffer_size": (int, (1000, 1000000)),
            "learning_starts": (int, (100, 100000)),
            "batch_size": (int, (32, 1024)),
        }

        for param, (param_type, value_range) in required_params.items():
            if param not in sac_config:
                self.errors.append(f"Missing SAC hyperparameter: {param}")
                continue

            value = sac_config[param]
            if not isinstance(value, param_type):
                self.errors.append(
                    f"SAC {param} must be {param_type.__name__}, got {type(value).__name__}"
                )
                continue

            if isinstance(value_range, tuple) and len(value_range) == 2:
                min_val, max_val = value_range
                if not (min_val <= value <= max_val):
                    self.warnings.append(
                        f"SAC {param}={value} is outside recommended range [{min_val}, {max_val}]"
                    )

        # Optional hyperparameters with defaults
        optional_params = {
            "tau": (0.005, (0.001, 0.1)),
            "gamma": (0.99, (0.8, 0.999)),
            "ent_coef": (0.01, (0.001, 0.1)),
        }

        for param, (default, value_range) in optional_params.items():
            if param in sac_config:
                value = sac_config[param]
                if isinstance(value_range, tuple) and len(value_range) == 2:
                    min_val, max_val = value_range
                    if not (min_val <= value <= max_val):
                        self.warnings.append(
                            f"SAC {param}={value} is outside recommended range [{min_val}, {max_val}]"
                        )

    def _validate_ppo_config(self, config: dict[str, Any]):
        """Validate PPO-specific configuration."""
        # PPO validation not yet implemented - add warnings
        self.warnings.append("PPO configuration validation not yet fully implemented")

    def _validate_data_config(self, config: dict[str, Any]):
        """Validate data configuration."""
        data_path = config.get("data_path", "btc_jpy_real_dataset.csv")

        # Check if data file exists
        if not os.path.exists(data_path):
            self.errors.append(f"Data file not found: {data_path}")
            return

        # Check if it's a CSV file
        if not data_path.endswith(".csv"):
            self.warnings.append(f"Data file should be CSV format: {data_path}")

        # Try to get file size
        try:
            file_size = os.path.getsize(data_path)
            if file_size < 1024:  # Less than 1KB
                self.warnings.append(
                    f"Data file seems very small ({file_size} bytes): {data_path}"
                )
        except OSError:
            self.warnings.append(f"Cannot access data file: {data_path}")

    def _validate_environment_config(self, config: dict[str, Any]):
        """Validate environment configuration."""
        env_config = config.get("environment", {})

        # Required environment parameters
        required_params = {
            "initial_balance": (float, (1000, 10000000)),
            "transaction_cost": (float, (0, 0.01)),
            "max_position_size": (float, (0.1, 2.0)),
        }

        for param, (param_type, value_range) in required_params.items():
            if param not in env_config:
                self.errors.append(f"Missing environment parameter: {param}")
                continue

            value = env_config[param]
            if not isinstance(value, param_type):
                self.errors.append(
                    f"Environment {param} must be {param_type.__name__}, got {type(value).__name__}"
                )
                continue

            if isinstance(value_range, tuple) and len(value_range) == 2:
                min_val, max_val = value_range
                if not (min_val <= value <= max_val):
                    self.warnings.append(
                        f"Environment {param}={value} is outside recommended range [{min_val}, {max_val}]"
                    )

    def _validate_training_config(self, config: dict[str, Any]):
        """Validate training configuration."""
        # Total timesteps
        total_timesteps = config["training"]["total_timesteps"]
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            self.errors.append(
                f"total_timesteps must be positive integer, got {total_timesteps}"
            )
        elif total_timesteps < 10000:
            self.warnings.append(
                f"total_timesteps={total_timesteps} is quite small, consider increasing for better training"
            )

        # Model name
        model_name = config.get("model_name", "")
        if not model_name:
            self.warnings.append("model_name not specified, using default")
        elif not isinstance(model_name, str):
            self.errors.append(
                f"model_name must be string, got {type(model_name).__name__}"
            )

class ConfigurationFileManager:
    """Configuration management with validation and enhancement."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        # Use TrainingConfigValidator for validation
        self.validator = TrainingConfigValidator(logger)

    def load_and_validate(
        self, config_path: str
    ) -> tuple[dict[str, Any] | None, bool, list[str], list[str]]:
        """
        Load configuration from file and validate it.

        Returns:
            tuple of (config, is_valid, errors, warnings)
        """
        try:
            config = read_json(config_path)

            self.logger.info(f"Loaded configuration from {config_path}")

            # Validate configuration
            is_valid, errors, warnings = self.validator.validate(config)

            if warnings:
                self.logger.warning(f"Configuration warnings: {len(warnings)}")
                for warning in warnings:
                    self.logger.warning(f"  - {warning}")

            if not is_valid:
                self.logger.error(
                    f"Configuration validation failed: {len(errors)} errors"
                )
                for error in errors:
                    self.logger.error(f"  - {error}")
                return None, False, errors, warnings

            # Enhance configuration with defaults
            enhanced_config = self._enhance_config(config)

            return enhanced_config, True, [], warnings

        except FileNotFoundError:
            error = f"Configuration file not found: {config_path}"
            self.logger.error(error)
            return None, False, [error], []
        except json.JSONDecodeError as e:
            error = f"Invalid JSON in configuration file: {e}"
            self.logger.error(error)
            return None, False, [error], []
        except Exception as e:
            error = f"Failed to load configuration: {e}"
            self.logger.error(error)
            return None, False, [error], []

    def _enhance_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Enhance configuration with sensible defaults and computed values."""
        enhanced = config.copy()

        # Ensure algorithm is lowercase
        if "algorithm" in enhanced:
            enhanced["algorithm"] = enhanced["algorithm"].lower()

        # Add default model name if not specified
        if "model_name" not in enhanced:
            algorithm = enhanced.get("algorithm", "unknown")
            timestamp = str(int(time.time()))  # Import time at top if needed
            enhanced["model_name"] = f"{algorithm}_{timestamp}"

        # Add default total_timesteps if not specified
        if "total_timesteps" not in enhanced:
            enhanced["total_timesteps"] = 50000

        return enhanced

    def save_config(self, config: dict[str, Any], file_path: str) -> bool:
        """Save configuration to file."""
        try:
            safe_json_dump(config, file_path, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
