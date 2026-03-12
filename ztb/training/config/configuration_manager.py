#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402
"""
Unified Configuration Management System for Training.

This module provides a comprehensive configuration management system that:
- Loads and validates configuration files (JSON/YAML)
- Supports configuration inheritance and merging
- Provides type-safe configuration access
- Integrates with environment variables and command-line arguments
- Offers configuration validation with detailed error reporting
"""

import json
import logging
import os
from dataclasses import dataclass, field

# from enum import Enum  # duplicate import removed
from pathlib import Path
from typing import Optional, Protocol, TypeVar, cast

import jsonschema
from ztb.config.schemas.zaif import DataConfig, EnvironmentConfig, TrainingConfig
from ztb.types.common import ConfigDict, ConfigValue

ConfigProfile = dict[str, ConfigValue]
ValidationResult = list[str]

# Type variables for generic functions
T = TypeVar("T")
ConfigType = TypeVar("ConfigType", bound=str)

class ValidatorFunc(Protocol):
    """Protocol for validation functions."""

    def __call__(self, value: ConfigValue) -> bool:
        ...

from enum import Enum

# Configuration classes for type safety
# DataConfig moved to ztb.config.schemas.zaif

class ConfigFormat(Enum):
    """Supported configuration file formats."""

    JSON = "json"
    YAML = "yaml"

class ValidationError(Exception):
    """Configuration validation error."""

    pass

class ConfigLoadError(Exception):
    """Configuration loading error."""

    pass

@dataclass
class ValidationRule:
    """Configuration validation rule."""

    field_path: str
    validator: ValidatorFunc
    error_message: str
    required: bool = True

@dataclass
class ConfigSchema:
    """Configuration schema definition."""

    schema: ConfigDict
    validators: list[ValidationRule] = field(default_factory=list)

    def validate(self, config: ConfigDict) -> ValidationResult:
        """Validate configuration against schema and rules."""
        errors: ValidationResult = []

        # JSON Schema validation
        try:
            jsonschema.validate(config, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")

        # Custom validation rules
        for rule in self.validators:
            value = self._get_nested_value(config, rule.field_path)
            if rule.required and value is None:
                errors.append(f"Required field '{rule.field_path}' is missing")
            elif value is not None and not rule.validator(value):
                errors.append(f"Field '{rule.field_path}': {rule.error_message}")

        return errors

    def _get_nested_value(self, config: ConfigDict, path: str) -> ConfigValue | None:
        """Get nested value from configuration using dot notation."""
        keys = path.split(".")
        current: ConfigDict | ConfigValue = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]  # type: ignore
            else:
                return None

        return cast(ConfigValue | None, current)

class ConfigurationManager:
    """
    Unified configuration management system.

    Features:
    - Multi-format configuration loading (JSON/YAML)
    - Configuration inheritance and merging
    - Environment variable integration
    - Command-line argument support
    - Type-safe configuration access
    - Comprehensive validation
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.schemas: dict[str, ConfigSchema] = {}
        self.loaded_configs: dict[str, ConfigDict] = {}
        self.profiles: dict[str, ConfigProfile] = {}
        self._register_default_schemas()
        self._register_default_profiles()
        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """Register default configuration schemas."""

        # Training configuration schema
        training_schema = ConfigSchema(
            schema={
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "training": {
                        "type": "object",
                        "properties": {
                            "model_name": {"type": "string"},
                            "algorithm": {
                                "type": "string",
                                "enum": ["sac", "ppo", "self_supervised"],
                            },
                            "total_timesteps": {"type": "integer", "minimum": 1},
                            "data_config": {
                                "type": "object",
                                "properties": {
                                    "data_path": {"type": "string"},
                                    "use_real_data": {"type": "boolean"},
                                },
                                "required": ["data_path"],
                            },
                            "environment": {
                                "type": "object",
                                "properties": {
                                    "initial_balance": {"type": "number", "minimum": 0},
                                    "transaction_cost": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "max_position_size": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                            },
                        },
                        "required": ["algorithm", "total_timesteps"],
                    },
                },
                "required": ["version", "training"],
            },
            validators=[
                ValidationRule(
                    "training.total_timesteps",
                    lambda x: isinstance(x, int) and x > 0,
                    "Total timesteps must be a positive integer",
                ),
                ValidationRule(
                    "training.algorithm",
                    lambda x: x in ["sac", "ppo", "self_supervised"],
                    "Algorithm must be one of: sac, ppo, self_supervised",
                ),
                ValidationRule(
                    "training.data_config.data_path",
                    lambda x: isinstance(x, str) and len(x) > 0,
                    "Data path must be a non-empty string",
                ),
            ],
        )

        self.schemas["training"] = training_schema

    def _register_default_profiles(self) -> None:
        """Register default configuration profiles."""
        # SAC algorithm profile
        self.profiles["sac_default"] = {
            "algorithm": "sac",
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_starts": 1000,
        }

        # PPO algorithm profile
        self.profiles["ppo_default"] = {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }

        # Self-supervised learning profile
        self.profiles["ssp_default"] = {
            "algorithm": "self_supervised",
            "learning_rate": 1e-3,
            "batch_size": 32,
            "num_epochs": 100,
            "latent_dim": 128,
            "num_layers": 3,
        }

        # Development environment profile
        self.profiles["dev"] = {
            "total_timesteps": 10000,
            "verbose": 1,
            "log_interval": 1,
        }

        # Production environment profile
        self.profiles["prod"] = {
            "total_timesteps": 1000000,
            "verbose": 0,
            "log_interval": 100,
        }

    def detect_environment(self) -> str:
        """Detect the current environment (dev, staging, prod)."""
        # Check environment variables
        env = os.environ.get("ENV", "").lower()
        if env in ["dev", "development"]:
            return "dev"
        elif env in ["staging", "stage"]:
            return "staging"
        elif env in ["prod", "production"]:
            return "prod"

        # Check hostname patterns
        import socket

        hostname = socket.gethostname().lower()
        if any(pattern in hostname for pattern in ["dev", "local", "laptop"]):
            return "dev"
        elif any(pattern in hostname for pattern in ["staging", "stage"]):
            return "staging"
        elif any(pattern in hostname for pattern in ["prod", "production"]):
            return "prod"

        # Default to development
        return "dev"

    def get_environment_profiles(self, environment: str | None = None) -> list[str]:
        """Get recommended profiles for the current environment."""
        if environment is None:
            environment = self.detect_environment()

        base_profiles = []

        # Add algorithm-specific profiles based on environment
        if environment == "dev":
            base_profiles.extend(["dev"])
        elif environment == "staging":
            base_profiles.extend(["ppo_default", "dev"])  # More conservative settings
        elif environment == "prod":
            base_profiles.extend(["sac_default", "prod"])  # Optimized for production

        return base_profiles

    def update_config_value(
        self, config: ConfigDict, path: str, value: ConfigValue
    ) -> ConfigDict:
        """Update a configuration value at the specified path."""
        keys = path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}  # type: ignore
            current = current[key]  # type: ignore

        # set the value
        current[keys[-1]] = value  # type: ignore
        self.logger.debug(f"Updated config value: {path} = {value}")
        return config

    def reload_config(
        self, config_path: str | Path, config_type: str = "training"
    ) -> ConfigDict:
        """Reload configuration from file, clearing any cached values."""
        config_path_str = str(config_path)
        if config_path_str in self.loaded_configs:
            del self.loaded_configs[config_path_str]

        return self.load_config(config_path, config_type)

    def create_typed_config(self, config_dict: ConfigDict) -> TrainingConfig:
        """Create a typed TrainingConfig from a configuration dictionary."""
        try:
            # Extract nested configurations
            # Use helpers that perform runtime validation and type narrowing
            from ztb.utils.config_helpers import (
                get_bool,
                get_dict,
                get_numeric,
                get_string,
            )

            training_section = get_dict(config_dict, "training")
            data_config_dict = get_dict(config_dict, "training.data_config")
            env_config_dict = get_dict(config_dict, "training.environment")
            training_dict = training_section

            # Create typed objects
            data_config = DataConfig(
                data_path=get_string(data_config_dict, "data_path"),
                use_real_data=get_bool(data_config_dict, "use_real_data", True),
            )

            # Filter env_config_dict to only include fields that EnvironmentConfig accepts
            # This prevents initialization errors from unknown keys like 'initial_balance'
            from dataclasses import fields as dataclass_fields

            valid_env_keys = {f.name for f in dataclass_fields(EnvironmentConfig)}
            filtered_env_config_dict = {
                k: v for k, v in env_config_dict.items() if k in valid_env_keys
            }
            env_config = EnvironmentConfig(
                initial_balance=get_numeric(
                    cast(ConfigDict, filtered_env_config_dict),
                    "initial_balance",
                    10000.0,
                ),
                transaction_cost=get_numeric(
                    cast(ConfigDict, filtered_env_config_dict),
                    "transaction_cost",
                    0.0015,
                ),
                max_position_size=get_numeric(
                    cast(ConfigDict, filtered_env_config_dict), "max_position_size", 1.0
                ),
            )

            # Create training config
            training_config = TrainingConfig(
                model_name=get_string(training_dict, "model_name", "default_model"),
                algorithm=get_string(training_dict, "algorithm", "sac"),
                total_timesteps=int(
                    get_numeric(config_dict, "training.total_timesteps", 100000)
                ),
                data_config=data_config,
                environment=env_config,
            )

            return training_config

        except Exception as e:
            self.logger.error(f"Failed to create typed config: {e}")
            # Return default config on error
            return TrainingConfig(model_name="default", algorithm="sac")

    def load_config(
        self,
        config_path: str | Path,
        config_type: str = "training",
        overrides: ConfigDict | None = None,
        env_prefix: str | None = None,
        profiles: list[str] | None = None,
    ) -> ConfigDict:
        """
        Load and validate configuration.

        Args:
            config_path: Path to configuration file
            config_type: Type of configuration (for schema validation)
            overrides: Configuration overrides
            env_prefix: Environment variable prefix (e.g., "TRAINING_")
            profiles: list of profile names to apply (e.g., ["sac_default", "dev"])

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigLoadError: If configuration cannot be loaded
            ValidationError: If configuration validation fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {config_path}")

        # Load configuration file
        config = self._load_config_file(config_path)

        # Apply profiles
        if profiles:
            config = self._apply_profiles(config, profiles, config_type)

        # Apply environment variable overrides
        if env_prefix:
            config = self._apply_env_overrides(config, env_prefix, config_type)

        # Apply runtime overrides
        if overrides:
            config = self._deep_merge(config, overrides)

        # Validate configuration
        if config_type in self.schemas:
            errors = self.schemas[config_type].validate(config)
            if errors:
                error_msg = "Configuration validation failed:\n" + "\n".join(
                    f"  - {error}" for error in errors
                )
                raise ValidationError(error_msg)

        # Cache loaded configuration
        self.loaded_configs[str(config_path)] = config

        self.logger.info(f"Configuration loaded successfully: {config_path}")
        return config

    def _load_config_file(self, config_path: Path) -> ConfigDict:
        """Load configuration from file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() in [".json"]:
                    return cast(ConfigDict, json.load(f))
                else:
                    raise ConfigLoadError(
                        f"Unsupported file format: {config_path.suffix}"
                    )
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load configuration file: {e}")

    def _apply_env_overrides(
        self, config: ConfigDict, prefix: str, config_type: str = "training"
    ) -> ConfigDict:
        """Apply environment variable overrides to configuration."""
        result = config.copy()

        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                config_key = env_key[len(prefix) :].lower()

                # For training config, assume training section
                if config_type == "training":
                    config_path = ["training", config_key]

                # Convert environment variable value to appropriate type
                typed_value = self._parse_env_value(env_value)

                # set nested configuration value
                self._set_nested_value(result, config_path, typed_value)
                self.logger.debug(
                    f"Applied environment override: {env_key} = {typed_value}"
                )

        return result

    def _apply_profiles(
        self, config: ConfigDict, profiles: list[str], config_type: str
    ) -> ConfigDict:
        """Apply configuration profiles to configuration."""
        result = config.copy()

        for profile_name in profiles:
            if profile_name in self.profiles:
                profile = self.profiles[profile_name]
                # Apply profile to appropriate section based on config_type
                if config_type == "training":
                    if "training" not in result:
                        result["training"] = {}
                    result["training"] = cast(
                        ConfigDict,
                        self._deep_merge(
                            cast(ConfigDict, result["training"]),
                            cast(ConfigDict, profile),
                        ),
                    )  # type: ignore
                else:
                    result = self._deep_merge(result, cast(ConfigDict, profile))
                self.logger.debug(f"Applied profile: {profile_name}")
            else:
                self.logger.warning(f"Profile not found: {profile_name}")

        return result

    def _parse_env_value(self, value: str) -> ConfigValue:
        """Parse environment variable value to appropriate type with enhanced type inference."""
        # Handle None/null values
        if value.lower() in ["none", "null", ""]:
            return None

        # Try boolean (expanded patterns)
        if value.lower() in ["true", "false", "1", "0", "yes", "no", "on", "off"]:
            return value.lower() in ["true", "1", "yes", "on"]

        # Try integer
        try:
            return cast(ConfigValue, int(value))
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try JSON parsing for complex values
        try:
            import json

            parsed = json.loads(value)
            return cast(ConfigValue, parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try list parsing (comma-separated)
        if "," in value:
            return cast(ConfigValue, [item.strip() for item in value.split(",")])

        # Default to string
        return cast(ConfigValue, value)

    def _set_nested_value(
        self, config: ConfigDict, path: list[str], value: ConfigValue
    ) -> None:
        """set nested configuration value using path list."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _deep_merge(self, base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """Deep merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)  # type: ignore
            else:
                result[key] = value  # type: ignore

        return result

    def get_config_value(
        self, config: ConfigDict, path: str, default: ConfigValue | None = None
    ) -> ConfigValue | None:
        """
        Get configuration value using dot notation.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "training.algorithm")
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        keys = path.split(".")
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return cast(ConfigValue | None, current)

    def validate_config_file(
        self, config_path: str | Path, config_type: str = "training"
    ) -> ValidationResult:
        """
        Validate configuration file without loading it.

        Returns:
            list of validation errors (empty if valid)
        """
        try:
            config_path = Path(config_path)

            if not config_path.exists():
                return [f"Configuration file not found: {config_path}"]

            # Load configuration file
            config = self._load_config_file(config_path)

            # Validate configuration
            if config_type in self.schemas:
                errors = self.schemas[config_type].validate(config)
                if errors:
                    return errors

            return []
        except Exception as e:
            return [f"Configuration validation failed: {e}"]

    def create_config_template(
        self, config_type: str, output_path: str | Path | None = None
    ) -> ConfigDict:
        """
        Create configuration template from schema.

        Args:
            config_type: Type of configuration
            output_path: Optional path to save template

        Returns:
            Configuration template dictionary
        """
        if config_type not in self.schemas:
            raise ValueError(f"Unknown configuration type: {config_type}")

        # Create template from schema (simplified implementation)
        template = {
            "version": "1.0",
            "training": {
                "model_name": "example_model",
                "algorithm": "sac",
                "total_timesteps": 100000,
                "data_config": {"data_path": "data/dataset.csv", "use_real_data": True},
                "environment": {
                    "initial_balance": 10000.0,
                    "transaction_cost": 0.0015,
                    "max_position_size": 1.0,
                },
            },
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration template saved: {output_path}")

        return cast(ConfigDict, template)

    def add_profile(self, name: str, profile: ConfigProfile) -> None:
        """Add a custom configuration profile."""
        self.profiles[name] = profile
        self.logger.info(f"Profile added: {name}")

    def remove_profile(self, name: str) -> bool:
        """Remove a configuration profile."""
        if name in self.profiles:
            del self.profiles[name]
            self.logger.info(f"Profile removed: {name}")
            return True
        return False

    def list_profiles(self) -> list[str]:
        """list available configuration profiles."""
        return list(self.profiles.keys())

    def get_profile(self, name: str) -> ConfigProfile | None:
        """Get a configuration profile by name."""
        return self.profiles.get(name)

    def list_available_schemas(self) -> list[str]:
        """list available configuration schema types."""
        return list(self.schemas.keys())

    def add_custom_schema(self, name: str, schema: ConfigSchema) -> None:
        """Add custom configuration schema."""
        self.schemas[name] = schema
        self.logger.info(f"Custom schema added: {name}")

# Global configuration manager instance
config_manager = ConfigurationManager()

def load_training_config(
    config_path: str | Path,
    overrides: ConfigDict | None = None,
    env_prefix: str = "TRAINING_",
    profiles: list[str] | None = None,
) -> ConfigDict:
    """
    Convenience function to load training configuration.

    Args:
        config_path: Path to configuration file
        overrides: Configuration overrides
        env_prefix: Environment variable prefix

    Returns:
        Validated training configuration
    """
    return config_manager.load_config(
        config_path, "training", overrides, env_prefix, profiles
    )

def validate_config_file(config_path: str | Path) -> ValidationResult:
    """
    Convenience function to validate configuration file.

    Returns:
        list of validation errors (empty if valid)
    """
    return config_manager.validate_config_file(config_path, "training")

def create_config_template(
    output_path: str | Path | None = None,
) -> ConfigDict:
    """Create training configuration template."""
    return config_manager.create_config_template("training", output_path)

def create_typed_training_config(
    config_path: str | Path,
    overrides: ConfigDict | None = None,
    env_prefix: str = "TRAINING_",
    profiles: list[str] | None = None,
) -> TrainingConfig:
    """Create a typed training configuration from file."""
    config_dict = load_training_config(config_path, overrides, env_prefix, profiles)
    return config_manager.create_typed_config(config_dict)
