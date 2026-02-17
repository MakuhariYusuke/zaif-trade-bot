#!/usr/bin/env python3
"""
config.py
Central configuration management for ZTB system
"""

import json
import os
from typing import TypeVar, cast, overload

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import safe_config_get, safe_to_bool, safe_to_float, safe_to_int
from ztb.io.json_io import read_json
from ztb.types.common import ObjectMap

logger = get_logger(__name__)

T = TypeVar("T")

_BOOL_ENV_KEYS = frozenset(
    {"ZTB_MEM_PROFILE", "ZTB_TEST_ISOLATION", "ZTB_ENABLE_PROFILING"}
)
_FLOAT_ENV_KEYS = frozenset(
    {"ZTB_CUDA_WARN_GB", "ZTB_MAX_MEMORY_GB", "ZTB_FLOAT_TOLERANCE"}
)
_INT_ENV_KEYS = frozenset({"ZTB_CHECKPOINT_INTERVAL", "ZTB_CACHE_SIZE"})


def _parse_json_value(value: str) -> object | None:
    """Parse JSON from text with safe fallback."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _coerce_json_container(
    raw_value: object, container_type: type[list[object]] | type[ObjectMap]
) -> object | None:
    """Coerce string/object to a list/dict container."""
    if isinstance(raw_value, container_type):
        return raw_value
    if isinstance(raw_value, str):
        parsed = _parse_json_value(raw_value)
        if isinstance(parsed, container_type):
            return parsed
    return None


def _convert_env_value(key: str, value: str) -> object:
    """Convert env-string to typed config value based on schema key."""
    if key in _BOOL_ENV_KEYS:
        return safe_to_bool(value, False)
    if key in _FLOAT_ENV_KEYS:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid float value for {key}: {value}") from exc
    if key in _INT_ENV_KEYS:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value for {key}: {value}") from exc
    return value


class ZTBConfig:
    """Central configuration management for all ZTB components"""

    # Configuration schema for validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "ZTB_MEM_PROFILE": {"type": "boolean"},
            "ZTB_CUDA_WARN_GB": {"type": "number", "minimum": 0},
            "ZTB_LOG_LEVEL": {
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            },
            "ZTB_CHECKPOINT_INTERVAL": {"type": "integer", "minimum": 1},
            "ZTB_MAX_MEMORY_GB": {"type": "number", "minimum": 0},
            "ZTB_TEST_ISOLATION": {"type": "boolean"},
            "ZTB_FLOAT_TOLERANCE": {"type": "number", "minimum": 0},
            "ZTB_MODEL_DIR": {"type": "string"},
            "ZTB_CACHE_SIZE": {"type": "integer", "minimum": 1},
            "ZTB_ENABLE_PROFILING": {"type": "boolean"},
        },
        "additionalProperties": True,
    }

    @overload
    def get(self, key: str) -> str | None:
        ...

    @overload
    def get(self, key: str, default: str) -> str:
        ...

    @overload
    def get(self, key: str, default: T) -> T:
        ...

    def get(self, key: str, default: str | T | None = None) -> str | T:
        """Get configuration value from environment variables"""
        return cast(str | T, os.getenv(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = os.getenv(key)
        return safe_to_bool(value, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        value = os.getenv(key)
        return safe_to_int(value, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value with type validation"""
        value = os.getenv(key)
        return safe_to_float(value, default)

    def log_config(self) -> None:
        """Log current configuration for debugging"""
        config_vars = [
            "ZTB_MEM_PROFILE",
            "ZTB_CUDA_WARN_GB",
            "ZTB_LOG_LEVEL",
            "ZTB_CHECKPOINT_INTERVAL",
            "ZTB_MAX_MEMORY_GB",
            "ZTB_TEST_ISOLATION",
            "ZTB_FLOAT_TOLERANCE",
        ]
        logger.info("Current ZTB Configuration:")
        for var in config_vars:
            value = os.getenv(var)
            if value is not None:
                logger.info(f"  {var}={value}")

    def validate_config(self) -> bool:
        """
        Validate current configuration against schema.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not JSONSCHEMA_AVAILABLE:
            logger.warning(
                "jsonschema not available, skipping configuration validation"
            )
            return True

        # Collect current configuration
        config_dict: ObjectMap = {}
        properties = cast(ObjectMap, self.CONFIG_SCHEMA.get("properties", {}))
        for key in properties:
            value = os.getenv(key)
            if value is not None:
                try:
                    config_dict[key] = _convert_env_value(key, value)
                except ValueError as e:
                    logger.error(str(e))
                    return False

        try:
            jsonschema.validate(instance=config_dict, schema=self.CONFIG_SCHEMA)
            logger.info("Configuration validation passed")
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            return False
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def get_validated_config(self) -> ObjectMap:
        """
        Get validated configuration as a dictionary.

        Returns:
            Dictionary of validated configuration values
        """
        if not self.validate_config():
            raise ValueError("Configuration validation failed")

        config: ObjectMap = {}
        properties = cast(ObjectMap, self.CONFIG_SCHEMA.get("properties", {}))
        for key in properties:
            value = os.getenv(key)
            if value is not None:
                try:
                    config[key] = _convert_env_value(key, value)
                except ValueError:
                    # Should not happen after validate_config, keep robust fallback.
                    config[key] = value

        return config

    def get_environment(self) -> str:
        """
        Get current environment (development, testing, production).

        Returns:
            Environment name
        """
        return self.get("ZTB_ENV", "development")

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == "development"

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.get_environment() == "testing"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == "production"

    def get_environment_config(self) -> ObjectMap:
        """
        Get environment-specific configuration overrides.

        Returns:
            Dictionary of environment-specific configuration
        """
        env = self.get_environment()

        # Base configuration for all environments
        config: ObjectMap = {
            "debug": self.is_development(),
            "log_level": "DEBUG" if self.is_development() else "INFO",
            "enable_profiling": self.is_development(),
            "strict_validation": not self.is_testing(),
        }

        # Environment-specific overrides
        if env == "testing":
            config.update(
                {
                    "cache_size": 64,  # Smaller cache for testing
                    "max_memory_gb": 2.0,  # Lower memory limit for testing
                    "test_isolation": True,
                }
            )
        elif env == "production":
            config.update(
                {
                    "cache_size": 512,  # Larger cache for production
                    "max_memory_gb": 16.0,  # Higher memory limit for production
                    "enable_profiling": False,  # Disable profiling in production
                }
            )

        return config

    def get_model_dir(self) -> str:
        """Get the base directory for model files."""
        return self.get("ZTB_MODEL_DIR", "models")

    def get_model_path(self, model_name: str) -> str:
        """Get full path for a specific model file."""
        return f"{self.get_model_dir()}/{model_name}"


def get_config_value(
    config_dict: ObjectMap, key: str, expected_type: type[T], default: T
) -> T:
    """
    Safely extract and convert configuration values from dict with type validation.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        expected_type: Expected type (str, int, float, bool, list, dict)
        default: Default value if key not found or conversion fails

    Returns:
        Converted value or default
    """
    raw_value = safe_config_get(config_dict, key, None)
    try:
        if raw_value is None:
            return default

        if expected_type is str:
            return cast(T, str(raw_value))
        elif expected_type is int:
            return (
                cast(T, int(raw_value))
                if isinstance(raw_value, (int, str))
                else default
            )
        elif expected_type is float:
            return (
                cast(T, float(raw_value))
                if isinstance(raw_value, (int, float, str))
                else default
            )
        elif expected_type is bool:
            if isinstance(raw_value, bool):
                return cast(T, raw_value)
            elif isinstance(raw_value, str):
                return cast(T, raw_value.lower() in ("true", "1", "yes", "on"))
            else:
                return default
        elif expected_type is list:
            parsed_list = _coerce_json_container(raw_value, list)
            if parsed_list is None:
                return default
            return cast(T, parsed_list)
        elif expected_type is dict:
            parsed_dict = _coerce_json_container(raw_value, dict)
            if parsed_dict is None:
                return default
            return cast(T, parsed_dict)
        else:
            return cast(T, raw_value)
    except (ValueError, TypeError):
        logger.warning(
            f"Failed to convert config value for {key}: {raw_value}, using default {default}"
        )
        return default


def get_config_list(
    config_dict: ObjectMap, key: str, default: list[object] | None = None
) -> list[object]:
    """
    Get a list configuration value.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default list value

    Returns:
        List value or default
    """
    if default is None:
        default = []
    return get_config_value(config_dict, key, list, default)


def get_config_dict(
    config_dict: ObjectMap, key: str, default: ObjectMap | None = None
) -> ObjectMap:
    """
    Get a dict configuration value.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default dict value

    Returns:
        Dict value or default
    """
    if default is None:
        default = {}
    return get_config_value(config_dict, key, dict, default)


def get_config_str(config_dict: ObjectMap, key: str, default: str = "") -> str:
    """
    Get a string configuration value.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default string value

    Returns:
        String value or default
    """
    return get_config_value(config_dict, key, str, default)


def get_config_int(config_dict: ObjectMap, key: str, default: int = 0) -> int:
    """
    Get an integer configuration value.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default integer value

    Returns:
        Integer value or default
    """
    return get_config_value(config_dict, key, int, default)


def get_config_float(
    config_dict: ObjectMap, key: str, default: float = 0.0
) -> float:
    """
    Get a float configuration value.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default float value

    Returns:
        Float value or default
    """
    return get_config_value(config_dict, key, float, default)


def get_config_bool(
    config_dict: ObjectMap, key: str, default: bool = False
) -> bool:
    """
    Get a boolean configuration value.

    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default boolean value

    Returns:
        Boolean value or default
    """
    return get_config_value(config_dict, key, bool, default)


# Global instance
config = ZTBConfig()


class TypedConfig:
    """型安全な設定管理クラス"""

    def __init__(self, **kwargs: object):
        """型安全な設定の初期化"""
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, f"_validate_{key}"):
                validator = getattr(self, f"_validate_{key}")
                if callable(validator):
                    try:
                        if not bool(validator(value)):
                            raise ValueError(f"Invalid value for {key}: {value}")
                    except Exception as exc:
                        raise ValueError(f"Invalid value for {key}: {value}") from exc
            setattr(self, key, value)

    def _validate_learning_rate(self, value: float) -> bool:
        """学習率のバリデーション"""
        return 0 < value < 1

    def _validate_batch_size(self, value: int) -> bool:
        """バッチサイズのバリデーション"""
        return value > 0

    def _validate_gamma(self, value: float) -> bool:
        """割引率のバリデーション"""
        return 0 <= value <= 1

    def _validate_total_timesteps(self, value: int) -> bool:
        """総タイムステップ数のバリデーション"""
        return value > 0

    def get_models(self) -> list[ObjectMap]:
        """Get ensemble model configurations."""
        # Default model configurations - can be overridden by config files
        return [
            {
                "path": self.__dict__.get(
                    "default_model_path", "models/trading_optimized_reward_v2_final.zip"
                ),
                "weight": self.__dict__.get("default_model_weight", 1.0),
                "feature_set": self.__dict__.get("default_feature_set", "full"),
            }
        ]

    def get_model_dir(self) -> str:
        """Get the base directory for model files."""
        return cast(str, self.__dict__.get("model_dir", "models"))

    def to_dict(self) -> ObjectMap:
        """設定を辞書形式に変換"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: ObjectMap) -> "TypedConfig":
        """辞書から設定を復元"""
        return cls(**data)


class ValidatedConfig(TypedConfig):
    """JSON Schemaベースの設定検証クラス"""

    SCHEMA = {
        "type": "object",
        "properties": {
            "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "batch_size": {"type": "integer", "minimum": 1},
            "gamma": {"type": "number", "minimum": 0, "maximum": 1},
            "total_timesteps": {"type": "integer", "minimum": 1},
            "default_model_path": {"type": "string"},
            "default_model_weight": {"type": "number", "minimum": 0},
            "default_feature_set": {"type": "string"},
        },
        "additionalProperties": True,
    }

    def __init__(self, **kwargs: object):
        """JSON Schema検証付き初期化"""
        super().__init__(**kwargs)
        self._validate_schema()

    def _validate_schema(self) -> None:
        """JSON Schemaを使って設定を検証"""
        try:
            import jsonschema

            config_dict = self.to_dict()
            jsonschema.validate(config_dict, self.SCHEMA)
        except ImportError:
            # jsonschema not available, skip validation
            logger.warning("jsonschema not available, skipping config validation")
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    @classmethod
    def from_json_file(cls, file_path: str) -> "ValidatedConfig":
        """JSONファイルから設定を読み込み検証"""
        data = read_json(file_path)
        return cls(**data)
