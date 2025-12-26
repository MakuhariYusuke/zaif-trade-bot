"""
Centralized Configuration Management System.

This module provides a unified interface for loading, validating, and managing
configuration across the ZTB system. It centralizes configuration handling
to ensure consistency and reduce duplication.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import yaml

from ztb.utils.exceptions.custom_exceptions import ConfigurationError, ValidationError
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseConfigManager(ABC):
    """Abstract base class for configuration managers."""

    @abstractmethod
    def load_config(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def save_config(self, *args, **kwargs) -> None:
        pass


class ConfigManager(BaseConfigManager):
    """
    Centralized configuration manager for ZTB system.

    Provides unified loading, validation, and saving of configuration files
    with support for YAML, JSON, and TOML formats.
    """

    SUPPORTED_FORMATS = {".yaml", ".yml", ".json", ".toml"}

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Base directory for configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_config(
        self, config_name: str, config_type: str = "general", validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from file with caching and validation.

        Args:
            config_name: Name of configuration file (without extension)
            config_type: Type of configuration for validation
            validate: Whether to validate the configuration

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        cache_key = f"{config_type}_{config_name}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        config_path = self._find_config_file(config_name)
        if not config_path:
            raise ConfigurationError(f"Configuration file '{config_name}' not found")

        try:
            config = self._load_config_file(config_path)
            if validate:
                self._validate_config(config, config_type)
            self._cache[cache_key] = config
            logger.info(f"Loaded configuration: {config_name}")
            return config.copy()
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration '{config_name}': {e}"
            ) from e

    def save_config(
        self,
        config: Dict[str, Any],
        config_name: str,
        config_type: str = "general",
        format: str = "yaml",
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            config_name: Name of configuration file (without extension)
            config_type: Type of configuration for validation
            format: File format ('yaml', 'json', 'toml')

        Raises:
            ConfigurationError: If configuration saving fails
        """
        self._validate_config(config, config_type)

        config_path = self.config_dir / f"{config_name}.{format}"
        try:
            self._save_config_file(config, config_path, format)
            cache_key = f"{config_type}_{config_name}"
            self._cache[cache_key] = config.copy()
            logger.info(f"Saved configuration: {config_name}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration '{config_name}': {e}"
            ) from e

    def _find_config_file(self, config_name: str) -> Optional[Path]:
        """Find configuration file with supported extensions."""
        for ext in self.SUPPORTED_FORMATS:
            config_path = self.config_dir / f"{config_name}{ext}"
            if config_path.exists():
                return config_path
        return None

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file based on extension."""
        suffix = config_path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            return self._load_yaml(config_path)
        elif suffix == ".json":
            return self._load_json(config_path)
        elif suffix == ".toml":
            return self._load_toml(config_path)
        else:
            raise ConfigurationError(f"Unsupported configuration format: {suffix}")

    def _save_config_file(
        self, config: Dict[str, Any], config_path: Path, format: str
    ) -> None:
        """Save configuration to file based on format."""
        if format == "yaml":
            self._save_yaml(config, config_path)
        elif format == "json":
            self._save_json(config, config_path)
        elif format == "toml":
            self._save_toml(config, config_path)
        else:
            raise ConfigurationError(f"Unsupported save format: {format}")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cast(Dict[str, Any], yaml.safe_load(f) or {})
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML config: {e}") from e

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load JSON config: {e}") from e

    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomli_fallback

                tomllib = tomli_fallback
            except ImportError:
                raise ConfigurationError(
                    "TOML support not available. Install tomli or tomllib"
                )

        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load TOML config: {e}") from e

    def _save_yaml(self, config: Dict[str, Any], path: Path) -> None:
        """Save YAML configuration."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save YAML config: {e}") from e

    def _save_json(self, config: Dict[str, Any], path: Path) -> None:
        """Save JSON configuration."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save JSON config: {e}") from e

    def _save_toml(self, config: Dict[str, Any], path: Path) -> None:
        """Save TOML configuration."""
        try:
            import tomli_w
        except ImportError:
            raise ConfigurationError(
                "TOML write support not available. Install tomli-w"
            )

        try:
            with open(path, "wb") as f:
                tomli_w.dump(config, f)
        except Exception as e:
            raise ConfigurationError(f"Failed to save TOML config: {e}") from e

    def _validate_config(self, config: Dict[str, Any], config_type: str) -> None:
        """
        Validate configuration based on type.

        Args:
            config: Configuration dictionary
            config_type: Type of configuration

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")

        # Type-specific validation
        if config_type == "training":
            self._validate_training_config(config)
        elif config_type == "trading":
            self._validate_trading_config(config)
        elif config_type == "model":
            self._validate_model_config(config)
        # Add more validation types as needed

    def _validate_training_config(self, config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        required_keys = ["learning_rate", "batch_size", "total_timesteps"]
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required training config key: {key}")

        if not (0 < config.get("learning_rate", 0) < 1):
            raise ValidationError("learning_rate must be between 0 and 1")

        if config.get("batch_size", 0) <= 0:
            raise ValidationError("batch_size must be positive")

        if config.get("total_timesteps", 0) <= 0:
            raise ValidationError("total_timesteps must be positive")

    def _validate_trading_config(self, config: Dict[str, Any]) -> None:
        """Validate trading configuration."""
        required_keys = ["initial_balance", "max_position_size"]
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required trading config key: {key}")

        if config.get("initial_balance", 0) <= 0:
            raise ValidationError("initial_balance must be positive")

        if config.get("max_position_size", 0) <= 0:
            raise ValidationError("max_position_size must be positive")

    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        required_keys = ["learning_rate", "batch_size"]
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required model config key: {key}")

        if not (0 < config.get("learning_rate", 0) < 1):
            raise ValidationError("learning_rate must be between 0 and 1")

        if config.get("batch_size", 0) <= 0:
            raise ValidationError("batch_size must be positive")

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._cache.clear()
        logger.info("Configuration cache cleared")

    def get_cached_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached configurations."""
        return {key: value.copy() for key, value in self._cache.items()}


def validate_config(config: Any, required_fields: list) -> bool:
    """
    設定オブジェクトの必須フィールドを検証

    Args:
        config: 検証する設定オブジェクト
        required_fields: 必須フィールド名のリスト

    Returns:
        bool: すべての必須フィールドが存在するかどうか
    """
    missing_fields = []
    for field in required_fields:
        if not hasattr(config, field):
            missing_fields.append(field)

    if missing_fields:
        logger.error(f"Missing required configuration fields: {missing_fields}")
        return False

    return True


def validate_dict_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    辞書型設定の必須キーを検証

    Args:
        config: 検証する設定辞書
        required_keys: 必須キーのリスト

    Returns:
        bool: すべての必須キーが存在するかどうか
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False

    return True
