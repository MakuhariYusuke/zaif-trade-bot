"""
Configuration loader with priority: CLI > ENV > YAML > defaults.

Supports merging configurations from multiple sources with proper precedence.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from pydantic import ValidationError

from ztb.config.core.base import BaseConfigLoader
from ztb.config.schemas.zaif import GlobalConfig
from ztb.io.json_io import write_json
from ztb.io.yaml_io import read_yaml
from ztb.utils.path_utils import get_project_root


class PriorityConfigLoader(BaseConfigLoader):
    """Configuration loader with source priority management."""

    def __init__(self) -> None:
        self.sources: Dict[str, Dict[str, Any]] = {
            "defaults": {},
            "yaml": {},
            "env": {},
            "cli": {},
        }
        # Cache for file-based configurations
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._file_mtimes: Dict[str, float] = {}
        # Environment support
        self.environment = os.getenv("ZTB_ENV", "development")

    def load_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Load configuration from file with auto-detected format."""
        if args:
            file_path = args[0]
            return self.load_yaml(file_path)
        return {}

    def load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with caching."""
        path = Path(config_path)

        # If file doesn't exist, return empty dict
        if not path.exists():
            return {}

        # Check if file has been modified
        current_mtime = path.stat().st_mtime
        cached_mtime = self._file_mtimes.get(config_path, 0)

        if config_path in self._file_cache and current_mtime <= cached_mtime:
            # Return cached result
            return dict(self._file_cache[config_path])

        # Load fresh configuration
        config: Dict[str, Any] = {}
        try:
            loaded_raw = self._load_yaml_impl(config_path)
            loaded = cast(Any, loaded_raw)
            if isinstance(loaded, dict):
                # mypy should see concrete Dict[str, Any]
                config = {str(k): v for k, v in loaded.items()}
            else:
                config = {}
        except Exception:
            # On any error return empty config
            config = {}

        # Cache the result (store a shallow copy)
        self._file_cache[config_path] = dict(config)
        self._file_mtimes[config_path] = current_mtime

        return config

    def load_yaml_with_env_fallback(self, base_config_path: str) -> Dict[str, Any]:
        """Load YAML config with environment-specific fallback.

        Tries to load {base_config_path}.{environment}.yaml first,
        then falls back to {base_config_path}.yaml.

        Args:
            base_config_path: Base configuration path (relative to project root if not absolute)
        """
        # Resolve path relative to project root if not absolute
        base_path = Path(base_config_path)
        if not base_path.is_absolute():
            base_path = get_project_root() / base_path

        env_config_path = (
            base_path.parent / f"{base_path.stem}.{self.environment}{base_path.suffix}"
        )
        base_config_full_path = base_path

        # Try environment-specific config first
        if env_config_path.exists():
            config = self.load_yaml(str(env_config_path))
            if config:
                return config

        # Fallback to base config
        return self.load_yaml(str(base_config_full_path))

    def validate_config(
        self, config: Dict[str, Any], schema: Any = None
    ) -> Dict[str, Any]:
        """Validate configuration against a schema.

        Args:
            config: Configuration dictionary to validate
            schema: Pydantic model class for validation (optional)

        Returns:
            Validated configuration dictionary

        Raises:
            ValidationError: If configuration doesn't match schema
        """
        if schema is None:
            schema = GlobalConfig

        try:
            validated = schema(**config)
            # model_dump() is typed as Any by pydantic in some versions; cast to a concrete dict
            return cast(Dict[str, Any], validated.model_dump())
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    def _load_yaml_impl(self, config_path: str) -> Dict[str, Any]:
        """Implementation of YAML config loading."""
        path = Path(config_path)

        # Resolve path relative to project root if not absolute
        if not path.is_absolute():
            path = get_project_root() / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = read_yaml(path)

        # Normalize to dict[str, Any] and ensure keys are strings
        config: Dict[str, Any] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(k, str):
                    config[k] = v

        self.sources["yaml"] = config
        return config

    def load_env(self, prefix: str = "ZTB_") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        try:
            return self._load_env_impl(prefix)
        except Exception:
            return {}

    def _load_env_impl(self, prefix: str = "ZTB_") -> Dict[str, Any]:
        """Implementation of environment config loading."""
        config: Dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                clean_key = key[len(prefix) :].lower()
                keys = clean_key.split("_")
                self._set_nested_value(config, keys, value)

        self.sources["env"] = config
        return config

    def load_cli(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from CLI arguments."""
        try:
            return self._load_cli_impl(args)
        except Exception:
            return {}

    def _load_cli_impl(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of CLI config loading."""
        # Convert flat CLI args to nested structure
        config: Dict[str, Any] = {}
        for key, value in args.items():
            if value is not None:
                keys = key.split(".")
                self._set_nested_value(config, keys, value)

        self.sources["cli"] = config
        return config

    def _set_nested_value(
        self, config: Dict[str, Any], keys: List[str], value: Any
    ) -> None:
        """Set value in nested dictionary structure."""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def merge_configs(self) -> Dict[str, Any]:
        """Merge configurations with priority: CLI > ENV > YAML > defaults."""
        # Start with defaults
        merged = dict(self.sources["defaults"])

        # Merge YAML
        self._deep_merge(merged, self.sources["yaml"])

        # Merge ENV
        self._deep_merge(merged, self.sources["env"])

        # Merge CLI (highest priority)
        self._deep_merge(merged, self.sources["cli"])

        return merged

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge update into base."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_config(
        self,
        config_path: Optional[str] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ) -> GlobalConfig:
        """Get validated GlobalConfig instance."""
        # Load defaults
        self.sources["defaults"] = GlobalConfig().model_dump()

        # Load YAML if provided
        if config_path:
            self.load_yaml(config_path)

        # Load ENV
        self.load_env()

        # Load CLI if provided
        if cli_args:
            self.load_cli(cli_args)

        # Merge and validate
        merged: Dict[str, Any] = self.merge_configs()
        try:
            return GlobalConfig(**merged)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    def dump_schema(self, output_path: str) -> None:
        """Dump JSON schema to file."""
        schema = GlobalConfig.model_json_schema()
        write_json(output_path, schema, indent=2, ensure_ascii=False)


ConfigLoader = PriorityConfigLoader


def load_config(
    config_path: Optional[str] = None, cli_args: Optional[Dict[str, Any]] = None
) -> GlobalConfig:
    """Load configuration with default loader."""
    loader = ConfigLoader()
    config = loader.get_config(config_path, cli_args)

    # Initialize risk profiles
    initialize_risk_profiles(config)

    return config


def initialize_risk_profiles(config: GlobalConfig) -> None:
    """Initialize risk profile manager with config presets."""
    from ztb.trading.live.risk.profiles import get_risk_manager

    manager = get_risk_manager()
    for profile in config.risk_profiles.values():
        manager.add_profile(profile)
