"""
Configuration loader with priority: CLI > ENV > YAML > defaults.

Supports merging configurations from multiple sources with proper precedence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from ztb.config.core.base import BaseConfigLoader
from ztb.config.schemas.zaif import GlobalConfig
from ztb.io.json_io import write_json
from ztb.io.yaml_io import read_yaml
from ztb.types.common import ObjectMap
from ztb.utils.path_utils import get_project_root

def _as_object_map(value: object) -> ObjectMap:
    if not isinstance(value, dict):
        return {}
    normalized: ObjectMap = {}
    for key, item in value.items():
        if isinstance(key, str):
            normalized[key] = item
    return normalized

class PriorityConfigLoader(BaseConfigLoader):
    """Configuration loader with source priority management."""

    def __init__(self) -> None:
        self.sources: dict[str, ObjectMap] = {
            "defaults": {},
            "yaml": {},
            "env": {},
            "cli": {},
        }
        self._file_cache: dict[str, ObjectMap] = {}
        self._file_mtimes: dict[str, float] = {}
        self.environment = os.getenv("ZTB_ENV", "development")

    def _safe_load(
        self, loader: Callable[..., ObjectMap], *args: object, **kwargs: object
    ) -> ObjectMap:
        try:
            return loader(*args, **kwargs)
        except Exception:
            return {}

    def load_config(self, *args: object, **kwargs: object) -> ObjectMap:
        """Load configuration from file with auto-detected format."""
        if args:
            return self.load_yaml(str(args[0]))
        _ = kwargs
        return {}

    def load_yaml(self, config_path: str) -> ObjectMap:
        """Load configuration from YAML file with caching."""
        path = Path(config_path)
        if not path.exists():
            return {}

        current_mtime = path.stat().st_mtime
        cached_mtime = self._file_mtimes.get(config_path, 0.0)
        if config_path in self._file_cache and current_mtime <= cached_mtime:
            return dict(self._file_cache[config_path])

        config = self._safe_load(self._load_yaml_impl, config_path)
        self._file_cache[config_path] = dict(config)
        self._file_mtimes[config_path] = current_mtime
        return config

    def load_yaml_with_env_fallback(self, base_config_path: str) -> ObjectMap:
        """Load YAML config with environment-specific fallback."""
        base_path = Path(base_config_path)
        if not base_path.is_absolute():
            base_path = get_project_root() / base_path

        env_config_path = (
            base_path.parent / f"{base_path.stem}.{self.environment}{base_path.suffix}"
        )
        if env_config_path.exists():
            env_config = self.load_yaml(str(env_config_path))
            if env_config:
                return env_config
        return self.load_yaml(str(base_path))

    def validate_config(
        self, config: ObjectMap, schema: Callable[..., object] | None = None
    ) -> ObjectMap:
        """Validate configuration against a pydantic schema."""
        schema_class = schema or GlobalConfig
        try:
            validated = schema_class(**config)
            if hasattr(validated, "model_dump"):
                return _as_object_map(validated.model_dump())
            return _as_object_map(validated)
        except ValidationError as exc:
            raise ValueError(f"Configuration validation failed: {exc}") from exc

    def _load_yaml_impl(self, config_path: str) -> ObjectMap:
        """Implementation of YAML config loading."""
        path = Path(config_path)
        if not path.is_absolute():
            path = get_project_root() / path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = read_yaml(path)
        config = _as_object_map(raw)
        self.sources["yaml"] = config
        return config

    def load_env(self, prefix: str = "ZTB_") -> ObjectMap:
        """Load configuration from environment variables."""
        return self._safe_load(self._load_env_impl, prefix)

    def _load_env_impl(self, prefix: str = "ZTB_") -> ObjectMap:
        """Implementation of environment config loading."""
        config: ObjectMap = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                clean_key = key[len(prefix) :].lower()
                keys = clean_key.split("_")
                self._set_nested_value(config, keys, value)
        self.sources["env"] = config
        return config

    def load_cli(self, args: ObjectMap) -> ObjectMap:
        """Load configuration from CLI arguments."""
        return self._safe_load(self._load_cli_impl, args)

    def _load_cli_impl(self, args: ObjectMap) -> ObjectMap:
        """Implementation of CLI config loading."""
        config: ObjectMap = {}
        for key, value in args.items():
            if value is None:
                continue
            keys = str(key).split(".")
            self._set_nested_value(config, keys, value)
        self.sources["cli"] = config
        return config

    def _set_nested_value(self, config: ObjectMap, keys: list[str], value: object) -> None:
        """set value in nested dictionary structure."""
        current = config
        for key in keys[:-1]:
            child = current.get(key)
            if not isinstance(child, dict):
                child = {}
                current[key] = child
            current = child
        current[keys[-1]] = value

    def merge_configs(self) -> ObjectMap:
        """Merge configurations with priority: CLI > ENV > YAML > defaults."""
        merged = dict(self.sources["defaults"])
        self._deep_merge(merged, self.sources["yaml"])
        self._deep_merge(merged, self.sources["env"])
        self._deep_merge(merged, self.sources["cli"])
        return merged

    def _deep_merge(self, base: ObjectMap, update: ObjectMap) -> None:
        """Deep merge update into base."""
        for key, value in update.items():
            base_value = base.get(key)
            if isinstance(base_value, dict) and isinstance(value, dict):
                self._deep_merge(base_value, value)
            else:
                base[key] = value

    def get_config(
        self,
        config_path: str | None = None,
        cli_args: ObjectMap | None = None,
    ) -> GlobalConfig:
        """Get validated GlobalConfig instance."""
        self.sources["defaults"] = _as_object_map(GlobalConfig().model_dump())

        if config_path:
            self.load_yaml(config_path)
        self.load_env()
        if cli_args:
            self.load_cli(cli_args)

        merged = self.merge_configs()
        try:
            return GlobalConfig(**merged)
        except ValidationError as exc:
            raise ValueError(f"Configuration validation failed: {exc}") from exc

    def dump_schema(self, output_path: str) -> None:
        """Dump JSON schema to file."""
        schema = GlobalConfig.model_json_schema()
        write_json(output_path, schema, indent=2, ensure_ascii=False)

ConfigLoader = PriorityConfigLoader

def load_config(
    config_path: str | None = None, cli_args: ObjectMap | None = None
) -> GlobalConfig:
    """Load configuration with default loader."""
    loader = ConfigLoader()
    config = loader.get_config(config_path, cli_args)
    initialize_risk_profiles(config)
    return config

def initialize_risk_profiles(config: GlobalConfig) -> None:
    """Initialize risk profile manager with config presets."""
    from ztb.trading.live.risk.profiles import get_risk_manager

    manager = get_risk_manager()
    for profile in config.risk_profiles.values():
        manager.add_profile(profile)
