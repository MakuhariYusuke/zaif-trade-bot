"""
Configuration loader with priority: CLI > ENV > YAML > defaults.

Supports merging configurations from multiple sources with proper precedence.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from .schema import GlobalConfig


class ConfigLoader:
    """Configuration loader with source priority management."""

    def __init__(self):
        self.sources = {"defaults": {}, "yaml": {}, "env": {}, "cli": {}}

    def load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        self.sources["yaml"] = config
        return config

    def load_env(self, prefix: str = "ZTB_") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
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
        # Convert flat CLI args to nested structure
        config = {}
        for key, value in args.items():
            if value is not None:
                keys = key.split(".")
                self._set_nested_value(config, keys, value)

        self.sources["cli"] = config
        return config

    def _set_nested_value(self, config: Dict[str, Any], keys: list, value: Any):
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

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge update into base."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_config(
        self, config_path: Optional[str] = None, cli_args: Optional[Dict] = None
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
        merged = self.merge_configs()
        try:
            return GlobalConfig(**merged)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    def dump_schema(self, output_path: str):
        """Dump JSON schema to file."""
        schema = GlobalConfig.model_json_schema()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            import json

            json.dump(schema, f, indent=2, ensure_ascii=False)


# Convenience function
def load_config(
    config_path: Optional[str] = None, cli_args: Optional[Dict] = None
) -> GlobalConfig:
    """Load configuration with default loader."""
    loader = ConfigLoader()
    config = loader.get_config(config_path, cli_args)

    # Initialize risk profiles
    initialize_risk_profiles(config)

    return config


def initialize_risk_profiles(config: GlobalConfig):
    """Initialize risk profile manager with config presets."""
    from ztb.live.risk_profiles import get_risk_manager

    manager = get_risk_manager()
    for profile in config.risk_profiles.values():
        manager.add_profile(profile)
