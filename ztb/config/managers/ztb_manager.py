"""
Centralized configuration management.
"""

from typing import Any, Optional

from ztb.config.loaders.priority_loader import PriorityConfigLoader
from ztb.config.schemas.zaif import UnifiedConfigLoader, ZaifTradeBotConfig
from ztb.utils.config_manager import BaseConfigManager
from ztb.utils.path_utils import get_project_root


class ZaifTradeBotConfigManager(BaseConfigManager):
    """Configuration manager singleton."""

    _instance: Optional["ZaifTradeBotConfigManager"] = None
    _config: Optional[ZaifTradeBotConfig] = None

    def __init__(self) -> None:
        if ZaifTradeBotConfigManager._instance is not None:
            raise RuntimeError("ZaifTradeBotConfigManager is a singleton")
        ZaifTradeBotConfigManager._instance = self
        self.loader = PriorityConfigLoader()

    @classmethod
    def get_instance(cls) -> "ZaifTradeBotConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_config(self, config_path: Optional[str] = None) -> ZaifTradeBotConfig:
        """Load and merge configuration from all sources."""
        if config_path is None:
            # Try to find config files in standard locations
            config_path = self._find_default_config_path()

        if config_path:
            self.loader.load_yaml(config_path)

        self.loader.load_env()
        # CLI args would be loaded here if available

        merged = self.loader.merge_configs()
        self._config = ZaifTradeBotConfig(**merged)
        return self._config

    def _find_default_config_path(self) -> Optional[str]:
        """Find default configuration file path."""
        project_root = get_project_root()
        search_paths = [
            project_root / "config" / "config.yaml",
            project_root / "config" / "default.yaml",
            project_root / "ztb_config.yaml",
        ]

        for config_path in search_paths:
            if config_path.exists():
                return str(config_path)

        return None

    def save_config(self, config_path: str, format: str = "yaml") -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise RuntimeError("No configuration loaded to save")

        # Use unified config loader for saving
        UnifiedConfigLoader.save_config(self._config, config_path, format)

    def create_default_config(
        self, config_path: str = "config/default.yaml"
    ) -> ZaifTradeBotConfig:
        """Create and save a default configuration file."""
        # Create default config with all default values
        config = ZaifTradeBotConfig()
        UnifiedConfigLoader.save_config(config, config_path, "yaml")
        return config

    def discover_config_files(self) -> list[str]:
        """Discover all available configuration files in standard locations."""
        project_root = get_project_root()
        config_files = []

        # Search in config directory (non-recursive, exclude cache/temp dirs)
        config_dir = project_root / "config"
        if config_dir.exists():
            for file_path in config_dir.iterdir():
                if file_path.is_file() and file_path.suffix in [
                    ".yaml",
                    ".yml",
                    ".json",
                ]:
                    # Skip files in cache/temp directories
                    if any(
                        part.startswith((".", "__"))
                        or part in ("node_modules", "venv", "venv311", "venv313")
                        for part in file_path.parts
                    ):
                        continue
                    config_files.append(str(file_path))

        # Search in project root for specific config files
        root_config_patterns = [
            "ztb_config.yaml",
            "ztb_config.yml",
            "ztb_config.json",
            "config.yaml",
            "config.yml",
            "config.json",
        ]
        for pattern in root_config_patterns:
            config_path = project_root / pattern
            if config_path.exists():
                config_files.append(str(config_path))

        return sorted(list(set(config_files)))

    def validate_config(self, config_path: Optional[str] = None) -> bool:
        """Validate configuration file."""
        try:
            if config_path:
                # Load and validate specific config file
                UnifiedConfigLoader.load_config(config_path)
            else:
                # Validate current loaded config
                if self._config is None:
                    self.load_config()
                # If we get here without exception, config is valid
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def get_config(self) -> ZaifTradeBotConfig:
        """Get current configuration."""
        if self._config is None:
            self.load_config()
        assert self._config is not None
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        config = self.get_config()
        keys = key.split(".")
        value = config.model_dump()
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        # This would update the config, but for simplicity, just update the dict
        config_dict = self.get_config().model_dump()
        keys = key.split(".")
        d = config_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self._config = ZaifTradeBotConfig(**config_dict)


# Global instance
config_manager = ZaifTradeBotConfigManager.get_instance()
