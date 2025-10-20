"""
Configuration utilities for ZTB.

This module provides standardized functions for loading and validating
configuration files in YAML, JSON, and TOML formats.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union, cast

import yaml

from ztb.utils.file_utils import safe_json_load

try:
    import tomli

    TOML_AVAILABLE = True
except ImportError:
    tomli = None
    TOML_AVAILABLE = False

try:
    import tomllib

    TOMLLIB_AVAILABLE = True
except ImportError:
    tomllib = None
    TOMLLIB_AVAILABLE = False

try:
    import tomli_w

    TOML_WRITE_AVAILABLE = True
except ImportError:
    tomli_w = None
    TOML_WRITE_AVAILABLE = False


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        file_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], yaml.safe_load(f))


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        file_path: Path to JSON configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    return cast(Dict[str, Any], safe_json_load(file_path))


def load_toml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        file_path: Path to TOML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        tomli.TOMLKitError or tomllib.TOMLDecodeError: If TOML parsing fails
        ImportError: If TOML library is not available
    """
    if not TOML_AVAILABLE and not TOMLLIB_AVAILABLE:
        raise ImportError("TOML support requires 'tomli' library (pip install tomli)")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "rb") as f:
        if TOMLLIB_AVAILABLE:
            import tomllib

            return cast(Dict[str, Any], tomllib.load(f))
        elif TOML_AVAILABLE and tomli is not None:
            return cast(Dict[str, Any], tomli.load(f))
        else:
            raise ImportError(
                "TOML support requires 'tomli' library (pip install tomli)"
            )


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file (auto-detect format from extension).

    Args:
        file_path: Path to configuration file (.yaml, .yml, .json, or .toml)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If file extension is not supported
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        return load_yaml_config(file_path)
    elif suffix == ".json":
        return load_json_config(file_path)
    elif suffix == ".toml":
        return load_toml_config(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}")


def find_config_file(
    config_name: str, search_paths: Optional[List[Path]] = None
) -> Optional[Path]:
    """
    Find configuration file in standard locations.

    Args:
        config_name: Name of config file (e.g., 'features.yaml')
        search_paths: Additional paths to search (optional)

    Returns:
        Path to config file if found, None otherwise
    """
    if search_paths is None:
        search_paths = [
            Path.cwd(),
            Path.cwd() / "config",
            Path(__file__).parent.parent / "config",
        ]

    for base_path in search_paths:
        config_path = base_path / config_name
        if config_path.exists():
            return config_path

    return None


class ConfigLoader:
    """
    Abstract configuration loader supporting multiple formats.

    This class provides a unified interface for loading and saving
    configuration files in different formats (YAML, JSON, TOML).
    """

    SUPPORTED_FORMATS = ["yaml", "yml", "json", "toml"]

    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file with auto-detected format.

        Args:
            file_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        return load_config(file_path)

    @staticmethod
    async def load_async(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Asynchronously load configuration from file with auto-detected format.

        Args:
            file_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_config, file_path)

    @staticmethod
    def save(
        config: Dict[str, Any],
        file_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            file_path: Path to save configuration file
            format: File format ('yaml', 'json', 'toml'). Auto-detected from extension if None.

        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)

        if format is None:
            suffix = file_path.suffix.lower().lstrip(".")
            if suffix not in ConfigLoader.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {suffix}")
            format = suffix

        if format not in ConfigLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            if format in ["yaml", "yml"]:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif format == "json":
                import json

                json.dump(config, f, indent=2, ensure_ascii=False)
            elif format == "toml":
                ConfigLoader._save_toml(config, f)

    @staticmethod
    def _save_toml(config: Dict[str, Any], file_obj: TextIO) -> None:
        """
        Save configuration as TOML.

        Args:
            config: Configuration dictionary
            file_obj: File object to write to

        Raises:
            ImportError: If TOML library is not available
        """
        if TOML_WRITE_AVAILABLE and tomli_w is not None:
            tomli_w.dump(config, file_obj)
        else:
            # Fallback: convert to basic TOML format
            ConfigLoader._write_basic_toml(config, file_obj)

    @staticmethod
    def _write_basic_toml(
        config: Dict[str, Any], file_obj: TextIO, prefix: str = ""
    ) -> None:
        """
        Write basic TOML format (fallback when tomli_w is not available).

        Args:
            config: Configuration dictionary
            file_obj: File object to write to
            prefix: Current key prefix for nested structures
        """
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                if prefix:
                    file_obj.write(f"\n[{full_key}]\n")
                else:
                    file_obj.write(f"[{key}]\n")
                ConfigLoader._write_basic_toml(value, file_obj, full_key)
            elif isinstance(value, list):
                # Simple array representation
                file_obj.write(f"{key} = {value!r}\n")
            else:
                # Simple value
                if isinstance(value, str):
                    file_obj.write(f'{key} = "{value}"\n')
                else:
                    file_obj.write(f"{key} = {value!r}\n")
