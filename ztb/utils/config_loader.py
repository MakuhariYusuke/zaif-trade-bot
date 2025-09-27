"""
Configuration utilities for ZTB.

This module provides standardized functions for loading and validating
configuration files in YAML and JSON formats.
"""

import yaml
import json
from pathlib import Path
from typing import Optional, List
from typing import Dict, Any, Union, Optional


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

    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file (auto-detect format from extension).

    Args:
        file_path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If file extension is not supported
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in ['.yaml', '.yml']:
        return load_yaml_config(file_path)
    elif suffix == '.json':
        return load_json_config(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}")


def find_config_file(
    config_name: str,
    search_paths: Optional[List[Path]] = None
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
            Path.cwd() / 'config',
            Path(__file__).parent.parent / 'config'
        ]

    for base_path in search_paths:
        config_path = base_path / config_name
        if config_path.exists():
            return config_path

    return None