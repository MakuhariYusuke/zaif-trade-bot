#!/usr/bin/env python3
"""
file_utils.py
File I/O utilities for ZTB system
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def safe_json_load(file_path: Path, default: Any = None) -> Any:
    """
    Safely load JSON from a file with error handling.

    Args:
        file_path: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        # If default is callable, call it to get the default value
        if callable(default):
            return default()
        return default
    except Exception as e:
        logger.error(f"Unexpected error loading JSON from {file_path}: {e}")
        # If default is callable, call it to get the default value
        if callable(default):
            return default()
        return default


def safe_json_dump(
    data: Any, file_path: Union[str, Path], indent: int = 2, default: Any = None
) -> bool:
    """
    Safely dump data to JSON file with error handling.

    Args:
        data: Data to serialize
        file_path: Path to save the JSON file
        indent: JSON indentation level
        default: Default function for objects that can't be serialized

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to Path if it's a string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=default)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def load_config_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load configuration from a JSON file.

    Args:
        file_path: Path to the config file

    Returns:
        Configuration dictionary or None if failed
    """
    config = safe_json_load(file_path)
    if config is None:
        return None

    if not isinstance(config, dict):
        logger.warning(f"Config file {file_path} does not contain a dictionary")
        return None

    return config


def save_config_file(config: Dict[str, Any], file_path: Path) -> bool:
    """
    Save configuration to a JSON file.

    Args:
        config: Configuration dictionary
        file_path: Path to save the config file

    Returns:
        True if successful, False otherwise
    """
    return safe_json_dump(config, file_path)


def read_text_file(file_path: Path, encoding: str = "utf-8") -> Optional[str]:
    """
    Read text content from a file.

    Args:
        file_path: Path to the file
        encoding: Text encoding

    Returns:
        File content as string or None if failed
    """
    try:
        return file_path.read_text(encoding=encoding)
    except Exception as e:
        logger.error(f"Failed to read text file {file_path}: {e}")
        return None


def write_text_file(content: str, file_path: Path, encoding: str = "utf-8") -> bool:
    """
    Write text content to a file.

    Args:
        content: Text content to write
        file_path: Path to the file
        encoding: Text encoding

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception as e:
        logger.error(f"Failed to write text file {file_path}: {e}")
        return False


def append_text_file(content: str, file_path: Path, encoding: str = "utf-8") -> bool:
    """
    Append text content to a file.

    Args:
        content: Text content to append
        file_path: Path to the file
        encoding: Text encoding

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to append to text file {file_path}: {e}")
        return False


def file_exists_and_not_empty(file_path: Path) -> bool:
    """
    Check if a file exists and is not empty.

    Args:
        file_path: Path to check

    Returns:
        True if file exists and has content
    """
    if not file_path.exists():
        return False

    try:
        return file_path.stat().st_size > 0
    except Exception:
        return False


def get_file_size(file_path: Path) -> Optional[int]:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes or None if failed
    """
    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return None


def backup_file(file_path: Path, suffix: str = ".backup") -> Optional[Path]:
    """
    Create a backup of a file.

    Args:
        file_path: Original file path
        suffix: Backup file suffix

    Returns:
        Backup file path or None if failed
    """
    if not file_path.exists():
        logger.warning(f"Cannot backup non-existent file: {file_path}")
        return None

    backup_path = file_path.with_suffix(file_path.suffix + suffix)

    try:
        import shutil

        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup file {file_path}: {e}")
        return None
