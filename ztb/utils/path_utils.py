#!/usr/bin/env python3
"""
path_utils.py
Path and directory utilities for ZTB system
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.

    This function finds the project root by looking for common markers
    like pyproject.toml, setup.py, or .git directory.

    Returns:
        Path to the project root directory
    """
    current = Path(__file__).resolve()

    # Look for project root markers
    for parent in [current] + list(current.parents):
        if any(
            (parent / marker).exists()
            for marker in [
                "pyproject.toml",
                "setup.py",
                ".git",
                "requirements.txt",
                "package.json",
            ]
        ):
            return parent

    # Fallback: assume we're in ztb/utils/, so go up 2 levels
    return current.parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path for chaining
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_path_join(*parts: str) -> Path:
    """
    Safely join path components, handling both strings and Paths.

    Args:
        *parts: Path components to join

    Returns:
        Combined path as Path object
    """
    if not parts:
        return Path()

    result = Path(parts[0])
    for part in parts[1:]:
        result = result / part

    return result


def get_relative_path(from_path: Path, to_path: Path) -> Path:
    """
    Get relative path from one path to another.

    Args:
        from_path: Source path
        to_path: Target path

    Returns:
        Relative path from source to target
    """
    try:
        return Path(os.path.relpath(str(to_path), str(from_path)))
    except ValueError:
        # Paths are on different drives (Windows)
        return to_path


def find_files_by_extension(directory: Path, extension: str) -> list[Path]:
    """
    Find all files with a specific extension in a directory recursively.

    Args:
        directory: Directory to search in
        extension: File extension (without dot, e.g., 'py', 'json')

    Returns:
        List of matching file paths
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    return list(directory.rglob(f"*{extension}"))


def get_cache_dir(subdir: Optional[str] = None) -> Path:
    """
    Get cache directory path, optionally with subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Cache directory path
    """
    cache_dir = get_project_root() / "cache"
    if subdir:
        cache_dir = cache_dir / subdir
    return ensure_dir(cache_dir)


def get_models_dir(subdir: Optional[str] = None) -> Path:
    """
    Get models directory path, optionally with subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Models directory path
    """
    models_dir = get_project_root() / "models"
    if subdir:
        models_dir = models_dir / subdir
    return ensure_dir(models_dir)


def get_file_dir(file_path: str) -> Path:
    """
    Get the directory containing a file.

    Args:
        file_path: Path to a file (can be relative or absolute)

    Returns:
        Directory containing the file
    """
    return Path(file_path).parent


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a path to its absolute, normalized form.

    Args:
        path: Path to resolve

    Returns:
        Resolved absolute path
    """
    return Path(path).resolve()


def get_config_dir(subdir: Optional[str] = None) -> Path:
    """
    Get config directory path, optionally with subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Config directory path
    """
    config_dir = get_project_root() / "config"
    if subdir:
        config_dir = config_dir / subdir
    return ensure_dir(config_dir)


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    Get data directory path, optionally with subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Data directory path
    """
    data_dir = get_project_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    return ensure_dir(data_dir)


def get_logs_dir(subdir: Optional[str] = None) -> Path:
    """
    Get logs directory path, optionally with subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Logs directory path
    """
    logs_dir = get_project_root() / "logs"
    if subdir:
        logs_dir = logs_dir / subdir
    return ensure_dir(logs_dir)


def get_tensorboard_dir(subdir: Optional[str] = None) -> Path:
    """
    Get tensorboard directory path, optionally with subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Tensorboard directory path
    """
    tb_dir = get_project_root() / "tensorboard"
    if subdir:
        tb_dir = tb_dir / subdir
    return ensure_dir(tb_dir)
