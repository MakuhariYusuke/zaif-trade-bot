"""Project setup utilities for consistent initialization."""

import sys
from pathlib import Path
from typing import Optional

from ztb.utils.path_utils import get_project_root


def setup_project_path(project_root: Optional[Path] = None) -> Path:
    """
    Setup project path and ensure it's in sys.path.

    Args:
        project_root: Optional project root path. If None, uses get_project_root().

    Returns:
        The project root path
    """
    if project_root is None:
        project_root = get_project_root()

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root


def get_project_root_from_file(file_path: Path, levels_up: int = 2) -> Path:
    """
    Get project root from a file path by going up specified levels.

    Args:
        file_path: The file path to start from
        levels_up: How many parent levels to go up (default: 2 for ztb/ structure)

    Returns:
        Project root path
    """
    return file_path.resolve().parents[levels_up]
