#!/usr/bin/env python3
"""
Path Manager for Analysis Components

Provides centralized path management for analysis-related file operations.
Ensures consistent directory structures and path resolution across components.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PathManagerError(Exception):
    """Exception raised when path management fails."""

    pass

class AnalysisPathManager:
    """Centralized path manager for analysis components."""

    def __init__(
        self, base_dir: str | Path | None = None, create_dirs: bool = True
    ):
        """
        Initialize path manager.

        Args:
            base_dir: Base directory for analysis operations
            create_dirs: Whether to create directories automatically
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.create_dirs = create_dirs
        self.logger = logging.getLogger(self.__class__.__name__)

        # Define standard directory structure
        self._init_standard_paths()

        if create_dirs:
            self._ensure_directories_exist()

    def _init_standard_paths(self) -> None:
        """Initialize standard path definitions."""
        self.paths = {
            # Data directories
            "data": self.base_dir / "data",
            "backtest_experiments": self.base_dir / "backtest_experiments",
            "results": self.base_dir / "results",
            "models": self.base_dir / "models",
            # Analysis directories
            "analysis": self.base_dir / "analysis",
            "analysis_results": self.base_dir / "analysis_results",
            "reports": self.base_dir / "reports",
            # Training directories
            "tensorboard": self.base_dir / "tensorboard",
            "checkpoints": self.base_dir / "checkpoints",
            "training_results": self.base_dir / "training_results",
            # Output directories
            "plots": self.base_dir / "plots",
            "exports": self.base_dir / "exports",
            "temp": self.base_dir / "temp",
        }

    def _ensure_directories_exist(self) -> None:
        """Ensure all standard directories exist."""
        for path_name, path in self.paths.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to create directory {path}: {e}")

    def get_path(self, name: str) -> Path:
        """
        Get a standard path by name.

        Args:
            name: Name of the path (e.g., 'analysis', 'results')

        Returns:
            Path object for the requested location

        Raises:
            PathManagerError: If path name is not recognized
        """
        if name not in self.paths:
            available_paths = list(self.paths.keys())
            raise PathManagerError(
                f"Unknown path name '{name}'. Available: {available_paths}"
            )
        return self.paths[name]

    def resolve_experiment_path(
        self, experiment_name: str, create_if_missing: bool = False
    ) -> Path:
        """
        Resolve path for a specific experiment.

        Args:
            experiment_name: Name of the experiment
            create_if_missing: Whether to create the directory if it doesn't exist

        Returns:
            Path to the experiment directory
        """
        experiment_path = self.get_path("backtest_experiments") / experiment_name

        if create_if_missing and not experiment_path.exists():
            experiment_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created experiment directory: {experiment_path}")

        return experiment_path

    def find_latest_experiment_dir(self, experiment_name: str) -> Path | None:
        """
        Find the latest (most recent) experiment directory.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the latest experiment directory, or None if not found
        """
        experiment_base = self.resolve_experiment_path(experiment_name)

        if not experiment_base.exists():
            return None

        subdirs = [d for d in experiment_base.iterdir() if d.is_dir()]
        if not subdirs:
            return None

        return max(subdirs, key=lambda x: x.stat().st_mtime)

    def resolve_output_path(
        self, filename: str, subdir: str = "analysis_results", create_dir: bool = True
    ) -> Path:
        """
        Resolve path for output files.

        Args:
            filename: Name of the output file
            subdir: Subdirectory within the output area
            create_dir: Whether to create the directory if it doesn't exist

        Returns:
            Full path for the output file
        """
        output_dir = self.get_path(subdir)
        if create_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / filename

    def list_files(
        self, path_name: str, pattern: str = "*", recursive: bool = False
    ) -> list[Path]:
        """
        list files in a standard path.

        Args:
            path_name: Name of the standard path
            pattern: Glob pattern for file matching
            recursive: Whether to search recursively

        Returns:
            list of matching file paths
        """
        base_path = self.get_path(path_name)

        if recursive:
            return list(base_path.rglob(pattern))
        else:
            return list(base_path.glob(pattern))

    def ensure_path_exists(self, path: str | Path) -> Path:
        """
        Ensure a path exists, creating it if necessary.

        Args:
            path: Path to ensure exists

        Returns:
            Path object (created if necessary)
        """
        path = Path(path)
        if self.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_relative_path(self, absolute_path: str | Path) -> Path:
        """
        Get relative path from the base directory.

        Args:
            absolute_path: Absolute path to convert

        Returns:
            Relative path from base directory
        """
        return Path(absolute_path).relative_to(self.base_dir)

    def validate_path_access(self, path: str | Path) -> bool:
        """
        Validate that a path is accessible.

        Args:
            path: Path to validate

        Returns:
            True if path is accessible, False otherwise
        """
        path = Path(path)
        return path.exists() and os.access(path, os.R_OK)

# Global path manager instance
_default_path_manager = None

def get_path_manager(
    base_dir: str | Path | None = None,
) -> AnalysisPathManager:
    """
    Get the global path manager instance.

    Args:
        base_dir: Base directory (only used for first initialization)

    Returns:
        Global path manager instance
    """
    global _default_path_manager

    if _default_path_manager is None:
        _default_path_manager = AnalysisPathManager(base_dir)

    return _default_path_manager

def resolve_project_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.

    Args:
        relative_path: Path relative to project root

    Returns:
        Absolute path
    """
    # Try to find project root by looking for common markers
    current = Path.cwd()

    # Look for project root markers
    root_markers = ["pyproject.toml", "setup.py", ".git", "ztb"]

    for marker in root_markers:
        candidate = current
        while candidate.parent != candidate:  # Not at filesystem root
            if (candidate / marker).exists():
                return candidate / relative_path
            candidate = candidate.parent

    # Fallback to current directory
    return Path(relative_path)
