#!/usr/bin/env python3
"""
Common Data Loaders for Analysis Scripts

Provides standardized data loading interfaces for various analysis tasks.
Ensures type safety and consistent error handling across analysis components.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from ztb.io.data_loader import DataLoader
from ztb.io.json_io import read_json_object


class DataLoadError(Exception):
    """Exception raised when data loading fails."""

    pass


class BacktestDataLoader:
    """Standardized loader for backtest result data."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path("backtest_experiments")
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_latest_backtest_results(
        self, experiment_name: str, required_files: Optional[list[str]] = None
    ) -> dict[str, object]:
        """
        Load the latest backtest results from an experiment directory.

        Args:
            experiment_name: Name of the experiment directory
            required_files: List of required files to load

        Returns:
            Dictionary containing loaded data

        Raises:
            DataLoadError: If required data cannot be loaded
        """
        experiment_path = self.base_path / experiment_name

        if not experiment_path.exists():
            raise DataLoadError(f"Experiment directory not found: {experiment_path}")

        # Find latest results directory
        subdirs = [d for d in experiment_path.iterdir() if d.is_dir()]
        if not subdirs:
            raise DataLoadError(f"No result directories found in {experiment_path}")

        latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Loading results from: {latest_dir.name}")

        results: dict[str, object] = {"experiment_dir": str(latest_dir)}

        # Default required files
        if required_files is None:
            required_files = [
                "backtest_results.json",
                "portfolio_values.csv",
                "trades_history.csv",
            ]

        # Load each required file
        for filename in required_files:
            file_path = latest_dir / filename
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue

            try:
                if filename.endswith(".json"):
                    key = filename.replace(".json", "").replace("_", "")
                    results[key] = read_json_object(file_path)
                elif filename.endswith(".csv"):
                    key = filename.replace(".csv", "").replace("_", "")
                    results[key] = DataLoader.load_csv_strict(file_path)
                else:
                    self.logger.warning(f"Unsupported file type: {filename}")

            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")
                raise DataLoadError(f"Failed to load {filename}: {e}")

        return results

    def load_backtest_results_from_path(
        self, results_path: Union[str, Path]
    ) -> dict[str, object]:
        """
        Load backtest results from a specific path.

        Args:
            results_path: Path to the results file or directory

        Returns:
            Dictionary containing loaded data
        """
        results_path = Path(results_path)

        if results_path.is_file():
            if results_path.suffix == ".json":
                return read_json_object(results_path)
            else:
                raise DataLoadError(f"Unsupported file type: {results_path.suffix}")
        elif results_path.is_dir():
            return self.load_latest_backtest_results(results_path.name)
        else:
            raise DataLoadError(f"Path does not exist: {results_path}")


class TrainingDataLoader:
    """Loader for training result data."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path("results")
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_training_results(self, model_version: str) -> dict[str, object]:
        """
        Load training results for a specific model version.

        Args:
            model_version: Version identifier for the model

        Returns:
            Dictionary containing training results
        """
        results_file = self.base_path / f"training_report_{model_version}.json"

        if not results_file.exists():
            raise DataLoadError(f"Training results not found: {results_file}")

        try:
            return read_json_object(results_file)
        except Exception as e:
            raise DataLoadError(f"Failed to load training results: {e}")


class AnalysisDataLoader:
    """Loader for analysis-specific data files."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path("analysis")
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_analysis_results(self, analysis_name: str) -> dict[str, object]:
        """
        Load analysis results by name.

        Args:
            analysis_name: Name of the analysis

        Returns:
            Dictionary containing analysis results
        """
        results_file = self.base_path / f"{analysis_name}_results.json"

        if not results_file.exists():
            raise DataLoadError(f"Analysis results not found: {results_file}")

        try:
            return read_json_object(results_file)
        except Exception as e:
            raise DataLoadError(f"Failed to load analysis results: {e}")


# Convenience functions for backward compatibility
def load_backtest_results(results_path: Union[str, Path]) -> dict[str, object]:
    """Backward compatibility function for loading backtest results."""
    loader = BacktestDataLoader()
    return loader.load_backtest_results_from_path(results_path)


def load_training_results(model_version: str) -> dict[str, object]:
    """Backward compatibility function for loading training results."""
    loader = TrainingDataLoader()
    return loader.load_training_results(model_version)
