#!/usr/bin/env python3
"""
Results Utilities

Unified utilities for saving and loading training/backtest results.
Provides consistent result storage format across the project.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ztb.utils.data_utils import load_csv_data
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_training_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "training_results.json",
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> str:
    """
    Save training results to JSON file with consistent format.

    Args:
        results: Training results dictionary
        output_dir: Directory to save results
        filename: Results filename (default: training_results.json)
        metadata: Additional metadata to include
        overwrite: Whether to overwrite existing file

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename

    if filepath.exists() and not overwrite:
        logger.warning(f"Results file already exists: {filepath}")
        return str(filepath)

    # Prepare results data
    results_data = {
        "results": results,
        "timestamp": pd.Timestamp.now().isoformat(),
        "metadata": metadata or {},
    }

    # Save to JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"Training results saved to {filepath}")
    return str(filepath)


def load_training_results(
    filepath: Union[str, Path], validate_keys: Optional[list] = None
) -> Dict[str, Any]:
    """
    Load training results from JSON file.

    Args:
        filepath: Path to results file
        validate_keys: Keys that must be present in results

    Returns:
        Loaded results dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if "results" not in data:
        raise ValueError(
            f"Invalid results file format: missing 'results' key in {filepath}"
        )

    results = data["results"]

    # Validate required keys
    if validate_keys:
        missing_keys = [key for key in validate_keys if key not in results]
        if missing_keys:
            raise ValueError(f"Missing required keys in results: {missing_keys}")

    logger.info(f"Training results loaded from {filepath}")
    return results


def save_backtest_results(
    portfolio_values: list,
    trade_history: list,
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    filename_prefix: str = "backtest",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save backtest results with consistent format.

    Args:
        portfolio_values: List of portfolio values over time
        trade_history: List of trade records
        metrics: Performance metrics
        output_dir: Directory to save results
        filename_prefix: Prefix for result files
        metadata: Additional metadata

    Returns:
        Dictionary mapping file types to saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save portfolio values
    portfolio_df = pd.DataFrame({"portfolio_value": portfolio_values})
    portfolio_path = output_dir / f"{filename_prefix}_portfolio.csv"
    portfolio_df.to_csv(portfolio_path, index=False)
    saved_files["portfolio"] = str(portfolio_path)

    # Save trade history if available
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        trade_path = output_dir / f"{filename_prefix}_trades.csv"
        trade_df.to_csv(trade_path, index=False)
        saved_files["trades"] = str(trade_path)

    # Save metrics
    metrics_data = {
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat(),
        "metadata": metadata or {},
        "portfolio_points": len(portfolio_values),
        "total_trades": len(trade_history) if trade_history else 0,
    }

    metrics_path = output_dir / f"{filename_prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=str, ensure_ascii=False)
    saved_files["metrics"] = str(metrics_path)

    logger.info(f"Backtest results saved to {output_dir}")
    return saved_files


def load_backtest_results(
    results_dir: Union[str, Path], filename_prefix: str = "backtest"
) -> Dict[str, Any]:
    """
    Load backtest results from directory.

    Args:
        results_dir: Directory containing results
        filename_prefix: Prefix used when saving

    Returns:
        Dictionary containing loaded results
    """
    results_dir = Path(results_dir)

    results = {}

    # Load portfolio values
    portfolio_path = results_dir / f"{filename_prefix}_portfolio.csv"
    if portfolio_path.exists():
        portfolio_df = load_csv_data(portfolio_path)
        results["portfolio_values"] = portfolio_df["portfolio_value"].tolist()

    # Load trade history
    trade_path = results_dir / f"{filename_prefix}_trades.csv"
    if trade_path.exists():
        trade_df = pd.read_csv(trade_path)
        results["trade_history"] = trade_df.to_dict("records")

    # Load metrics
    metrics_path = results_dir / f"{filename_prefix}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        results["metrics"] = metrics_data["metrics"]
        results["metadata"] = metrics_data.get("metadata", {})

    logger.info(f"Backtest results loaded from {results_dir}")
    return results
