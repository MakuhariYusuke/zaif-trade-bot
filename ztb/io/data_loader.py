"""
Standardized data loading module for ZTB.

This module provides unified interfaces for loading data from various sources:
- Parquet files
- JSON files
- SQLite databases
- CSV files

All data loading should go through this module to ensure consistency.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from ztb.utils.errors import safe_operation


class DataLoader:
    """Unified data loader for ZTB."""

    @staticmethod
    def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from Parquet file."""
        return safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._load_parquet_impl(file_path),
            context="parquet_data_loading",
            default_result=pd.DataFrame(),  # Return empty DataFrame on failure
        )

    @staticmethod
    def _load_parquet_impl(file_path: Union[str, Path]) -> pd.DataFrame:
        """Implementation of Parquet data loading."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        return pd.read_parquet(file_path)

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file."""
        return safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._load_json_impl(file_path),
            context="json_data_loading",
            default_result={},  # Return empty dict on failure
        )

    @staticmethod
    def _load_json_impl(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Implementation of JSON data loading."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    @staticmethod
    def load_sqlite(db_path: Union[str, Path], query: str) -> pd.DataFrame:
        """Load data from SQLite database."""
        return safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._load_sqlite_impl(db_path, query),
            context="sqlite_data_loading",
            default_result=pd.DataFrame(),  # Return empty DataFrame on failure
        )

    @staticmethod
    def _load_sqlite_impl(db_path: Union[str, Path], query: str) -> pd.DataFrame:
        """Implementation of SQLite data loading."""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            return pd.read_sql_query(query, conn)

    @staticmethod
    def load_csv(file_path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        """Load data from CSV file."""
        return safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._load_csv_impl(file_path, **kwargs),
            context="csv_data_loading",
            default_result=pd.DataFrame(),  # Return empty DataFrame on failure
        )

    @staticmethod
    def _load_csv_impl(file_path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        """Implementation of CSV data loading."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        return pd.read_csv(file_path, **kwargs)  # type: ignore[no-any-return]

    @staticmethod
    def save_parquet(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """Save DataFrame to Parquet file."""
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._save_parquet_impl(df, file_path),
            context="parquet_data_saving",
            default_result=None,  # No meaningful default for save operations
        )

    @staticmethod
    def _save_parquet_impl(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """Implementation of Parquet data saving."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save data to JSON file."""
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._save_json_impl(data, file_path),
            context="json_data_saving",
            default_result=None,  # No meaningful default for save operations
        )

    @staticmethod
    def _save_json_impl(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Implementation of JSON data saving."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def save_sqlite(
        df: pd.DataFrame, db_path: Union[str, Path], table_name: str
    ) -> None:
        """Save DataFrame to SQLite database."""
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._save_sqlite_impl(df, db_path, table_name),
            context="sqlite_data_saving",
            default_result=None,  # No meaningful default for save operations
        )

    @staticmethod
    def _save_sqlite_impl(
        df: pd.DataFrame, db_path: Union[str, Path], table_name: str
    ) -> None:
        """Implementation of SQLite data saving."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)


# Convenience functions
def load_ohlc_data(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Load OHLC data from various sources.

    Args:
        source: File path or DataFrame

    Returns:
        OHLC DataFrame with standardized columns
    """
    if isinstance(source, pd.DataFrame):
        return source

    source_path = Path(source)
    if source_path.suffix == ".parquet":
        return DataLoader.load_parquet(source_path)
    elif source_path.suffix == ".csv":
        return DataLoader.load_csv(source_path)
    elif source_path.suffix == ".json":
        data = DataLoader.load_json(source_path)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {source_path.suffix}")


def save_evaluation_results(
    results: Dict[str, Any], output_path: Union[str, Path]
) -> None:
    """Save evaluation results to file."""
    output_path = Path(output_path)
    if output_path.suffix == ".json":
        DataLoader.save_json(results, output_path)
    else:
        # Save as pickle for complex objects
        import pickle

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
