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
from typing import Any, Iterator, cast

import pandas as pd

from ztb.utils.errors import safe_operation
from ztb.utils.file_utils import safe_json_load

class DataLoader:
    """Unified data loader for ZTB."""

    @staticmethod
    def _resolve_csv_engine(requested_engine: str | None) -> str | None:
        """Resolve the CSV engine, preferring pyarrow when available."""
        if requested_engine:
            return requested_engine
        try:
            import pyarrow  # noqa: F401
        except Exception:
            return None
        return "pyarrow"

    @staticmethod
    def load_parquet(file_path: str | Path) -> pd.DataFrame:
        """Load data from Parquet file."""
        return cast(
            pd.DataFrame,
            safe_operation(
                logger=None,  # Use default logger
                operation=lambda: DataLoader._load_parquet_impl(file_path),
                context="parquet_data_loading",
                default_result=pd.DataFrame(),  # Return empty DataFrame on failure
            ),
        )

    @staticmethod
    def _load_parquet_impl(file_path: str | Path) -> pd.DataFrame:
        """Implementation of Parquet data loading."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        return pd.read_parquet(file_path)

    @staticmethod
    def load_json(file_path: str | Path) -> dict[str, Any]:
        """Load data from JSON file."""
        return cast(
            dict[str, Any],
            safe_operation(
                logger=None,  # Use default logger
                operation=lambda: DataLoader._load_json_impl(file_path),
                context="json_data_loading",
                default_result={},  # Return empty dict on failure
            ),
        )

    @staticmethod
    def _load_json_impl(file_path: str | Path) -> dict[str, Any]:
        """Implementation of JSON data loading."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        return cast(dict[str, Any], safe_json_load(file_path))

    @staticmethod
    def load_sqlite(db_path: str | Path, query: str) -> pd.DataFrame:
        """Load data from SQLite database."""
        return cast(
            pd.DataFrame,
            safe_operation(
                logger=None,  # Use default logger
                operation=lambda: DataLoader._load_sqlite_impl(db_path, query),
                context="sqlite_data_loading",
                default_result=pd.DataFrame(),  # Return empty DataFrame on failure
            ),
        )

    @staticmethod
    def _load_sqlite_impl(db_path: str | Path, query: str) -> pd.DataFrame:
        """Implementation of SQLite data loading."""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            return pd.read_sql_query(query, conn)

    @staticmethod
    def load_csv(
        file_path: str | Path, *, strict: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        if strict:
            return DataLoader._load_csv_impl(file_path, **kwargs)
        return cast(
            pd.DataFrame,
            safe_operation(
                logger=None,  # Use default logger
                operation=lambda: DataLoader._load_csv_impl(file_path, **kwargs),
                context="csv_data_loading",
                default_result=pd.DataFrame(),  # Return empty DataFrame on failure
            ),
        )

    @staticmethod
    def load_csv_strict(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load data from CSV file and raise on failure."""
        return DataLoader._load_csv_impl(file_path, **kwargs)

    @staticmethod
    def load_csv_iter(
        file_path: str | Path, chunksize: int, **kwargs: Any
    ) -> Iterator[pd.DataFrame]:
        """Load CSV data in chunks."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            return cast(
                Iterator[pd.DataFrame],
                pd.read_csv(file_path, chunksize=chunksize, **kwargs),
            )
        except Exception as e:
            raise ValueError(f"Failed to load data from {file_path}: {e}") from e

    @staticmethod
    def load_csv_optimized(
        file_path: str | Path,
        usecols: list[str] | None = None,
        dtype: dict[str, Any] | None = None,
        parse_dates: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load CSV data with memory optimization."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if dtype is None:
            dtype = {
                "close": "float32",
                "high": "float32",
                "low": "float32",
                "open": "float32",
                "volume": "float32",
                "qty": "float32",
                "price": "float32",
                "rsi": "float32",
                "sma_short": "float32",
                "sma_long": "float32",
                "ADX": "float32",
                "ATR": "float32",
                "ATR_simplified": "float32",
                "BB_Lower": "float32",
                "BB_Middle": "float32",
                "BB_Position": "float32",
                "BB_Upper": "float32",
                "BB_Width": "float32",
                "CCI": "float32",
                "DOW": "float32",
                "Donchian_Pos_2": "float32",
                "Donchian_Slope_20": "float32",
                "Donchian_Width_Rel_20": "float32",
                "EMACross_Diff": "float32",
                "EMACross_Signal": "float32",
                "HV": "float32",
                "HeikinAshi_Close": "float32",
                "HeikinAshi_High": "float32",
                "HeikinAshi_Low": "float32",
                "HeikinAshi_Open": "float32",
                "HourOfDay": "int32",
                "Ichimoku_Chikou": "float32",
                "Ichimoku_Cloud_Thickness": "float32",
                "Ichimoku_Composite_Signal": "float32",
                "Ichimoku_Cross": "float32",
                "Ichimoku_Diff_Norm": "float32",
                "Ichimoku_Kijun": "float32",
                "Ichimoku_Price_Cloud_Distance": "float32",
                "Ichimoku_Senkou_A": "float32",
                "Ichimoku_Senkou_B": "float32",
                "Ichimoku_Tenkan": "float32",
                "Ichimoku_Trend": "float32",
                "KAMA": "float32",
                "Kalman_Estimate": "float32",
                "Kalman_Residual": "float32",
                "Kalman_Residual_Norm": "float32",
                "MACD": "float32",
                "MFI": "float32",
                "MinusDI": "float32",
                "OBV": "float32",
                "PlusDI": "float32",
                "PriceVolumeCorr": "float32",
                "ROC": "float32",
                "RSI": "float32",
                "ReturnMA_Medium": "float32",
                "ReturnMA_Short": "float32",
                "ReturnStdDev": "float32",
                "Stochastic": "float32",
                "Supertrend": "float32",
                "Supertrend_Direction": "float32",
                "TEMA": "float32",
                "VWAP": "float32",
                "ZScore": "float32",
                "atr_10": "float32",
                "ema_5": "float32",
                "rolling_mean_20": "float32",
                "win": "int32",
            }

        header_df = DataLoader.load_csv_strict(file_path, nrows=0)
        available_columns = list(header_df.columns)

        if dtype:
            dtype = {
                key: value
                for key, value in dtype.items()
                if key in available_columns
                and (usecols is None or key in usecols)
            }
        else:
            dtype = None

        if parse_dates is None:
            parse_dates = [
                col for col in ["timestamp", "ts"] if col in available_columns
            ]

        if "memory_map" not in kwargs:
            kwargs["memory_map"] = True
        engine = DataLoader._resolve_csv_engine(kwargs.pop("engine", None))
        read_kwargs = {
            "usecols": usecols,
            "dtype": cast(Any, dtype) if dtype else None,
            "parse_dates": parse_dates,
            **kwargs,
        }
        if engine:
            try:
                df = cast(
                    pd.DataFrame,
                    pd.read_csv(file_path, engine=engine, **read_kwargs),
                )
            except ValueError as exc:
                message = str(exc).lower()
                if "engine" not in message and "pyarrow" not in message:
                    raise
                df = cast(pd.DataFrame, pd.read_csv(file_path, **read_kwargs))
        else:
            df = cast(pd.DataFrame, pd.read_csv(file_path, **read_kwargs))

        if df.empty and kwargs.get("nrows") != 0:
            raise ValueError(f"Loaded data is empty: {file_path}")

        return df

    @staticmethod
    def _load_csv_impl(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Implementation of CSV data loading."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if "memory_map" not in kwargs:
            kwargs["memory_map"] = True
        engine = DataLoader._resolve_csv_engine(kwargs.pop("engine", None))
        if engine:
            try:
                return cast(
                    pd.DataFrame, pd.read_csv(file_path, engine=engine, **kwargs)
                )
            except ValueError as exc:
                message = str(exc).lower()
                if "engine" not in message and "pyarrow" not in message:
                    raise
        return cast(pd.DataFrame, pd.read_csv(file_path, **kwargs))

    @staticmethod
    def save_parquet(df: pd.DataFrame, file_path: str | Path) -> None:
        """Save DataFrame to Parquet file."""
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._save_parquet_impl(df, file_path),
            context="parquet_data_saving",
            default_result=None,  # No meaningful default for save operations
        )

    @staticmethod
    def _save_parquet_impl(df: pd.DataFrame, file_path: str | Path) -> None:
        """Implementation of Parquet data saving."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path)

    @staticmethod
    def save_json(data: dict[str, Any], file_path: str | Path) -> None:
        """Save data to JSON file."""
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: DataLoader._save_json_impl(data, file_path),
            context="json_data_saving",
            default_result=None,  # No meaningful default for save operations
        )

    @staticmethod
    def _save_json_impl(data: dict[str, Any], file_path: str | Path) -> None:
        """Implementation of JSON data saving."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def save_sqlite(
        df: pd.DataFrame, db_path: str | Path, table_name: str
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
        df: pd.DataFrame, db_path: str | Path, table_name: str
    ) -> None:
        """Implementation of SQLite data saving."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

# Convenience functions
def load_ohlc_data(source: str | Path | pd.DataFrame) -> pd.DataFrame:
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
    results: dict[str, Any], output_path: str | Path
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
