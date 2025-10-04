#!/usr/bin/env python3
"""
parquet_io.py
Parquet I/O abstraction with configuration-driven compression, chunking, and column projection.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def load_config(config_path: Path = Path("configs/io.yaml")) -> Dict[str, Any]:
    """Load I/O configuration"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], yaml.safe_load(f))


def load_features_config(features_config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load features configuration for dependency analysis"""
    if features_config_path is None:
        features_config_path = Path("configs/features.yaml")
    if not features_config_path.exists():
        print(f"Features config not found: {features_config_path}")
        return {}

    with open(features_config_path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], yaml.safe_load(f))


def analyze_column_dependencies(
    features_config: Dict[str, Any], target_features: Optional[List[str]] = None
) -> Set[str]:
    """
    Analyze column dependencies from features configuration

    Args:
        features_config: Features configuration dict
        target_features: Specific features to analyze (if None, analyze all)

    Returns:
        Set of required column names
    """
    required_columns: Set[str] = set()

    # Base OHLCV columns that are always needed
    base_columns = {"open", "high", "low", "close", "volume", "timestamp"}
    required_columns.update(base_columns)

    # Analyze feature dependencies
    features = features_config.get("features", {})

    for category, feature_list in features.items():
        if not isinstance(feature_list, list):
            continue

        for feature_info in feature_list:
            if isinstance(feature_info, str):
                feature_name = feature_info
                feature_deps = []
            elif isinstance(feature_info, dict):
                feature_name = feature_info.get("name", "")
                feature_deps = feature_info.get("dependencies", [])
            else:
                continue

            # Skip if target_features is specified and this feature is not in it
            if target_features and feature_name not in target_features:
                continue

            # Add feature-specific dependencies
            required_columns.update(feature_deps)

            # Add common dependencies based on feature type
            if any(
                keyword in feature_name.lower()
                for keyword in ["ma", "sma", "ema", "moving"]
            ):
                required_columns.update(["close"])
            elif any(
                keyword in feature_name.lower()
                for keyword in ["rsi", "stoch", "williams"]
            ):
                required_columns.update(["close", "high", "low"])
            elif any(
                keyword in feature_name.lower() for keyword in ["volume", "vwap", "mfi"]
            ):
                required_columns.update(["close", "volume"])
            elif any(
                keyword in feature_name.lower()
                for keyword in ["bb", "bollinger", "atr"]
            ):
                required_columns.update(["close", "high", "low"])
            elif any(
                keyword in feature_name.lower() for keyword in ["ichimoku", "donchian"]
            ):
                required_columns.update(["open", "high", "low", "close"])
            elif any(keyword in feature_name.lower() for keyword in ["kalman"]):
                required_columns.update(["close"])
    # Convert to list and sort for consistency
    return required_columns


def smart_column_detection(
    parquet_path: Path, required_columns: Optional[Set[str]] = None
) -> List[str]:
    """
    Smart column detection based on parquet metadata and requirements

    Args:
        parquet_path: Path to parquet file
        required_columns: Set of required column names

    Returns:
        List of columns to read
    """
    try:
        # Read parquet metadata to get available columns
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = parquet_file.schema.names

        if required_columns:
            # Filter to only available required columns
            columns_to_read = [
                col for col in available_columns if col in required_columns
            ]
            missing_columns = required_columns - set(available_columns)

            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")

            return columns_to_read
        else:
            return cast(List[str], available_columns)

    except Exception as e:
        print(f"Error reading parquet metadata: {e}")
        return []


def write_parquet(
    df: pd.DataFrame, path: Path, config: Optional[Dict[str, Any]] = None
) -> None:
    """Write DataFrame to Parquet with configuration"""
    if config is None:
        config = load_config()

    parquet_cfg = config.get("parquet", {})
    compression = parquet_cfg.get("compression", "snappy")
    row_group_size = parquet_cfg.get("row_group_size", 50000)
    engine = parquet_cfg.get("engine", "pyarrow")

    # Convert to PyArrow Table for better control
    table = pa.Table.from_pandas(df)

    # Write with specified settings
    pq.write_table(
        table,
        path,
        compression=compression,
        row_group_size=row_group_size,
        use_dictionary=True,
        data_page_size=1024 * 1024,  # 1MB pages
    )


def read_parquet(
    path: Path,
    config: Optional[Dict[str, Any]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read Parquet to DataFrame with memory monitoring"""
    if config is None:
        config = load_config()

    limits = config.get("limits", {})
    peak_memory_mb = limits.get("peak_memory_mb", 1200)
    if peak_memory_mb is None:
        peak_memory_mb = 1200
    peak_memory_limit = int(peak_memory_mb) * 1024 * 1024  # MB to bytes

    # Monitor memory before read
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    # Determine columns to read
    parquet_cfg = config.get("parquet", {})
    use_columns = parquet_cfg.get("use_columns", [])
    if columns:
        use_columns = columns
    elif use_columns:
        pass  # use config
    else:
        use_columns = None  # read all

    # Read with column projection if specified
    df = pd.read_parquet(path, columns=use_columns, engine="pyarrow")

    # Check memory after read
    mem_after = process.memory_info().rss
    mem_increase = (mem_after - mem_before) / (1024 * 1024)  # MB

    if mem_after > peak_memory_limit:
        print(
            f"WARNING: Memory usage exceeded limit. Current: {mem_after / (1024 * 1024):.1f}MB, Limit: {peak_memory_limit / (1024 * 1024):.1f}MB"
        )

    return df


def read_parquet_with_features(
    path: Path,
    config: Optional[Dict[str, Any]] = None,
    target_features: Optional[List[str]] = None,
    features_config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Read Parquet with automatic column selection based on feature dependencies

    Args:
        path: Path to parquet file
        config: I/O configuration
        target_features: Specific features to optimize for (if None, analyze all)
        features_config_path: Path to features configuration

    Returns:
        DataFrame with only required columns
    """
    if config is None:
        config = load_config()

    # Load features configuration
    features_config = load_features_config(features_config_path)

    # Determine required columns based on target features
    if target_features:
        required_columns_set = set[str]()
        for feature in target_features:
            if feature in features_config:
                deps = features_config[feature].get("dependencies", [])
                required_columns_set.update(deps)
        required_columns = list(required_columns_set)
    else:
        required_columns = None

    # Smart column detection
    all_columns = smart_column_detection(path, None)
    columns_to_read = (
        [col for col in all_columns if col in required_columns]
        if required_columns
        else all_columns
    )

    if columns_to_read:
        original_column_count = len(all_columns)
        optimized_column_count = len(columns_to_read)
        reduction_pct = (
            (1 - optimized_column_count / original_column_count) * 100
            if original_column_count
            else 0
        )

        print(
            f"Column optimization: {original_column_count} → {optimized_column_count} ({reduction_pct:.1f}% reduction)"
        )
        print(f"Reading columns: {columns_to_read}")

    # Use optimized read_parquet with specific columns
    return read_parquet(path, config, columns_to_read)


def convert_to_parquet(
    input_path: Path, output_path: Path, compression: Optional[str] = None
) -> None:
    """Convert CSV/JSON to Parquet"""
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() == ".json":
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    config = load_config()
    if compression:
        config.setdefault("parquet", {})
        config["parquet"]["compression"] = compression

    write_parquet(df, output_path, config)
    print(f"Converted {input_path} to {output_path}")
    print(f"Converted {input_path} to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parquet I/O utilities")
    parser.add_argument("command", choices=["convert"], help="Command to run")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--compression", "-c", help="Override compression")

    args = parser.parse_args()

    if args.command == "convert":
        convert_to_parquet(Path(args.input), Path(args.output), args.compression)
