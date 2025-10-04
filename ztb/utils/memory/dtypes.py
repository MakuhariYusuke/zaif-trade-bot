"""Helpers for optimizing pandas DataFrame dtypes to reduce memory footprint."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


@dataclass
class OptimizationReport:
    """Summary of a dtype optimization pass."""

    columns_optimized: List[str]
    memory_before_bytes: int
    memory_after_bytes: int

    @property
    def memory_saved_bytes(self) -> int:
        return max(0, self.memory_before_bytes - self.memory_after_bytes)

    @property
    def memory_before_mb(self) -> float:
        return self.memory_before_bytes / (1024 * 1024)

    @property
    def memory_after_mb(self) -> float:
        return self.memory_after_bytes / (1024 * 1024)

    @property
    def memory_saved_mb(self) -> float:
        return self.memory_saved_bytes / (1024 * 1024)

    @property
    def percent_reduction(self) -> float:
        if self.memory_before_bytes == 0:
            return 0.0
        return (self.memory_saved_bytes / self.memory_before_bytes) * 100


def _iter_chunks(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    if chunk_size <= 0:
        yield items
        return
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _downcast_float_column(
    series: pd.Series,
    target_dtype: str,
    sample_size: int,
    atol: float,
    rtol: float,
) -> Tuple[pd.Series, bool]:
    if series.empty:
        return series.astype(target_dtype), True  # type: ignore

    try:
        converted = series.astype(target_dtype)  # type: ignore
    except (TypeError, ValueError, OverflowError):
        return series, False

    sample = series.dropna()
    if sample.empty:
        return converted, True

    if 0 < sample_size < len(sample):
        sample = sample.sample(sample_size, random_state=0)
    converted_sample = converted.loc[sample.index]

    # Compare in float64 space to evaluate precision loss.
    original_values = sample.astype("float64").to_numpy(copy=False)
    converted_values = converted_sample.astype("float64").to_numpy(copy=False)

    if not np.allclose(
        original_values, converted_values, rtol=rtol, atol=atol, equal_nan=True
    ):
        return series, False

    return converted, True


def _optimize_integer_column(series: pd.Series) -> Tuple[pd.Series, bool]:
    try:
        downcasted = pd.to_numeric(series, downcast="integer")
        return downcasted, downcasted.dtype != series.dtype
    except (TypeError, ValueError):
        return series, False


def _optimize_boolean_column(series: pd.Series) -> Tuple[pd.Series, bool]:
    if ptypes.is_bool_dtype(series):
        return series, False

    unique_values = series.dropna().unique()
    if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False}):
        try:
            return series.astype("bool"), True
        except (TypeError, ValueError):
            return series, False
    return series, False


def _optimize_object_column(
    series: pd.Series,
    max_categories: int,
    max_ratio: float,
    treat_small_as_category: bool,
) -> Tuple[pd.Series, bool]:
    if not ptypes.is_object_dtype(series) and not ptypes.is_string_dtype(series):
        return series, False

    if not treat_small_as_category or series.empty:
        return series, False

    nunique = series.nunique(dropna=True)
    if nunique == 0:
        return series, False

    ratio = nunique / len(series)
    if nunique <= max_categories or ratio <= max_ratio:
        try:
            return series.astype("category"), True
        except (TypeError, ValueError):
            return series, False

    return series, False


def optimize_dtypes(
    df: pd.DataFrame,
    *,
    target_float_dtype: str = "float32",
    float_atol: float = 1e-6,
    float_rtol: float = 1e-4,
    float_sample_size: int = 5000,
    target_int_dtype: str = "int32",
    convert_objects_to_category: bool = True,
    max_category_cardinality: int = 512,
    max_category_ratio: float = 0.5,
    chunk_size: int = 25,
    enable_gc: bool = True,
    memory_report: bool = False,
) -> Tuple[pd.DataFrame, OptimizationReport]:
    """Downcast numeric columns and convert low-cardinality objects to categories.

    Args:
        df: Input dataframe.
        target_float_dtype: Preferred dtype for float columns (requires precision check).
        float_atol: Absolute tolerance for float comparison when checking precision loss.
        float_rtol: Relative tolerance for float comparison when checking precision loss.
        float_sample_size: Number of non-null samples used to validate float downcasts.
        target_int_dtype: Preferred dtype for integer columns when range allows.
        convert_objects_to_category: Enable object->category conversion for low-cardinality columns.
        max_category_cardinality: Upper bound for unique values when converting to category.
        max_category_ratio: Maximum (nunique / len(column)) ratio to allow category conversion.
        chunk_size: Number of columns processed per optimization chunk; reduces peak memory use.
        enable_gc: Run garbage collection after each chunk to release temporary objects.
        memory_report: Log memory savings summary to stdout when True.

    Returns:
        Tuple of optimized dataframe and an OptimizationReport describing the changes.
    """

    optimized = df.copy()
    before_bytes = int(optimized.memory_usage(deep=True).sum())
    optimized_columns: List[str] = []

    columns = list(optimized.columns)
    for chunk in _iter_chunks(columns, chunk_size):
        for col in chunk:
            series = optimized[col]

            if ptypes.is_float_dtype(series):
                if str(series.dtype) != target_float_dtype:
                    converted, changed = _downcast_float_column(
                        series,
                        target_float_dtype,
                        float_sample_size,
                        float_atol,
                        float_rtol,
                    )
                    if changed:
                        optimized[col] = converted
                        optimized_columns.append(col)
                continue

            if ptypes.is_integer_dtype(series):
                base_series = (
                    series.astype("Int64")
                    if ptypes.is_extension_array_dtype(series.dtype)
                    else series
                )
                converted, changed = _optimize_integer_column(base_series)
                if changed:
                    optimized[col] = converted
                    optimized_columns.append(col)
                series = optimized[col]

                series_dtype_str = str(series.dtype)
                if (
                    series_dtype_str in {"int64", "Int64"}
                    and series_dtype_str != target_int_dtype
                ):
                    try:
                        casted = series.astype(target_int_dtype)  # type: ignore
                    except (TypeError, ValueError, OverflowError):
                        pass
                    else:
                        optimized[col] = casted
                        if col not in optimized_columns:
                            optimized_columns.append(col)
                continue

            if ptypes.is_bool_dtype(series):
                converted, changed = _optimize_boolean_column(series)
                if changed:
                    optimized[col] = converted
                    optimized_columns.append(col)
                continue

            converted, changed = _optimize_object_column(
                series,
                max_categories=max_category_cardinality,
                max_ratio=max_category_ratio,
                treat_small_as_category=convert_objects_to_category,
            )
            if changed:
                optimized[col] = converted
                optimized_columns.append(col)

        if enable_gc:
            gc.collect()

    after_bytes = int(optimized.memory_usage(deep=True).sum())
    report = OptimizationReport(
        columns_optimized=sorted(set(optimized_columns)),
        memory_before_bytes=before_bytes,
        memory_after_bytes=after_bytes,
    )

    optimized.attrs.setdefault("memory_optimization", {})
    optimized.attrs["memory_optimization"].update(
        {
            "before_bytes": before_bytes,
            "after_bytes": after_bytes,
            "saved_bytes": report.memory_saved_bytes,
            "columns": report.columns_optimized,
        }
    )

    if memory_report:
        print(
            "Memory optimized: {:.2f} MB -> {:.2f} MB ({:.2f}% reduction)".format(
                report.memory_before_mb,
                report.memory_after_mb,
                report.percent_reduction,
            )
        )

    return optimized, report


def downcast_df(
    df: pd.DataFrame,
    float_dtype: str = "float32",
    int_dtype: str = "int32",
    *,
    memory_report: bool = False,
) -> pd.DataFrame:
    """Backward compatible wrapper for the previous downcast_df helper."""

    optimized, _ = optimize_dtypes(
        df,
        target_float_dtype=float_dtype,
        target_int_dtype=int_dtype,
        memory_report=memory_report,
    )
    return optimized
