"""
Feature registry for trading features.
特徴量レジストリ
"""

import gc
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import psutil
from pandas.api import types as ptypes

from ztb.features.utils.rolling import generate_intermediate_report
from ztb.utils.memory.dtypes import optimize_dtypes

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None  # type: ignore


class FeatureRegistry:
    """全特徴量関数を一元管理するレジストリ"""

    _registry: Dict[str, Callable[..., pd.Series]] = {}
    _cache_enabled: bool = True
    _parallel_enabled: bool = True
    _initialized: bool = False
    _config: Dict[str, Any] = {}
    _base_seed: int = 42

    @classmethod
    def initialize(
        cls,
        seed: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        parallel_enabled: Optional[bool] = None,
    ) -> None:
        """Initialize the registry with seed, cache and parallel settings"""
        if cls._initialized:
            return

        # Set seed from parameter or default
        final_seed = seed if seed is not None else 42

        # Fix seeds for reproducibility
        np.random.seed(final_seed)
        random.seed(final_seed)
        os.environ["PYTHONHASHSEED"] = str(final_seed)

        # Set PyTorch seed if available
        try:
            import torch

            torch.manual_seed(final_seed)
            torch.cuda.manual_seed_all(final_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass  # PyTorch not available

        # Set BLAS thread limits for optimal parallel performance
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        # Store base seed for parallel processing
        cls._base_seed = final_seed

        # Set cache enabled from parameter or default
        if cache_enabled is not None:
            cls._cache_enabled = cache_enabled
        else:
            cls._cache_enabled = True

        # Set parallel enabled from parameter or default
        if parallel_enabled is not None:
            cls._parallel_enabled = parallel_enabled
        else:
            cls._parallel_enabled = True

        cls._config.update(
            {
                "memory_monitor_enabled": False,
                "memory_log_interval": 10,
                "memory_threshold_pct": 80.0,
                "gc_collect_interval": 10,
                "max_parallel_workers": None,
                "parallel_batch_size": 20,
                "feature_chunk_size": 20,
                "optimize_dtypes": True,
                "dtype_chunk_size": 100,
                "dtype_memory_report": False,
                "dtype_convert_objects": True,
                "feature_float_dtype": "float32",
                "coerce_feature_floats": True,
                "enable_chunking": True,
                "chunk_size": 1000,
                "gc_after_dtype_optimization": True,
            }
        )

        cls._initialized = True

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset registry state for testing purposes"""
        cls._initialized = False
        cls._registry.clear()
        cls._config.clear()

    @classmethod
    def set_worker_seed(cls, worker_id: int) -> None:
        """Set deterministic seed for parallel worker processes"""
        if not cls._initialized:
            raise RuntimeError(
                "FeatureRegistry must be initialized before setting worker seed"
            )

        worker_seed = cls._base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Set PyTorch seed if available
        try:
            import torch

            torch.manual_seed(worker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(worker_seed)
        except ImportError:
            pass  # PyTorch not available

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get current configuration"""
        from copy import deepcopy

        return deepcopy(cls._config)

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[..., pd.Series]], Callable[..., pd.Series]]:
        def decorator(
            func: Callable[..., pd.Series],
        ) -> Callable[..., pd.Series]:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[..., pd.Series]:
        if name not in cls._registry:
            raise KeyError(
                f"Feature '{name}' is not registered in the FeatureRegistry."
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def is_cache_enabled(cls) -> bool:
        """Check if caching is enabled"""
        return cls._cache_enabled

    @classmethod
    def is_parallel_enabled(cls) -> bool:
        """Check if parallel processing is enabled"""
        return cls._parallel_enabled

    def get_enabled_features(self, wave: Optional[int] = None) -> List[str]:
        """Get enabled features for the given wave"""
        return type(self).list()

    @classmethod
    def compute_features_batch(
        cls,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        report_interval: int = 10000,
        verbose: bool = True,
        enable_chunking: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        return_timing: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute features in batch with stability and efficiency improvements

        Args:
            df: Input DataFrame with OHLCV data
            feature_names: List of feature names to compute (None for all)
            report_interval: Steps between intermediate reports
            verbose: Whether to print progress information
            enable_chunking: Whether to process data in chunks (overrides config)
            chunk_size: Size of chunks for processing (overrides config)

        Returns:
            DataFrame with computed features
        """
        cls.initialize()

        if feature_names is None:
            feature_names = cls.list()

        config = cls.get_config()

        # Override chunking settings if provided
        if enable_chunking is not None:
            config = config.copy()
            config["enable_chunking"] = enable_chunking
        if chunk_size is not None:
            config = config.copy()
            config["chunk_size"] = chunk_size

        # Check if chunking is enabled and beneficial
        enable_chunking = config.get("enable_chunking", True)
        chunk_size = config.get("chunk_size", 5000)
        if chunk_size is None:
            chunk_size = 5000
        chunk_size = max(1000, chunk_size)  # Minimum chunk size

        if enable_chunking and len(df) > chunk_size:
            features_df = cls._compute_features_chunked(
                df, feature_names, chunk_size, report_interval, verbose
            )
            if return_timing:
                # For chunked processing, we don't have detailed timing info
                return features_df, {}
        else:
            features_df, feature_times = cls._compute_features_single(
                df, feature_names, report_interval, verbose
            )

        if return_timing:
            return features_df, feature_times
        else:
            return features_df

    @classmethod
    def _precompute_common_calculations(
        cls, df: pd.DataFrame, feature_names: List[str]
    ) -> pd.DataFrame:
        """Pre-compute common calculations shared across multiple features"""
        df = df.copy()

        # Common EMA periods used by multiple features
        common_ema_periods = {12, 26, 9, 14, 21, 50, 200}

        # Common SMA periods
        common_sma_periods = {10, 20, 50, 200}

        # Common ATR periods used by Supertrend and other features
        common_atr_periods = {10, 14, 20}

        # Check which calculations are actually needed
        needs_ema = any(
            name in feature_names
            and any(
                keyword in name.upper()
                for keyword in ["MACD", "ADX", "TREND", "MA", "SMA", "EMA"]
            )
            for name in feature_names
        )

        needs_sma = any(
            name in feature_names
            and any(keyword in name.upper() for keyword in ["SMA", "MA", "MOVING"])
            for name in feature_names
        )

        needs_atr = any(
            name in feature_names
            and any(
                keyword in name.upper() for keyword in ["SUPER", "ATR", "VOLATILITY"]
            )
            for name in feature_names
        )

        # Pre-compute EMAs
        if needs_ema:
            for period in common_ema_periods:
                col_name = f"ema_{period}"
                if col_name not in df.columns:
                    df[col_name] = df["close"].ewm(span=period, adjust=False).mean()

        # Pre-compute SMAs
        if needs_sma:
            for period in common_sma_periods:
                col_name = f"sma_{period}"
                if col_name not in df.columns:
                    df[col_name] = df["close"].rolling(window=period).mean()

        # Pre-compute ATRs
        if needs_atr:
            for period in common_atr_periods:
                col_name = f"atr_{period}"
                if col_name not in df.columns:
                    # Simplified ATR calculation for pre-computation
                    high = df["high"]
                    low = df["low"]
                    close = df["close"]
                    tr = pd.concat(
                        [
                            high - low,
                            (high - close.shift(1)).abs(),
                            (low - close.shift(1)).abs(),
                        ],
                        axis=1,
                    ).max(axis=1)
                    df[col_name] = tr.rolling(window=period).mean()

        return df

    @classmethod
    def _compute_features_single(
        cls,
        df: pd.DataFrame,
        feature_names: List[str],
        report_interval: int,
        verbose: bool,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Compute features on a single chunk"""
        if verbose:
            print(f"Computing {len(feature_names)} features on single chunk...")

        if not feature_names:
            return pd.DataFrame(index=df.index.copy()), {}

        # Pre-compute common calculations to share across features
        df = cls._precompute_common_calculations(df, feature_names)

        def _as_bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return bool(value)
            value_str = str(value).strip().lower()
            if value_str in {"true", "1", "yes", "y", "on"}:
                return True
            if value_str in {"false", "0", "no", "n", "off"}:
                return False
            return default

        config = cls.get_config()

        memory_monitor_enabled = _as_bool(config.get("memory_monitor_enabled", True))

        target_float_dtype = str(
            config.get("feature_float_dtype", "float16")
        )  # More aggressive: float16 instead of float32
        coerce_feature_floats = _as_bool(config.get("coerce_feature_floats", True))

        memory_log_interval = config.get("memory_log_interval")
        try:
            memory_log_interval = (
                int(memory_log_interval)
                if memory_log_interval is not None
                else max(5, min(50, len(feature_names) // 8 or 1))
            )  # More frequent monitoring
        except (TypeError, ValueError):
            memory_log_interval = max(5, min(50, len(feature_names) // 8 or 1))
        if memory_log_interval <= 0:
            memory_monitor_enabled = False

        gc_collect_interval = config.get("gc_collect_interval")
        try:
            gc_collect_interval = (
                int(gc_collect_interval) if gc_collect_interval is not None else 5
            )  # More frequent GC
        except (TypeError, ValueError):
            gc_collect_interval = 5
        gc_collect_interval = max(0, gc_collect_interval)

        cpu_count = os.cpu_count() or 1
        configured_workers = config.get("max_parallel_workers")
        if configured_workers is not None:
            try:
                configured_workers = int(configured_workers)
            except (TypeError, ValueError):
                configured_workers = None

        if configured_workers is None or configured_workers <= 0:
            # More aggressive parallelization for I/O bound tasks
            # Use more workers than CPU cores since feature computation is often I/O bound
            max_workers = min(len(feature_names), max(1, cpu_count * 2))
        else:
            max_workers = max(1, min(configured_workers, len(feature_names)))

        if not cls.is_parallel_enabled():
            max_workers = 1

        # 動的ワーカー調整: メモリ使用率に基づく + フィーチャー数に基づく最適化
        initial_memory_pct = psutil.virtual_memory().percent
        if initial_memory_pct > 80.0:
            max_workers = max(
                1, max_workers // 4
            )  # More aggressive reduction for high memory
        elif initial_memory_pct > 70.0:
            max_workers = max(1, max_workers // 2)
        elif initial_memory_pct > 50.0:
            max_workers = max(1, max_workers * 3 // 4)  # Moderate reduction

        # Adjust based on feature count - fewer features don't benefit from many workers
        if len(feature_names) < 10:
            max_workers = min(max_workers, 4)
        elif len(feature_names) < 50:
            max_workers = min(max_workers, cpu_count)

        if verbose and max_workers != min(len(feature_names), max(1, cpu_count * 2)):
            print(
                f"Adjusted workers from {min(len(feature_names), max(1, cpu_count * 2))} to {max_workers} (memory: {initial_memory_pct:.1f}%)"
            )

        parallel_batch_size = config.get("parallel_batch_size")
        try:
            parallel_batch_size = (
                int(parallel_batch_size) if parallel_batch_size is not None else 0
            )
        except (TypeError, ValueError):
            parallel_batch_size = 0

        if parallel_batch_size <= 0:
            # Dynamic batch sizing based on feature count and memory
            if len(feature_names) <= 20:
                parallel_batch_size = len(feature_names)  # No batching for small sets
            elif initial_memory_pct > 60.0:
                parallel_batch_size = min(
                    10, len(feature_names)
                )  # Smaller batches for high memory
            else:
                parallel_batch_size = min(
                    20, len(feature_names)
                )  # Larger batches for normal memory
        else:
            parallel_batch_size = max(1, parallel_batch_size)

        feature_chunk_size = config.get("feature_chunk_size")
        try:
            feature_chunk_size = (
                int(feature_chunk_size)
                if feature_chunk_size is not None
                else parallel_batch_size
            )
        except (TypeError, ValueError):
            feature_chunk_size = parallel_batch_size

        feature_chunk_size = max(1, min(feature_chunk_size, len(feature_names)))

        # Optimize chunk size based on memory and feature count
        if len(feature_names) > 100:
            feature_chunk_size = max(
                feature_chunk_size, 10
            )  # Ensure minimum chunk size for large feature sets
        elif len(feature_names) < 20:
            feature_chunk_size = min(
                feature_chunk_size, 5
            )  # Smaller chunks for quick features

        memory_threshold_pct = float(config.get("memory_threshold_pct", 85.0))

        # パフォーマンスメトリクス収集
        perf_metrics: Dict[str, Any] = {
            "start_time": time.time(),
            "cpu_count": cpu_count,
            "max_workers": max_workers,
            "initial_memory_pct": initial_memory_pct,
            "features_count": len(feature_names),
            "sequential_fallbacks": 0,
            "memory_pressure_events": 0,
            "failed_features": [],
        }

        process = psutil.Process()
        if memory_monitor_enabled and verbose:
            input_mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            rss_mb = process.memory_info().rss / 1024 / 1024
            print(f"[Memory] input frame={input_mem_mb:.2f} MB RSS={rss_mb:.2f} MB")
        features_df = pd.DataFrame(index=df.index.copy())
        feature_times = {}
        nan_rates = {}
        step_count = 0

        pending_features = deque(feature_names)
        sequential_mode = False

        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                total=len(feature_names),
                desc="Computing features",
                unit="feature",
                disable=not verbose,
            )

        def _compute_feature(
            feature_name: str,
        ) -> Tuple[str, Optional[pd.Series], float, float, Optional[Exception]]:
            start_time = time.time()
            try:
                feature_func = cls.get(feature_name)
                feature_series = feature_func(df)
                if coerce_feature_floats and ptypes.is_float_dtype(feature_series):
                    feature_series = feature_series.astype(target_float_dtype, copy=False)  # type: ignore
                nan_rate = feature_series.isna().mean()
                return (
                    feature_name,
                    feature_series,
                    time.time() - start_time,
                    nan_rate,
                    None,
                )
            except Exception as exc:
                return feature_name, None, 0.0, 1.0, exc

        def _process_result(
            result: Tuple[str, Optional[pd.Series], float, float, Optional[Exception]],
        ) -> None:
            nonlocal step_count
            feature_name, feature_series, computation_time, nan_rate, error = result

            if error is not None:
                print(f"Error computing {feature_name}: {error}")
                perf_metrics["failed_features"].append(feature_name)
            else:
                features_df[feature_name] = feature_series
                # Immediately delete the series to free memory
                del feature_series

            feature_times[feature_name] = computation_time
            nan_rates[feature_name] = nan_rate

            step_count += 1

            if progress_bar is not None:
                progress_bar.update(1)

            # More aggressive memory management
            if gc_collect_interval and step_count % gc_collect_interval == 0:
                gc.collect()

            if (
                memory_monitor_enabled
                and memory_log_interval
                and step_count % memory_log_interval == 0
            ):
                frame_memory_mb = (
                    features_df.memory_usage(deep=True).sum() / 1024 / 1024
                )
                rss_mb = process.memory_info().rss / 1024 / 1024
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {"rss_mb": f"{rss_mb:.1f}", "df_mb": f"{frame_memory_mb:.1f}"},
                        refresh=False,
                    )
                if verbose:
                    print(
                        f"[Memory] step {step_count}: features_df={frame_memory_mb:.2f} MB RSS={rss_mb:.2f} MB"
                    )

            if gc_collect_interval and step_count % gc_collect_interval == 0:
                gc.collect()

            if step_count % report_interval == 0:
                memory_usage_mb = process.memory_info().rss / 1024 / 1024
                generate_intermediate_report(
                    step_count, feature_times, memory_usage_mb, nan_rates
                )
                print(f"Generated intermediate report at step {step_count}")

            # Clean up local variables
            del feature_name, computation_time, nan_rate, error

        try:
            while pending_features:
                current_memory_pct = psutil.virtual_memory().percent
                if (
                    not sequential_mode
                    and current_memory_pct >= memory_threshold_pct
                    and max_workers > 1
                ):
                    sequential_mode = True
                    perf_metrics["sequential_fallbacks"] += 1
                    perf_metrics["memory_pressure_events"] += 1
                    if verbose:
                        print(
                            f"High memory usage detected ({current_memory_pct:.1f}%). "
                            "Switching to sequential batch mode."
                        )
                    gc.collect()

                batch_size = (
                    1
                    if sequential_mode
                    else min(feature_chunk_size, len(pending_features))
                )
                batch = [pending_features.popleft() for _ in range(batch_size)]
                worker_count = 1 if sequential_mode else min(max_workers, len(batch))

                if worker_count == 1 and len(batch) == 1:
                    _process_result(_compute_feature(batch[0]))
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = [
                            executor.submit(_compute_feature, name) for name in batch
                        ]
                        for future in as_completed(futures):
                            _process_result(future.result())
                    del futures
                del batch
        finally:
            if progress_bar is not None:
                progress_bar.close()
            pending_features.clear()

        computed_features = [
            name for name in feature_names if name in features_df.columns
        ]
        features_df = features_df.loc[:, computed_features]

        if features_df.empty:
            return features_df, {}

        if config.get("optimize_dtypes", True):
            dtype_chunk_size = config.get("dtype_chunk_size")
            if dtype_chunk_size is not None:
                try:
                    dtype_chunk_size = int(dtype_chunk_size)
                except (TypeError, ValueError):
                    dtype_chunk_size = 50
            else:
                dtype_chunk_size = 50

            dtype_memory_report = _as_bool(config.get("dtype_memory_report", False))
            dtype_convert_objects = _as_bool(
                config.get("dtype_convert_objects", True), default=True
            )

            features_df = optimize_dtypes(
                features_df,
                target_float_dtype=target_float_dtype,
                target_int_dtype="int32",
                chunk_size=dtype_chunk_size,
                memory_report=dtype_memory_report and verbose,
                convert_objects_to_category=dtype_convert_objects,
            )[0]
            if verbose:
                print("Optimized DataFrame dtypes")

            if _as_bool(config.get("gc_after_dtype_optimization", True)):
                gc.collect()

        if memory_monitor_enabled and verbose:
            frame_memory_mb = features_df.memory_usage(deep=True).sum() / 1024 / 1024
            rss_mb = process.memory_info().rss / 1024 / 1024
            print(
                f"[Memory] final features_df={frame_memory_mb:.2f} MB RSS={rss_mb:.2f} MB"
            )

        failed_features = perf_metrics["failed_features"]
        if failed_features and verbose:
            print(f"\n⚠️  Failed to compute {len(failed_features)} features: {failed_features}")

        return features_df, feature_times

    @classmethod
    def _compute_features_chunked(
        cls,
        df: pd.DataFrame,
        feature_names: List[str],
        chunk_size: int,
        report_interval: int,
        verbose: bool,
    ) -> pd.DataFrame:
        """
        Compute features by processing data in chunks to reduce memory usage
        """
        if verbose:
            print(
                f"Computing {len(feature_names)} features in chunks of {chunk_size} rows..."
            )

        if not feature_names:
            return pd.DataFrame(index=df.index.copy())

        # Split dataframe into chunks
        chunks = []
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            chunks.append((start_idx, chunk))

        if verbose:
            print(f"Processing {len(chunks)} chunks...")

        # Process each chunk
        chunk_results: List[Tuple[int, pd.DataFrame]] = []
        for start_idx, chunk_df in chunks:
            if verbose:
                print(
                    f"Processing chunk {len(chunk_results) + 1}/{len(chunks)} (rows {start_idx}-{start_idx + len(chunk_df) - 1})..."
                )

            # Compute features for this chunk
            chunk_features, _ = cls._compute_features_single(
                chunk_df, feature_names, report_interval, verbose=False
            )

            chunk_results.append((start_idx, chunk_features))

            # Clean up
            del chunk_df
            gc.collect()

        # Combine results
        if verbose:
            print("Combining chunk results...")

        all_features = pd.concat(
            [features for _, features in chunk_results], ignore_index=False
        )
        all_features = all_features.sort_index()  # Ensure proper ordering

        # Clean up
        del chunk_results
        gc.collect()

        return all_features

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for the given dataframe"""
        features = self.get_enabled_features()
        for feature in features:
            func = type(self).get(feature)
            df[feature] = func(df)
        return df

    @classmethod
    def select_features_by_correlation(
        cls, correlation_threshold: float = 0.95, analysis_file: Optional[str] = None
    ) -> List[str]:
        """
        Select features by removing highly correlated ones based on analysis results.

        Args:
            correlation_threshold: Correlation threshold above which features are considered redundant
            analysis_file: Path to feature analysis JSON file (auto-detects latest if None)

        Returns:
            List of selected feature names
        """
        import json
        from pathlib import Path

        # Auto-detect latest analysis file if not provided
        if analysis_file is None:
            reports_dir = Path("reports")
            if reports_dir.exists():
                analysis_files = list(reports_dir.glob("feature_analysis_*.json"))
                if analysis_files:
                    analysis_file = cast(
                        str, max(analysis_files, key=lambda f: f.stat().st_mtime)
                    )
                else:
                    # Fallback to all registered features if no analysis file found
                    return cls.list()
            else:
                return cls.list()

        if analysis_file is None or not Path(analysis_file).exists():
            return cls.list()

        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)

            high_corr_pairs = analysis_data.get("correlation_analysis", {}).get(
                "high_correlation_pairs", []
            )

            # Build correlation graph to identify redundant features
            correlated_features = set()
            for pair in high_corr_pairs:
                if abs(pair["correlation"]) >= correlation_threshold:
                    # Mark the second feature as redundant (keep the first one)
                    correlated_features.add(pair["feature2"])

            # Get all available features
            all_features = set(cls.list())

            # Remove correlated features
            selected_features = all_features - correlated_features

            return sorted(list(selected_features))

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            # Return all features if analysis fails
            return cls.list()

    @classmethod
    def get_optimized_feature_set(
        cls, correlation_threshold: float = 0.95, analysis_file: Optional[str] = None
    ) -> List[str]:
        """
        Get optimized feature set with correlation-based selection.

        This is a convenience method that applies feature selection
        and returns the optimized feature list.
        """
        return cls.select_features_by_correlation(correlation_threshold, analysis_file)
