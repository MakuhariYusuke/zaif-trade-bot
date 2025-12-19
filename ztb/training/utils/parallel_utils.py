#!/usr/bin/env python3
"""
CPU Parallel Processing Utilities for SAC v446 Training Optimization

This module provides parallel processing utilities to optimize CPU usage
during training, data preprocessing, and evaluation phases.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CPUParallelProcessor:
    """CPU parallel processing utilities for training optimization."""

    def __init__(self, n_workers: Optional[int] = None, use_multiprocessing: bool = True):
        """
        Initialize parallel processor.

        Args:
            n_workers: Number of worker processes/threads. If None, uses CPU count.
            use_multiprocessing: Whether to use multiprocessing (True) or threading (False).
        """
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)  # Reserve 1 core for system
        self.use_multiprocessing = use_multiprocessing
        self.executor_type = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor

        logger.info(f"Initialized CPUParallelProcessor with {self.n_workers} workers "
                   f"(multiprocessing: {use_multiprocessing})")

    def parallel_map(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """
        Apply function to each item in parallel.

        Args:
            func: Function to apply
            items: List of items to process
            **kwargs: Additional arguments passed to func

        Returns:
            List of results
        """
        if len(items) == 0:
            return []

        # For small datasets, sequential processing might be faster
        if len(items) < self.n_workers:
            logger.debug(f"Dataset too small ({len(items)} items) for parallel processing, using sequential")
            return [func(item, **kwargs) for item in items]

        try:
            with self.executor_type(max_workers=self.n_workers) as executor:
                # Create partial function with kwargs
                def partial_func(item):
                    return func(item, **kwargs)

                results = list(executor.map(partial_func, items))

            logger.debug(f"Parallel processing completed: {len(results)} items processed")
            return results

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}, falling back to sequential")
            return [func(item, **kwargs) for item in items]

    def parallel_batch_process(self, func: Callable, data: np.ndarray,
                             batch_size: int = 1000, **kwargs) -> List[Any]:
        """
        Process data in batches using parallel workers.

        Args:
            func: Function to apply to each batch
            data: Input data array
            batch_size: Size of each batch
            **kwargs: Additional arguments passed to func

        Returns:
            List of batch results
        """
        if len(data) == 0:
            return []

        # Split data into batches
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)

        logger.debug(f"Split data into {len(batches)} batches of size {batch_size}")

        # Process batches in parallel
        return self.parallel_map(func, batches, **kwargs)


class DataLoaderParallelizer:
    """Parallel data loading and preprocessing utilities."""

    def __init__(self, processor: Optional[CPUParallelProcessor] = None):
        """
        Initialize data loader parallelizer.

        Args:
            processor: CPUParallelProcessor instance. If None, creates default one.
        """
        self.processor = processor or CPUParallelProcessor()

    def parallel_csv_loading(self, file_paths: List[str], **read_kwargs) -> List[pd.DataFrame]:
        """
        Load multiple CSV files in parallel.

        Args:
            file_paths: List of CSV file paths
            **read_kwargs: Additional arguments for pd.read_csv

        Returns:
            List of DataFrames
        """
        def load_csv(file_path: str) -> pd.DataFrame:
            try:
                df = pd.read_csv(file_path, **read_kwargs)
                logger.debug(f"Loaded CSV: {file_path} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                return pd.DataFrame()

        return self.processor.parallel_map(load_csv, file_paths)

    def parallel_data_preprocessing(self, dataframes: List[pd.DataFrame],
                                  preprocess_func: Callable[[pd.DataFrame], pd.DataFrame],
                                  **kwargs) -> List[pd.DataFrame]:
        """
        Apply preprocessing function to multiple DataFrames in parallel.

        Args:
            dataframes: List of DataFrames to preprocess
            preprocess_func: Preprocessing function
            **kwargs: Additional arguments for preprocess_func

        Returns:
            List of preprocessed DataFrames
        """
        return self.processor.parallel_map(preprocess_func, dataframes, **kwargs)

    def parallel_feature_engineering(self, dataframes: List[pd.DataFrame],
                                   feature_funcs: List[Callable[[pd.DataFrame], pd.DataFrame]],
                                   **kwargs) -> List[pd.DataFrame]:
        """
        Apply multiple feature engineering functions in parallel.

        Args:
            dataframes: List of DataFrames
            feature_funcs: List of feature engineering functions
            **kwargs: Additional arguments for feature functions

        Returns:
            List of DataFrames with engineered features
        """
        def apply_features(df: pd.DataFrame) -> pd.DataFrame:
            result_df = df.copy()
            for func in feature_funcs:
                try:
                    result_df = func(result_df, **kwargs)
                except Exception as e:
                    logger.error(f"Feature engineering failed: {e}")
                    continue
            return result_df

        return self.processor.parallel_map(apply_features, dataframes)


class NumpyParallelUtils:
    """NumPy/SciPy parallel computation utilities."""

    @staticmethod
    def parallel_array_operations(arrays: List[np.ndarray],
                                operation: str, **kwargs) -> List[np.ndarray]:
        """
        Apply NumPy operations to multiple arrays in parallel.

        Args:
            arrays: List of numpy arrays
            operation: Operation name ('mean', 'std', 'sum', etc.)
            **kwargs: Additional arguments for the operation

        Returns:
            List of operation results
        """
        processor = CPUParallelProcessor(use_multiprocessing=False)  # Use threading for NumPy

        def apply_operation(arr: np.ndarray) -> Any:
            try:
                if hasattr(arr, operation):
                    method = getattr(arr, operation)
                    return method(**kwargs)
                else:
                    # Use numpy function
                    np_func = getattr(np, operation)
                    return np_func(arr, **kwargs)
            except Exception as e:
                logger.error(f"Array operation failed: {e}")
                return None

        return processor.parallel_map(apply_operation, arrays)

    @staticmethod
    def parallel_statistical_tests(data_groups: List[Tuple[np.ndarray, np.ndarray]],
                                 test_type: str = 'ttest_ind', **kwargs) -> List[Dict[str, Any]]:
        """
        Perform statistical tests on multiple data groups in parallel.

        Args:
            data_groups: List of (group1, group2) tuples
            test_type: Type of statistical test
            **kwargs: Additional arguments for the test

        Returns:
            List of test results
        """
        processor = CPUParallelProcessor(use_multiprocessing=False)

        def perform_test(groups: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
            try:
                group1, group2 = groups
                if test_type == 'ttest_ind':
                    stat, p_value = stats.ttest_ind(group1, group2, **kwargs)
                elif test_type == 'mannwhitneyu':
                    stat, p_value = stats.mannwhitneyu(group1, group2, **kwargs)
                elif test_type == 'levene':
                    stat, p_value = stats.levene(group1, group2, **kwargs)
                else:
                    raise ValueError(f"Unsupported test type: {test_type}")

                return {
                    'statistic': stat,
                    'p_value': p_value,
                    'test_type': test_type,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.error(f"Statistical test failed: {e}")
                return {'error': str(e)}

        return processor.parallel_map(perform_test, data_groups)


# Global instances for easy access
default_processor = CPUParallelProcessor()
data_loader = DataLoaderParallelizer(default_processor)


def get_optimal_worker_count() -> int:
    """
    Get optimal number of workers based on system resources.

    Returns:
        Optimal number of workers
    """
    cpu_count = mp.cpu_count()

    # Reserve cores for system operations
    if cpu_count <= 4:
        return max(1, cpu_count - 1)
    elif cpu_count <= 8:
        return cpu_count - 2
    else:
        return cpu_count - 3


def parallel_data_loading_example():
    """Example usage of parallel data loading."""
    # Example file paths
    file_paths = [
        "data/file1.csv",
        "data/file2.csv",
        "data/file3.csv"
    ]

    # Load CSV files in parallel
    dataframes = data_loader.parallel_csv_loading(file_paths)

    # Combine results
    combined_df = pd.concat(dataframes, ignore_index=True)

    logger.info(f"Loaded and combined {len(dataframes)} files into {len(combined_df)} rows")
    return combined_df


if __name__ == "__main__":
    # Test the parallel processor
    logging.basicConfig(level=logging.INFO)

    # Test basic parallel processing
    def square(x):
        return x * x

    numbers = list(range(10))
    results = default_processor.parallel_map(square, numbers)

    print(f"Parallel map results: {results}")

    # Test numpy operations
    arrays = [np.random.randn(1000) for _ in range(5)]
    means = NumpyParallelUtils.parallel_array_operations(arrays, 'mean')

    print(f"Parallel array means: {means}")
