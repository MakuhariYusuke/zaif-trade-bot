"""
Advanced CSV loading helpers for specialized use-cases.

This module intentionally keeps a narrow surface area so that core loading
remains centralized in ztb.io.data_loader.DataLoader.
"""

from __future__ import annotations

import asyncio
import mmap
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable

import pandas as pd

from ztb.io.data_loader import DataLoader
from ztb.trading.environment.constants import BYTES_PER_MB

def read_csv_mmap(
    file_path: str | Path,
    *,
    chunk_size: int | None = None,
    enable_mmap: bool = True,
    min_mmap_bytes: int = 10 * BYTES_PER_MB,
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV using memory-mapped IO when beneficial.

    Falls back to DataLoader.load_csv_optimized or chunked loading when mmap
    is disabled or the file is below the mmap threshold.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    use_mmap = enable_mmap and file_path.stat().st_size > min_mmap_bytes
    if not use_mmap:
        if chunk_size:
            chunks = []
            for chunk in DataLoader.load_csv_iter(
                file_path, chunksize=chunk_size, **kwargs
            ):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        return DataLoader.load_csv_optimized(file_path, **kwargs)

    with open(file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if chunk_size:
                chunks = []
                offset = 0
                while offset < len(mm):
                    chunk_data = mm[offset : offset + chunk_size * 1000]
                    if not chunk_data:
                        break
                    chunk_df = pd.read_csv(
                        pd.io.common.StringIO(chunk_data.decode("utf-8")),
                        **kwargs,
                    )
                    chunks.append(chunk_df)
                    offset += len(chunk_data)
                return pd.concat(chunks, ignore_index=True)
            content = mm.read().decode("utf-8")
            return pd.read_csv(pd.io.common.StringIO(content), **kwargs)
        finally:
            mm.close()

async def read_csv_async(
    file_path: str | Path,
    *,
    executor: ThreadPoolExecutor | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV asynchronously using a thread executor."""
    loop = asyncio.get_event_loop()
    func = partial(read_csv_mmap, file_path, **kwargs)
    if executor is None:
        return await loop.run_in_executor(None, func)
    return await loop.run_in_executor(executor, func)

async def read_csvs_async(
    file_paths: Iterable[str | Path],
    *,
    max_workers: int = 4,
    **kwargs,
) -> list[pd.DataFrame]:
    """Read multiple CSV files asynchronously."""
    file_list = list(file_paths)
    if not file_list:
        return []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        tasks = [read_csv_async(path, executor=executor, **kwargs) for path in file_list]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)
    finally:
        executor.shutdown(wait=True)

def prefetch_csv(
    file_paths: Iterable[str | Path],
    *,
    max_workers: int = 4,
    **kwargs,
) -> ThreadPoolExecutor:
    """Prefetch CSV files using a thread pool. Returns the executor."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    for path in file_paths:
        executor.submit(read_csv_mmap, path, **kwargs)
    return executor

def read_csv_cached(
    file_path: str | Path,
    *,
    use_cache: bool = True,
    max_cache_entries: int = 32,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV with a simple in-memory LRU cache."""
    # Local import to avoid heavy module-level cache dependencies.
    from collections import OrderedDict

    if not hasattr(read_csv_cached, "_cache"):
        read_csv_cached._cache = OrderedDict()  # type: ignore[attr-defined]
    cache: "OrderedDict[str, pd.DataFrame]" = read_csv_cached._cache  # type: ignore[attr-defined]

    path = Path(file_path)
    cache_key = str(path.resolve())

    if use_cache and cache_key in cache:
        cached = cache.pop(cache_key)
        cache[cache_key] = cached
        return cached.copy()

    df = read_csv_mmap(path, **kwargs)
    if use_cache:
        cache[cache_key] = df.copy()
        while len(cache) > max_cache_entries:
            cache.popitem(last=False)
    return df
