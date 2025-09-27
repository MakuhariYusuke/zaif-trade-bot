"""
Binance data acquisition module for ZTB.

This module provides functions to fetch historical 1-minute BTC/USDT data
from Binance API and save it in optimized Parquet format.
"""

import pandas as pd
import requests
import time
import logging
from typing import Optional, List
from pathlib import Path
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

logger = logging.getLogger(__name__)

# Binance API configuration
BASE_URL = "https://api.binance.com/api/v3"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000

def fetch_binance_klines(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    limit: int = LIMIT,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    max_retries: int = 5
) -> pd.DataFrame:
    """
    Fetch historical klines from Binance API with rate limiting and retries.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m')
        limit: Number of klines per request (max 1000)
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        max_retries: Maximum number of retries on rate limit

    Returns:
        DataFrame with OHLCV data
    """
    url = f"{BASE_URL}/klines"
    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    backoff_time = 1.0
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data:
                logger.warning("No data received from Binance API")
                return pd.DataFrame()

            # Convert to DataFrame
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]

            df = pd.DataFrame(data, columns=columns)

            # Convert types and normalize timezone
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Normalize Kline boundaries (1-minute intervals should be at :00 seconds)
            df = df[df.index.second == 0]  # type: ignore

            logger.info(f"Fetched {len(df)} klines from Binance (UTC, normalized boundaries)")
            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                if attempt < max_retries:
                    logger.warning(f"Rate limit hit, backing off for {backoff_time}s")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 60)  # Exponential backoff, max 60s
                    continue
                else:
                    logger.error("Max retries exceeded for rate limit")
                    raise
            else:
                logger.error(f"HTTP error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)
                continue
            raise

    return pd.DataFrame()

def fetch_historical_klines(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    days: int = 30,
    max_requests: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch historical klines by making multiple requests.

    Args:
        symbol: Trading pair symbol
        interval: Kline interval
        days: Number of days of historical data to fetch
        max_requests: Maximum number of requests (None for unlimited)

    Returns:
        Combined DataFrame with historical data
    """
    # Calculate start time (days ago)
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    all_data = []
    current_start = start_time
    request_count = 0

    while current_start < end_time:
        if max_requests and request_count >= max_requests:
            break

        df = fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time
        )

        if df.empty:
            break

        all_data.append(df)

        # Update start time for next request (last timestamp + 1ms)
        current_start = int(df.index[-1].timestamp() * 1000) + 1
        request_count += 1

        # Small delay between requests to be respectful
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    logger.info(f"Combined {len(all_data)} requests into {len(combined_df)} total klines")
    return combined_df

def interpolate_missing_data(df: pd.DataFrame, max_gap_minutes: int = 5, remove_long_gaps: bool = True) -> pd.DataFrame:
    """
    Interpolate missing data in the DataFrame.

    Args:
        df: Input DataFrame with OHLCV data
        max_gap_minutes: Maximum gap to fill with forward fill
        remove_long_gaps: Whether to remove gaps longer than 1 hour

    Returns:
        DataFrame with interpolated data
    """
    # Create complete 1-minute index
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min')
    df_reindexed = df.reindex(full_index)

    # Forward fill missing values (max 5 minutes)
    df_filled = df_reindexed.ffill(limit=max_gap_minutes)

    # Remove long gaps (> 1 hour) if enabled
    if remove_long_gaps:
        # Find gaps longer than 1 hour
        missing_mask = df_reindexed.isnull().any(axis=1)
        gap_groups = missing_mask.groupby((missing_mask != missing_mask.shift()).cumsum())
        long_gaps = gap_groups.sum() > 60  # type: ignore

        if long_gaps.any():
            long_gap_indices = []
            for group_id, is_long in long_gaps.items():
                if is_long:
                    group_mask = gap_groups == group_id
                    long_gap_indices.extend(df_reindexed[group_mask].index.tolist())

            df_filled = df_filled.drop(long_gap_indices)
            logger.info(f"Removed {len(long_gap_indices)} rows with long gaps")

    # Remove remaining NaN rows
    df_clean = df_filled.dropna()

    logger.info(f"Interpolated data: {len(df)} -> {len(df_clean)} rows")
    return df_clean

def save_parquet_chunked(df: pd.DataFrame, path: str, chunk: str = 'M', compression: str = 'zstd') -> List[str]:
    """
    Save DataFrame to Parquet files in monthly chunks with zstd compression.

    Args:
        df: DataFrame to save (must have datetime index)
        path: Base path for output files
        chunk: Chunk frequency ('M' for monthly, 'W' for weekly, etc.)
        compression: Compression algorithm

    Returns:
        List of saved file paths
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    saved_files = []
    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Group by chunk period
    for period, group_df in df.groupby(pd.Grouper(freq=chunk.replace('M', 'ME'))):
        # Create filename with period
        if chunk == 'M':
            filename = f"{period.year}-{period.month:02d}.parquet"  # type: ignore
        elif chunk == 'W':
            filename = f"{period.year}-W{period.week:02d}.parquet"  # type: ignore
        else:
            filename = f"{period.strftime('%Y%m%d')}.parquet"  # type: ignore

        file_path = base_path / filename

        # Convert to PyArrow table and save
        table = pa.Table.from_pandas(group_df)
        pq.write_table(table, file_path, compression=compression)
        saved_files.append(str(file_path))

        logger.info(f"Saved {len(group_df)} rows to {file_path}")

    logger.info(f"Saved {len(saved_files)} Parquet files to {base_path}")
    return saved_files

def load_parquet_pattern(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load Parquet files from a directory pattern.

    Args:
        path: Directory path containing Parquet files
        columns: Columns to load (None for all)

    Returns:
        Combined DataFrame
    """
    import glob

    pattern = str(Path(path) / "*.parquet")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {path}")

    dfs = []
    for file_path in files:
        table = pq.read_table(file_path, columns=columns)
        df = table.to_pandas()
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=False)
    combined_df = combined_df.sort_index()

    logger.info(f"Loaded {len(combined_df)} rows from {len(files)} files")
    return combined_df  # type: ignore