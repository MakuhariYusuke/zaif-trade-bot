"""
Multi-Timeframe Data Pipeline

Data loading and preprocessing system for multiple timeframes.
Handles data synchronization, resampling, and quality validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.features.timeframe import Timeframe
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultiTimeframeDataPipeline:
    """
    Data pipeline for loading and processing multiple timeframe data.

    Handles data loading, synchronization, resampling, and quality validation
    across multiple timeframes.
    """

    def __init__(self, data_base_path: Optional[str] = None):
        """
        Initialize data pipeline.

        Args:
            data_base_path: Base path for data files
        """
        self.data_base_path = Path(data_base_path or "data")
        self.timeframe_data: Dict[Timeframe, pd.DataFrame] = {}

    def load_timeframe_data(
        self,
        timeframes: List[Timeframe],
        data_files: Optional[Dict[Timeframe, str]] = None,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Load data for multiple timeframes.

        Args:
            timeframes: List of timeframes to load
            data_files: Optional mapping of timeframes to file paths

        Returns:
            Dictionary of loaded dataframes
        """
        loaded_data = {}

        for timeframe in timeframes:
            try:
                if data_files and timeframe in data_files:
                    file_path = data_files[timeframe]
                else:
                    file_path = self._get_default_data_path(timeframe)

                logger.info(f"Loading data for {timeframe.value} from {file_path}")

                df = self._load_single_timeframe(file_path, timeframe)
                if df is not None and not df.empty:
                    loaded_data[timeframe] = df
                    logger.info(f"Loaded {len(df)} rows for {timeframe.value}")
                else:
                    logger.warning(f"No data loaded for {timeframe.value}")

            except Exception as e:
                logger.error(f"Failed to load data for {timeframe.value}: {e}")

        self.timeframe_data = loaded_data
        return loaded_data

    def _get_default_data_path(self, timeframe: Timeframe) -> str:
        """Get default data path for timeframe."""
        # Map timeframes to common file patterns
        timeframe_files = {
            Timeframe.M1: "btc_jpy_1min.csv",
            Timeframe.M5: "btc_jpy_5min.csv",
            Timeframe.M15: "btc_jpy_15min.csv",
            Timeframe.H1: "btc_jpy_1hour.csv",
            Timeframe.H4: "btc_jpy_4hour.csv",
            Timeframe.D1: "btc_jpy_1day.csv",
        }

        filename = timeframe_files.get(timeframe, f"btc_jpy_{timeframe.value}.csv")
        return str(self.data_base_path / filename)

    def _load_single_timeframe(self, file_path: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """Load data for a single timeframe."""
        if not Path(file_path).exists():
            logger.warning(f"Data file not found: {file_path}")
            return None

        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            # Standardize column names
            df = self._standardize_columns(df)

            # Ensure timestamp column exists and is datetime
            df = self._process_timestamps(df)

            # Basic data validation
            df = self._validate_data(df, timeframe)

            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to common format."""
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Timestamp': 'timestamp',
            'Date': 'timestamp',
            'Time': 'timestamp',
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate timestamps."""
        if 'timestamp' not in df.columns:
            # Try to create timestamp from index or other columns
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                raise ValueError("No timestamp column found")

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def _validate_data(self, df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Validate data quality."""
        # Remove rows with all NaN OHLC values
        ohlc_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=ohlc_cols, how='all')

        # Ensure OHLC relationships are logical
        df = df[
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]

        # Remove zero or negative prices
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

        return df

    def synchronize_timeframes(
        self,
        data_dict: Dict[Timeframe, pd.DataFrame],
        base_timeframe: Timeframe = Timeframe.M5,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Synchronize data across timeframes to common timestamps.

        Args:
            data_dict: Dictionary of timeframe data
            base_timeframe: Base timeframe for synchronization

        Returns:
            Synchronized data dictionary
        """
        if base_timeframe not in data_dict:
            logger.warning(f"Base timeframe {base_timeframe.value} not found, using first available")
            base_timeframe = list(data_dict.keys())[0]

        base_df = data_dict[base_timeframe]
        synchronized_data = {base_timeframe: base_df}

        # Synchronize other timeframes to base timeframe
        for timeframe, df in data_dict.items():
            if timeframe == base_timeframe:
                continue

            try:
                synced_df = self._synchronize_to_base(df, base_df, timeframe, base_timeframe)
                synchronized_data[timeframe] = synced_df
                logger.info(f"Synchronized {timeframe.value} to {base_timeframe.value}")
            except Exception as e:
                logger.error(f"Failed to synchronize {timeframe.value}: {e}")

        return synchronized_data

    def _synchronize_to_base(
        self,
        df: pd.DataFrame,
        base_df: pd.DataFrame,
        source_timeframe: Timeframe,
        target_timeframe: Timeframe,
    ) -> pd.DataFrame:
        """Synchronize dataframe to base timeframe timestamps."""
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        base_indexed = base_df.set_index('timestamp')

        # Resample to target frequency if needed
        # This is a simplified approach - in practice, you'd want more sophisticated alignment

        # For now, forward fill to align with base timestamps
        df_aligned = df_indexed.reindex(base_indexed.index, method='ffill')

        # Reset index
        df_aligned = df_aligned.reset_index()

        return df_aligned

    def generate_missing_timeframes(
        self,
        available_data: Dict[Timeframe, pd.DataFrame],
        target_timeframes: List[Timeframe],
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Generate missing timeframe data through resampling.

        Args:
            available_data: Available timeframe data
            target_timeframes: Target timeframes to generate

        Returns:
            Complete timeframe data dictionary
        """
        complete_data = available_data.copy()

        # Find the finest granularity available
        available_timeframes = list(available_data.keys())
        finest_timeframe = min(available_timeframes, key=lambda tf: self._get_timeframe_minutes(tf))

        base_df = available_data[finest_timeframe]

        for target_tf in target_timeframes:
            if target_tf in complete_data:
                continue

            try:
                resampled_df = self._resample_timeframe(base_df, finest_timeframe, target_tf)
                complete_data[target_tf] = resampled_df
                logger.info(f"Generated {target_tf.value} data through resampling")
            except Exception as e:
                logger.error(f"Failed to generate {target_tf.value} data: {e}")

        return complete_data

    def _resample_timeframe(
        self,
        df: pd.DataFrame,
        source_tf: Timeframe,
        target_tf: Timeframe,
    ) -> pd.DataFrame:
        """Resample data from source to target timeframe."""
        df_indexed = df.set_index('timestamp')

        # Map timeframes to pandas frequency strings
        freq_map = {
            Timeframe.M1: '1min',
            Timeframe.M5: '5min',
            Timeframe.M15: '15min',
            Timeframe.H1: '1H',
            Timeframe.H4: '4H',
            Timeframe.D1: '1D',
        }

        source_freq = freq_map[source_tf]
        target_freq = freq_map[target_tf]

        # Resample OHLCV data
        resampled = df_indexed.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        return resampled.reset_index()

    def _get_timeframe_minutes(self, timeframe: Timeframe) -> int:
        """Get timeframe duration in minutes."""
        minute_map = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
        }
        return minute_map.get(timeframe, 60)

    def get_data_quality_report(self, data_dict: Dict[Timeframe, pd.DataFrame]) -> Dict[str, Any]:
        """Generate data quality report."""
        report = {
            'timeframes': {},
            'summary': {},
        }

        for timeframe, df in data_dict.items():
            tf_report = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if not df.empty else None,
                    'end': df['timestamp'].max().isoformat() if not df.empty else None,
                },
                'missing_data': {
                    'total_nan': df.isnull().sum().sum(),
                    'nan_rate': df.isnull().mean().mean(),
                },
                'data_quality': {
                    'valid_ohlc': self._check_ohlc_validity(df),
                    'positive_prices': (df[['open', 'high', 'low', 'close']] > 0).all().all(),
                }
            }
            report['timeframes'][timeframe.value] = tf_report

        # Summary statistics
        report['summary'] = {
            'total_timeframes': len(data_dict),
            'avg_rows_per_timeframe': np.mean([r['row_count'] for r in report['timeframes'].values()]),
            'data_completeness': np.mean([r['data_quality']['valid_ohlc'] for r in report['timeframes'].values()]),
        }

        return report

    def _check_ohlc_validity(self, df: pd.DataFrame) -> float:
        """Check OHLC data validity."""
        if df.empty:
            return 0.0

        valid_rows = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['open'] > 0) &
            (df['close'] > 0)
        )

        return valid_rows.mean()