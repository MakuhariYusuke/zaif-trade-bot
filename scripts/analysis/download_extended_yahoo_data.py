#!/usr/bin/env python3
"""
Extended Yahoo Finance Data Retrieval for Action Signal Guide Testing

This script retrieves longer historical data from Yahoo Finance for comprehensive
backtesting of the Action Signal Guide patterns.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_extended_yahoo_data(
    symbol: str = "BTC-JPY",
    start_date: str = "2020-01-01",
    end_date: str = None,
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Download extended historical data from Yahoo Finance.

    Args:
        symbol: Trading symbol (default: BTC-JPY)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
        interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Downloading {symbol} data from {start_date} to {end_date} (interval: {interval})")

    try:
        # Download data
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )

        if data.empty:
            logger.error(f"No data retrieved for {symbol}")
            return pd.DataFrame()

        # Clean column names
        data.columns = data.columns.str.lower()
        data = data.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adj close': 'adj_close',
            'volume': 'volume'
        })

        # Reset index to have timestamp as column
        data = data.reset_index()

        logger.info(f"Downloaded {len(data)} records for {symbol}")
        return data

    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return pd.DataFrame()

def save_extended_dataset(data: pd.DataFrame, output_path: Path) -> None:
    """
    Save extended dataset to CSV and pickle formats.

    Args:
        data: DataFrame to save
        output_path: Base output path (without extension)
    """
    if data.empty:
        logger.error("No data to save")
        return

    # Save as CSV
    csv_path = output_path.with_suffix('.csv')
    data.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Save as pickle for faster loading
    pkl_path = output_path.with_suffix('.pkl')
    data.to_pickle(pkl_path)
    logger.info(f"Saved pickle: {pkl_path}")

    # Log data statistics
    logger.info(f"Data range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    logger.info(f"Total records: {len(data)}")
    logger.info(f"Missing values: {data.isnull().sum().sum()}")

def main():
    """Main function to download extended Yahoo Finance data."""
    # Configuration
    symbol = "BTC-JPY"
    start_date = "2018-01-01"  # Extended historical data
    end_date = datetime.now().strftime("%Y-%m-%d")
    interval = "1h"  # 1-hour intervals for pattern analysis

    # Output path
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = data_dir / f"btc_jpy_yahoo_extended_{timestamp}"

    # Ensure data directory exists
    data_dir.mkdir(exist_ok=True)

    # Download data
    data = download_extended_yahoo_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    if not data.empty:
        # Basic data validation
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return

        # Save data
        save_extended_dataset(data, output_base)

        # Create summary
        summary = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "total_records": len(data),
            "date_range": {
                "start": str(data['timestamp'].min()),
                "end": str(data['timestamp'].max())
            },
            "data_quality": {
                "missing_values": data.isnull().sum().to_dict(),
                "duplicates": data.duplicated().sum()
            }
        }

        # Save summary
        summary_path = output_base.with_suffix('.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Extended dataset created successfully: {output_base}")
        logger.info(f"Summary saved: {summary_path}")

    else:
        logger.error("Failed to download extended dataset")

if __name__ == "__main__":
    main()