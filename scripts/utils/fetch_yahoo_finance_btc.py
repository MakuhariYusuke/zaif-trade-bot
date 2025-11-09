#!/usr/bin/env python3
"""
Yahoo Finance BTC/JPY Data Fetcher

Fetch BTC/JPY minute-level data from Yahoo Finance for enhanced trading analysis.
Supports multiple timeframes and data validation.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class YahooFinanceBTCFetcher:
    """Fetcher for BTC/JPY data from Yahoo Finance."""

    def __init__(self):
        """Initialize the fetcher."""
        self.ticker = "BTC-JPY"  # Yahoo Finance ticker for BTC/JPY

    def fetch_minute_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1m",
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch minute-level BTC/JPY data from Yahoo Finance.

        Args:
            start_date: Start date in YYYY-MM-DD format (optional, will use 60 days ago for minute data)
            end_date: End date in YYYY-MM-DD format (optional, will use today)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            max_retries: Maximum number of retries for failed requests

        Returns:
            DataFrame with OHLCV data
        """
        # Auto-adjust dates based on interval limitations
        now = datetime.now()

        if end_date is None:
            end_date = now.strftime("%Y-%m-%d")
        else:
            end_dt = pd.to_datetime(end_date)
            if end_dt > now:
                end_date = now.strftime("%Y-%m-%d")

        # Set start date based on interval limitations
        if start_date is None:
            if interval == "1m":
                start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            elif interval in ["5m", "15m", "30m"]:
                start_date = (now - timedelta(days=60)).strftime("%Y-%m-%d")
            elif interval == "1h":
                start_date = (now - timedelta(days=700)).strftime(
                    "%Y-%m-%d"
                )  # Reduced from 730 to be safe
            else:  # 1d, 1wk, 1mo
                start_date = (now - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        else:
            # Validate and adjust start_date if needed
            start_dt = pd.to_datetime(start_date)
            if interval == "1m" and (now - start_dt).days > 7:
                start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
                logger.warning("Adjusted start_date to 7 days ago for 1m interval")
            elif interval in ["5m", "15m", "30m"] and (now - start_dt).days > 60:
                start_date = (now - timedelta(days=60)).strftime("%Y-%m-%d")
                logger.warning(
                    "Adjusted start_date to 60 days ago for minute intervals"
                )
            elif interval == "1h" and (now - start_dt).days > 700:
                start_date = (now - timedelta(days=700)).strftime("%Y-%m-%d")
                logger.warning("Adjusted start_date to 700 days ago for 1h interval")

        logger.info(
            f"Fetching BTC/JPY data from {start_date} to {end_date} with {interval} interval"
        )

        # Fetch data with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Downloading data...")
                data = yf.download(
                    self.ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    prepost=True,  # Include pre/post market data
                )

                if data.empty:
                    logger.warning(f"No data received for {self.ticker}")
                    continue

                # Clean and format data
                df = self._clean_data(data)
                logger.info(f"Successfully fetched {len(df)} records")
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                continue

        raise RuntimeError(f"Failed to fetch data after {max_retries} attempts")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format the downloaded data."""
        # Reset index to get timestamp as column
        df = df.reset_index()

        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            df.columns = df.columns.get_level_values(0)

        # Rename columns to match our expected format
        column_mapping = {
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }

        df = df.rename(columns=column_mapping)

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Remove any duplicate timestamps
        df = df.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Handle missing values
        df = df.dropna()

        # Add additional columns that might be expected
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Validate data quality
        self._validate_data(df)

        return df

    def _validate_data(self, df: pd.DataFrame):
        """Validate the quality of downloaded data."""
        if len(df) == 0:
            raise ValueError("No data to validate")

        # Check for reasonable price ranges (BTC/JPY should be in millions)
        if df["close"].max() < 1000 or df["close"].min() < 1000:
            logger.warning("Price data seems unusually low. Check ticker symbol.")

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 5:
            logger.warning(f"High percentage of missing values: {missing_pct:.2f}%")

        # Check for price anomalies (sudden jumps)
        price_changes = df["close"].pct_change().abs()
        anomalies = (price_changes > 0.5).sum()  # More than 50% change
        if anomalies > 0:
            logger.warning(f"Found {anomalies} potential price anomalies")

        logger.info(
            f"Data validation complete. Records: {len(df)}, "
            f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )

    def save_data(
        self, df: pd.DataFrame, output_path: str, include_metadata: bool = True
    ):
        """Save data to CSV with optional metadata."""
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if include_metadata:
            # Save metadata
            metadata = {
                "ticker": self.ticker,
                "fetched_at": datetime.now().isoformat(),
                "records": len(df),
                "start_date": str(df["timestamp"].min()),
                "end_date": str(df["timestamp"].max()),
                "columns": list(df.columns),
                "data_quality": {
                    "missing_values_pct": df.isnull().sum().sum()
                    / (len(df) * len(df.columns))
                    * 100,
                    "price_range": [float(df["close"].min()), float(df["close"].max())],
                    "avg_volume": float(df["volume"].mean())
                    if "volume" in df.columns
                    else None,
                },
            }

            metadata_path = output_path.with_suffix(".metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Metadata saved to {metadata_path}")

        # Save data
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Data saved to {output_path}")

    def fetch_multiple_periods(
        self,
        periods: list,
        interval: str = "5m",
        output_dir: str = "data/yahoo_finance",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple time periods.

        Args:
            periods: List of (start_date, end_date, name) tuples
            interval: Data interval
            output_dir: Output directory

        Returns:
            Dictionary of period_name -> DataFrame
        """
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for start_date, end_date, period_name in periods:
            logger.info(f"Fetching data for period: {period_name}")

            try:
                df = self.fetch_minute_data(start_date, end_date, interval)
                results[period_name] = df

                # Save individual period data
                filename = f"btc_jpy_{period_name}_{interval}.csv"
                output_path = output_dir / filename
                self.save_data(df, output_path)

            except Exception as e:
                logger.error(f"Failed to fetch data for {period_name}: {e}")
                continue

        return results

    def get_available_intervals(self) -> Dict[str, str]:
        """Get information about available intervals and their limitations."""
        return {
            "1m": "Last 7 days",
            "5m": "Last 60 days",
            "15m": "Last 60 days",
            "30m": "Last 60 days",
            "1h": "Last 730 days (2 years)",
            "1d": "No limit",
            "1wk": "No limit",
            "1mo": "No limit",
        }


def main():
    """Main function to fetch BTC/JPY data."""
    parser = argparse.ArgumentParser(
        description="Fetch BTC/JPY data from Yahoo Finance"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (optional, auto-adjusted based on interval)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (optional, defaults to today)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
        help="Data interval",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/yahoo_finance/btc_jpy_minute_data.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--show-intervals",
        action="store_true",
        help="Show available intervals and exit",
    )

    args = parser.parse_args()

    if args.show_intervals:
        fetcher = YahooFinanceBTCFetcher()
        intervals = fetcher.get_available_intervals()
        print("Available intervals:")
        for interval, limit in intervals.items():
            print(f"  {interval}: {limit}")
        return

    # Initialize fetcher
    fetcher = YahooFinanceBTCFetcher()

    try:
        # Fetch data
        df = fetcher.fetch_minute_data(args.start_date, args.end_date, args.interval)

        # Display summary
        print("\n📊 データ取得完了:")
        print(f"  レコード数: {len(df)}")
        print(f"  期間: {df['timestamp'].min()} から {df['timestamp'].max()}")
        print(f"  価格範囲: ¥{df['close'].min():,.0f} - ¥{df['close'].max():,.0f}")
        if "volume" in df.columns:
            print(f"  平均出来高: {df['volume'].mean():,.0f}")

        # Save data
        fetcher.save_data(df, args.output)
        print(f"\n✅ データを {args.output} に保存しました。")

    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        raise


if __name__ == "__main__":
    main()
