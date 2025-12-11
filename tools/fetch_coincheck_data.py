import logging
import time
from pathlib import Path

import pandas as pd
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoincheckDataFetcher:
    def __init__(self, data_path: str, pair: str = "btc_jpy"):
        self.data_path = Path(data_path)
        self.pair = pair
        self.url = "https://coincheck.com/api/charts/candle_rates"
        self.unit = 60  # 1 minute

        # Ensure directory exists
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if available
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path)
            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
                self.df.set_index("timestamp", inplace=True)
            logger.info(f"Loaded {len(self.df)} rows from {self.data_path}")
        else:
            self.df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            logger.info("Created new DataFrame")

    def fetch_latest_candles(self, limit: int = 300):
        """Fetch latest candles from Coincheck API."""
        params = {"pair": self.pair, "unit": self.unit, "limit": limit}

        try:
            response = requests.get(self.url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Data format: [timestamp, open, high, low, close, volume]
                # Note: Coincheck timestamp is Unix timestamp (seconds)

                new_rows = []
                for row in data:
                    # Handle None values safely
                    def safe_float(val, default=0.0):
                        if val is None:
                            return default
                        return float(val)

                    ts = pd.to_datetime(row[0], unit="s", utc=True)
                    open_p = safe_float(row[1])
                    high_p = safe_float(row[2])
                    low_p = safe_float(row[3])
                    close_p = safe_float(row[4])
                    volume = safe_float(row[5])

                    # If OHLC are 0 (missing), maybe skip or use previous?
                    # Coincheck might return nulls for gaps.
                    if open_p == 0 and close_p == 0:
                        continue

                    new_rows.append(
                        {
                            "timestamp": ts,
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close_p,
                            "volume": volume,
                            "adj_close": close_p,  # For compatibility
                        }
                    )

                if not new_rows:
                    return 0

                new_df = pd.DataFrame(new_rows)
                new_df.set_index("timestamp", inplace=True)

                # Merge with existing data
                # Combine and drop duplicates based on index (timestamp)
                combined = pd.concat([self.df, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)

                added_count = len(combined) - len(self.df)
                self.df = combined

                return added_count
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return 0
        except Exception as e:
            logger.error(f"Fetch Error: {e}")
            return 0

    def save_data(self):
        """Save DataFrame to CSV."""
        # Reset index to save timestamp as column
        save_df = self.df.reset_index()
        save_df.to_csv(self.data_path, index=False)
        logger.info(f"Saved {len(self.df)} rows to {self.data_path}")

    def run_continuous(self, interval_sec: int = 60):
        """Run continuous collection loop."""
        logger.info(f"Starting continuous data collection for {self.pair}...")
        logger.info(f"Saving to {self.data_path}")

        try:
            while True:
                added = self.fetch_latest_candles(
                    limit=300
                )  # Fetch overlap to ensure no gaps
                if added > 0:
                    logger.info(
                        f"Added {added} new candles. Latest: {self.df.index[-1]}"
                    )
                    self.save_data()
                else:
                    logger.debug("No new data.")

                time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            self.save_data()


if __name__ == "__main__":
    # Use the existing dataset path to extend it
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "data" / "btc_jpy_1m_dataset.csv"

    fetcher = CoincheckDataFetcher(str(data_file))

    # Initial fetch
    logger.info("Performing initial fetch...")
    fetcher.fetch_latest_candles(limit=300)
    fetcher.save_data()

    # Start loop
    fetcher.run_continuous()
