"""
Data pipeline scheduler for automated Binance data acquisition.

Uses APScheduler for cron-like scheduling of data fetching tasks.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ztb.notifications import DiscordNotifier
from ztb.data.binance_data import fetch_historical_klines, interpolate_missing_data, save_parquet_chunked

logger = logging.getLogger(__name__)

class DataAcquisitionScheduler:
    """
    Automated scheduler for Binance data acquisition.

    Schedules daily data fetching at midnight, with retry logic and notifications.
    """

    def __init__(self, data_dir: str = "data/binance", max_retries: int = 3):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.scheduler = BlockingScheduler()
        self.notifier = DiscordNotifier()

    def _fetch_daily_data(self):
        """
        Fetch yesterday's data from Binance.
        Called daily at midnight.
        """
        try:
            # Calculate yesterday's date range
            yesterday = datetime.now() - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

            logger.info(f"Starting daily data fetch for {yesterday.date()}")

            # Fetch data with retries
            df = None
            for attempt in range(self.max_retries):
                try:
                    df = fetch_historical_klines(days=1, max_requests=50)
                    if not df.empty:
                        break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(60 * (attempt + 1))  # Exponential backoff

            if df is None or df.empty:
                error_msg = f"Failed to fetch data for {yesterday.date()} after {self.max_retries} attempts"
                logger.error(error_msg)
                self.notifier.send_notification("Data Fetch Error", error_msg, "error")
                return

            # Interpolate missing data
            df_clean = interpolate_missing_data(df)

            # Save to Parquet
            save_path = self.data_dir / "daily"
            files = save_parquet_chunked(df_clean, str(save_path))

            success_msg = f"Successfully fetched and saved {len(df_clean)} records for {yesterday.date()}"
            logger.info(success_msg)
            self.notifier.send_notification("Data Fetch Success", success_msg, "info")

        except Exception as e:
            error_msg = f"Unexpected error in daily data fetch: {e}"
            logger.error(error_msg)
            self.notifier.send_notification("Data Save Error", error_msg, "error")

    def schedule_daily_fetch(self, hour: int = 0, minute: int = 0):
        """
        Schedule daily data fetching.

        Args:
            hour: Hour of day (0-23)
            minute: Minute of hour (0-59)
        """
        trigger = CronTrigger(hour=hour, minute=minute)
        self.scheduler.add_job(
            self._fetch_daily_data,
            trigger=trigger,
            id='daily_binance_fetch',
            name='Daily Binance Data Fetch',
            max_instances=1,
            replace_existing=True
        )
        logger.info(f"Scheduled daily data fetch at {hour:02d}:{minute:02d}")

    def start(self):
        """Start the scheduler"""
        logger.info("Starting data acquisition scheduler")
        self.notifier.send_notification("Scheduler", "Data acquisition scheduler started", "info")
        self.scheduler.start()

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping data acquisition scheduler")
        self.scheduler.shutdown()
        self.notifier.send_notification("Scheduler", "Data acquisition scheduler stopped", "info")

    def run_once(self):
        """Run data fetch once for testing"""
        self._fetch_daily_data()