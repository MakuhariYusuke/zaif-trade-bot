"""
Yahoo Finance から BTC/JPY 1分足データを更新する共通ロジック。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from ztb.utils.path_utils import get_project_root

    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

from scripts.v456.data_source_update_common import collect_new_rows, update_file_with_fetcher
from scripts.v456.data_update_utils import fetch_yahoo_ohlcv


YAHOO_VALIDATE_KWARGS = {
    "min_rows": 1,
    "expected_interval_seconds": 60,
    "require_minute_alignment": False,
    "require_volume": False,
}


class YahooDataFetcher:
    """Yahoo Finance から BTC/JPY OHLCV を取得する fetcher。"""

    def __init__(self, ticker: str = "BTC-JPY", interval: str = "1m"):
        self.ticker = ticker
        self.interval = interval

    def close(self) -> None:
        # セッションは保持しないため no-op（共通インターフェース維持用）。
        return None

    def fetch_recent_ohlc(self, days: int = 7, max_retries: int = 3) -> pd.DataFrame:
        del max_retries  # yfinance は内部で retry されるため未使用。
        capped_days = max(1, min(int(days), 7))
        period = f"{capped_days}d"
        return fetch_yahoo_ohlcv(ticker=self.ticker, interval=self.interval, period=period)


def collect_yahoo_rows(existing_df: pd.DataFrame, days: int = 7) -> Optional[pd.DataFrame]:
    """オーケストレーション用: Yahoo の増分行のみを取得して返す。"""
    fetcher = YahooDataFetcher(ticker="BTC-JPY", interval="1m")
    try:
        return collect_new_rows(
            existing_df=existing_df,
            fetcher=fetcher,
            source_name="YahooFinance",
            days=days,
            validate_kwargs=YAHOO_VALIDATE_KWARGS,
            max_retries=3,
        )
    finally:
        fetcher.close()


def update_with_yahoo(data_file: Optional[Path] = None, days: int = 7) -> bool:
    """Yahoo を使って対象CSVを更新する。"""
    return update_file_with_fetcher(
        project_root=project_root,
        data_file=data_file,
        source_name="YahooFinance",
        days=days,
        fetcher_factory=YahooDataFetcher,
        fetcher_kwargs={"ticker": "BTC-JPY", "interval": "1m"},
        validate_kwargs=YAHOO_VALIDATE_KWARGS,
        max_retries=3,
    )

