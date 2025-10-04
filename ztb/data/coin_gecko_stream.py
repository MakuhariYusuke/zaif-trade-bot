"""CoinGecko streaming client with pagination and resilience helpers."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Sequence

import pandas as pd
import requests

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoinGeckoStreamError(RuntimeError):
    """Raised when the CoinGecko client cannot satisfy a request."""


class RateLimitExceeded(CoinGeckoStreamError):
    """Raised when the API rate limit is exceeded after retries."""


@dataclass
class StreamConfig:
    """Configuration for the streaming client."""

    coin_id: str = "bitcoin"
    vs_currency: str = "usd"
    granularity: str = "1m"
    page_size: int = 200
    poll_interval: float = 30.0
    max_empty_polls: int = 4
    overlap: int = (
        3  # number of historical steps to refetch per poll to ensure continuity
    )


@dataclass
class MarketDataBatch:
    """Container describing a batch of market data."""

    dataframe: pd.DataFrame
    fetched_at: datetime
    request_params: Dict[str, Any]


class CoinGeckoStream:
    """Resilient client for CoinGecko market chart streaming."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        timeout: float = 10.0,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        rate_limit_per_minute: int = 45,
        base_url: Optional[str] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_factor = max(backoff_factor, 0.1)
        self.rate_limit_per_minute = max(1, rate_limit_per_minute)
        self.base_url = base_url or self.BASE_URL

        self._call_times: Deque[float] = deque()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_range(
        self,
        coin_id: str,
        vs_currency: str,
        start: datetime,
        end: datetime,
        *,
        granularity: str = "1m",
        page_size: int = 200,
    ) -> MarketDataBatch:
        """Fetch historical market data between two timestamps via pagination."""
        if start >= end:
            raise ValueError("start must be earlier than end")

        step = self._granularity_to_timedelta(granularity)
        if step.total_seconds() <= 0:
            raise ValueError("granularity must be positive")

        page_span = step * max(1, page_size - 1)
        current_start = start
        frames: List[pd.DataFrame] = []

        while current_start < end:
            current_end = min(end, current_start + page_span)
            payload = self._request_market_chart_range(
                coin_id,
                vs_currency,
                current_start,
                current_end,
            )
            frame = self._parse_market_chart(payload)
            frames.append(frame)
            if len(frame) == 0:
                # No more data in this window; advance conservatively to avoid infinite loop
                current_start = current_end + step
            else:
                last_ts = frame["timestamp"].max()
                # Advance start to last timestamp plus a step to avoid duplication
                current_start = (
                    last_ts.to_pydatetime().replace(tzinfo=timezone.utc) + step
                )

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = (
                df.drop_duplicates(subset="timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        else:
            df = pd.DataFrame(columns=["timestamp", "price", "market_cap", "volume"])

        return MarketDataBatch(
            dataframe=df,
            fetched_at=datetime.now(timezone.utc),
            request_params={
                "coin_id": coin_id,
                "vs_currency": vs_currency,
                "start": start,
                "end": end,
                "granularity": granularity,
                "page_size": page_size,
            },
        )

    def stream(
        self,
        config: StreamConfig,
        *,
        start_at: Optional[datetime] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[MarketDataBatch]:
        """Yield batches of live market data."""
        last_timestamp = start_at
        empty_polls = 0
        step = self._granularity_to_timedelta(config.granularity)

        while True:
            if stop_event and stop_event.is_set():
                break

            window_end = datetime.now(timezone.utc)
            if last_timestamp is None:
                window_start = window_end - step * config.page_size
            else:
                window_start = last_timestamp - step * config.overlap
                window_start = max(window_start, window_end - timedelta(days=30))

            batch = self.fetch_range(
                config.coin_id,
                config.vs_currency,
                window_start,
                window_end,
                granularity=config.granularity,
                page_size=config.page_size,
            )

            df = batch.dataframe
            if last_timestamp is not None:
                df = df[df["timestamp"] > pd.Timestamp(last_timestamp)]

            if df.empty:
                empty_polls += 1
                if empty_polls >= config.max_empty_polls:
                    logger.warning(
                        "No new data received from CoinGecko for %s polls (coin=%s)",
                        empty_polls,
                        config.coin_id,
                    )
                    empty_polls = 0
            else:
                empty_polls = 0
                last_timestamp = (
                    df["timestamp"].max().to_pydatetime().replace(tzinfo=timezone.utc)
                )
                yield MarketDataBatch(
                    df.reset_index(drop=True), batch.fetched_at, batch.request_params
                )

            time.sleep(config.poll_interval)

    def close(self) -> None:
        self.session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request_market_chart_range(
        self,
        coin_id: str,
        vs_currency: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        path = f"/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": int(start.replace(tzinfo=timezone.utc).timestamp()),
            "to": int(end.replace(tzinfo=timezone.utc).timestamp()),
        }
        return self._request(path, params)

    def _request(self, path: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise CoinGeckoStreamError(
                        f"connection failure after {self.max_retries} retries"
                    ) from exc
                self._sleep_for_retry(attempt)
                continue

            if response.status_code == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise RateLimitExceeded("CoinGecko rate limit exceeded")
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                delay = max(retry_after, self.backoff_factor * (2**attempt))
                logger.debug("Rate limited by CoinGecko; sleeping for %.2fs", delay)
                time.sleep(delay)
                continue

            if response.status_code >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise CoinGeckoStreamError(
                        f"CoinGecko server error {response.status_code} after retries"
                    )
                self._sleep_for_retry(attempt)
                continue

            if response.status_code >= 400:
                raise CoinGeckoStreamError(
                    f"CoinGecko returned {response.status_code}: {response.text[:200]}"
                )

            try:
                data: Dict[str, Any] = response.json()
                return data
            except ValueError as exc:
                raise CoinGeckoStreamError(
                    "invalid JSON response from CoinGecko"
                ) from exc

    def _parse_market_chart(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        prices = payload.get("prices", [])
        market_caps = payload.get("market_caps", [])
        volumes = payload.get("total_volumes", [])

        if not prices:
            return pd.DataFrame(columns=["timestamp", "price", "market_cap", "volume"])

        def _to_frame(series: Sequence[Sequence[Any]], column: str) -> pd.DataFrame:
            frame = pd.DataFrame(series, columns=["timestamp", column])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame

        price_df = _to_frame(prices, "price")
        cap_df = _to_frame(market_caps, "market_cap") if market_caps else None
        vol_df = _to_frame(volumes, "volume") if volumes else None

        df = price_df
        for extra in (cap_df, vol_df):
            if extra is not None:
                df = df.merge(extra, on="timestamp", how="left")

        df = df.sort_values("timestamp").reset_index(drop=True)
        df[["price", "market_cap", "volume"]] = df[
            ["price", "market_cap", "volume"]
        ].astype("float32")
        return df

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            window = 60.0
            while self._call_times and now - self._call_times[0] > window:
                self._call_times.popleft()
            if len(self._call_times) >= self.rate_limit_per_minute:
                wait_time = window - (now - self._call_times[0]) + 0.01
                logger.debug("Throttling CoinGecko requests for %.2fs", wait_time)
                time.sleep(max(wait_time, 0))
            self._call_times.append(time.monotonic())

    def _sleep_for_retry(self, attempt: int) -> None:
        delay = self.backoff_factor * (2 ** (attempt - 1))
        time.sleep(delay)

    def _granularity_to_timedelta(self, granularity: str) -> timedelta:
        try:
            delta = pd.to_timedelta(granularity)
        except ValueError as exc:
            raise ValueError(f"invalid granularity '{granularity}'") from exc
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            raise ValueError(f"invalid granularity '{granularity}'")
        return timedelta(seconds=int(delta.total_seconds()))

    # Context manager support -------------------------------------------------
    def __enter__(self) -> "CoinGeckoStream":
        return self

    def __exit__(
        self,
        _exc_type: Optional[type],
        _exc: Optional[BaseException],
        _tb: Optional[Any],
    ) -> None:
        self.close()
