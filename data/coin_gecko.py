"""
CoinGecko API data fetcher for cryptocurrency price data.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

import pandas as pd
import requests


class CoinGeckoAPI:
    """CoinGecko API client for cryptocurrency data"""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "zaif-trade-bot/1.0"})
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = 24  # キャッシュ有効期間

    def _get_cache_key(
        self, coin_id: str, vs_currency: str, days: int, interval: str
    ) -> str:
        """Generate cache key for request parameters"""
        key_data = f"{coin_id}_{vs_currency}_{days}_{interval}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            # Check if cache is expired
            file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if file_age_hours > self.cache_expiry_hours:
                return None

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Convert back to DataFrame
            df_data = cache_data["data"]
            df = pd.DataFrame(df_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            print(f"Loaded {len(df)} records from cache (age: {file_age_hours:.1f}h)")
            return df

        except Exception as e:
            print(f"Cache load error: {e}")
            return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save data to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"

            # Convert DataFrame to serializable format
            df_copy = df.reset_index()
            df_copy["timestamp"] = df_copy["timestamp"].astype(str)
            cache_data = {"timestamp": time.time(), "data": df_copy.to_dict("records")}

            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

        except Exception as e:
            print(f"Cache save error: {e}")

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make API request with rate limiting"""
        url = f"{self.BASE_URL}{endpoint}"

        # Rate limiting: CoinGecko allows 10-50 calls per minute
        time.sleep(0.2)  # Conservative delay

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return cast(Dict[str, Any], response.json())

    def get_coin_history(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "jpy",
        days: int = 90,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """
        Get historical price data for a coin.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            vs_currency: Target currency (e.g., 'jpy', 'usd')
            days: Number of days of data to fetch
            interval: Data interval ('daily', 'hourly')

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        # キャッシュチェック
        cache_key = self._get_cache_key(coin_id, vs_currency, days, interval)
        cached_df = self._load_from_cache(cache_key)
        if cached_df is not None:
            return cached_df

        # CoinGecko API endpoint for historical data
        endpoint = f"/coins/{coin_id}/market_chart"

        params = {"vs_currency": vs_currency, "days": days, "interval": interval}

        data = self._make_request(endpoint, params)

        # Parse the response
        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        total_volumes = data.get("total_volumes", [])

        # Create DataFrame from prices data
        # prices format: [[timestamp_ms, price], ...]
        df_data = []

        for i, (timestamp_ms, price) in enumerate(prices):
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

            # Get corresponding market cap and volume if available
            market_cap = market_caps[i][1] if i < len(market_caps) else None
            volume = total_volumes[i][1] if i < len(total_volumes) else None

            df_data.append(
                {
                    "timestamp": timestamp,
                    "open": price,  # OHLC not available, use close
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "market_cap": market_cap,
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # キャッシュ保存
        self._save_to_cache(cache_key, df)

        return df


def fetch_btc_jpy(
    days: int = 365, interval: str = "daily", years: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch BTC/JPY OHLCV price data from CoinGecko.

    Args:
        days: Number of days of historical data to fetch (ignored if years is specified)
        interval: Data interval ('daily' or 'hourly')
        years: Number of years of data to fetch (optional, overrides days)

    Returns:
        DataFrame with OHLCV data for BTC/JPY
    """
    # 複数年データサポート
    if years is not None:
        days = years * 365

    api = CoinGeckoAPI()

    try:
        df = api.get_coin_history(
            coin_id="bitcoin", vs_currency="jpy", days=days, interval=interval
        )

        # Validate data
        if df.empty:
            raise ValueError("No data received from CoinGecko API")

        # Ensure we have close prices
        if "close" not in df.columns:
            raise ValueError("Close price data not available")

        # Fill any missing values with forward fill, then backward fill
        df = df.ffill().bfill()

        # Remove any remaining NaN rows
        df = df.dropna()

        print(f"Fetched {len(df)} BTC/JPY price records from CoinGecko")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch data from CoinGecko API: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing CoinGecko data: {e}")


if __name__ == "__main__":
    # Test the module
    try:
        df = fetch_btc_jpy(days=30, interval="daily")
        print("Sample data:")
        print(df.head())
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error: {e}")
