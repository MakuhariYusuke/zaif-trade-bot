"""
CoinCheck API を使用して BTC/JPY の1分足データを最新に更新
CoinCheck は日本の取引所で JPY ペアに対応

Usage:
    python scripts/v456/update_data_coincheck.py [--days 30] [--output-file path/to/file.csv]
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import requests
from urllib.parse import urljoin

try:
    from ztb.utils.path_utils import get_project_root
    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

from scripts.v456.data_update_utils import (
    resolve_data_file,
    load_ohlcv_csv,
    save_ohlcv_csv,
    prepare_new_ohlcv,
    validate_ohlcv,
    filter_new_rows,
    merge_ohlcv,
)


class CoinCheckDataFetcher:
    """CoinCheck REST API で BTC/JPY の OHLC データを取得"""
    
    BASE_URL = "https://api.coincheck.com"
    
    # CoinCheck API エンドポイント
    RATE_ENDPOINT = "/api/rate/btc_jpy"  # 現在レート
    TRADES_ENDPOINT = "/api/trades"  # 約定データ
    TICKER_ENDPOINT = "/api/ticker"  # ティッカー
    
    def __init__(self, pair: str = "btc_jpy", rate_limit_delay: float = 0.5):
        """
        Args:
            pair: CoinCheck pair code (default: btc_jpy)
            rate_limit_delay: API呼び出し間隔（秒）
        """
        self.pair = pair
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        # タイムアウト設定（DNS解決とデータ取得）
        self.session.timeout = (5, 10)  # (connect timeout, read timeout)
    
    def fetch_ohlc_from_trades(
        self,
        start_time: datetime,
        end_time: datetime,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        CoinCheck API から約定データを取得してOHLCを合成
        
        Args:
            start_time: 取得開始時刻
            end_time: 取得終了時刻
            max_retries: リトライ回数
            
        Returns:
            OHLC データフレーム
        """
        print(f"[CoinCheck] Fetching trade data from {start_time} to {end_time}")
        
        all_trades = []
        current_page = 1
        max_pages = 10  # リミット設定
        
        # CoinCheck API はページネーション対応
        while current_page <= max_pages:
            try:
                trades = self._get_trades_page(current_page, max_retries=max_retries)
                if not trades:
                    break
                
                all_trades.extend(trades)
                current_page += 1
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"  Error fetching page {current_page}: {e}")
                break
        
        if not all_trades:
            print("[CoinCheck] No trade data fetched")
            return pd.DataFrame()
        
        # 約定データを DataFrame に変換
        df_trades = pd.DataFrame(all_trades)
        
        # タイムスタンプを処理
        if 'timestamp' in df_trades.columns:
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='s', utc=True)
        elif 'created_at' in df_trades.columns:
            df_trades['timestamp'] = pd.to_datetime(df_trades['created_at'], utc=True)
        else:
            print("Warning: No timestamp column found in trades data")
            return pd.DataFrame()
        
        # 時間範囲でフィルタ
        df_trades = df_trades[
            (df_trades['timestamp'] >= start_time) &
            (df_trades['timestamp'] <= end_time)
        ]
        
        if df_trades.empty:
            print(f"[CoinCheck] No trades in date range {start_time} to {end_time}")
            return pd.DataFrame()
        
        # 1分足に集約（OHLCV）
        df_trades = df_trades.set_index('timestamp')
        df_trades['rate'] = pd.to_numeric(df_trades['rate'], errors='coerce')
        df_trades['amount'] = pd.to_numeric(df_trades['amount'], errors='coerce')
        
        # 1分ごとに集約
        ohlc = df_trades['rate'].resample('1min').ohlc()
        volume = df_trades['amount'].resample('1min').sum()
        
        result = ohlc.copy()
        result['volume'] = volume
        result.columns = ['open', 'high', 'low', 'close', 'volume']
        
        print(f"[CoinCheck] Generated {len(result)} OHLC records from {len(df_trades)} trades")
        
        return result
    
    def _get_trades_page(
        self,
        page: int = 1,
        limit: int = 100,
        max_retries: int = 3
    ) -> Optional[List[Dict]]:
        """CoinCheck API から指定ページの約定データを取得"""
        url = urljoin(self.BASE_URL, self.TRADES_ENDPOINT)
        params = {
            'pair': self.pair,
            'limit': limit,
            'page': page,
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, dict) and 'success' in data and not data.get('success'):
                    print(f"  API error: {data.get('error', 'Unknown error')}")
                    return None
                
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    return data
                    
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to fetch trades after {max_retries} attempts")
                    return None
    
    def fetch_recent_ohlc(
        self,
        days: int = 30,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        最近の N日間の OHLC データを取得
        
        Args:
            days: 遡る日数
            max_retries: リトライ回数
            
        Returns:
            OHLC データフレーム
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        print(f"[CoinCheck] Fetching last {days} days ({start_time} to {end_time})")
        
        return self.fetch_ohlc_from_trades(
            start_time=start_time,
            end_time=end_time,
            max_retries=max_retries
        )


class OHLCMerger:
    """既存の OHLC データと新規データをマージ"""
    
    @staticmethod
    def merge_dataframes(
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """既存データと新規データをマージ"""
        if new_df.empty:
            return existing_df
        if existing_df.empty:
            return new_df
        return merge_ohlcv(existing_df, new_df)
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> bool:
        """OHLC データの妥当性を検証"""
        required_columns = ['open', 'high', 'low', 'close']
        
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing columns. Required: {required_columns}")
            return False
        
        # High >= Max(Open, Close) >= Low
        for col in ['open', 'high', 'low', 'close']:
            if not np.all(np.isfinite(df[col].dropna())):
                print(f"Error: {col} contains invalid values")
                return False
        
        invalid_rows = df[df['high'] < df['low']]
        if len(invalid_rows) > 0:
            print(f"Warning: {len(invalid_rows)} rows have high < low")
        
        return True


def update_with_coincheck(
    data_file: Optional[Path] = None,
    days: int = 30
) -> bool:
    """
    CoinCheck API から最新データを取得して既存ファイルを更新
    
    Args:
        data_file: 更新対象のCSVファイル
        days: 遡る日数
        
    Returns:
        成功時 True
    """
    data_file = resolve_data_file(project_root, data_file)
    if data_file is None or not data_file.exists():
        print("Error: Data file not found. Checked default candidates.")
        return False
    
    print(f"Target file: {data_file}")
    
    # 既存データを読み込む
    print("Loading existing data...")
    try:
        df_existing = load_ohlcv_csv(data_file)
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return False

    last_timestamp = df_existing.index.max()
    print(f"Existing data range: {df_existing.index.min()} to {last_timestamp}")
    print(f"Existing rows: {len(df_existing)}")
    
    # CoinCheck から新規データを取得
    print("\n[CoinCheck API] Fetching new data...")
    fetcher = CoinCheckDataFetcher(pair="btc_jpy", rate_limit_delay=0.5)
    
    try:
        df_new = fetcher.fetch_recent_ohlc(days=days, max_retries=3)
    except Exception as e:
        print(f"Error fetching from CoinCheck: {e}")
        print("\nNote: CoinCheck may have API restrictions or maintenance")
        return False
    
    if df_new.empty:
        print("No new data fetched from CoinCheck")
        return False

    try:
        df_new = prepare_new_ohlcv(df_new)
    except Exception as e:
        print(f"Invalid new data format: {e}")
        return False

    df_new_filtered = filter_new_rows(df_existing, df_new)
    if df_new_filtered.empty:
        print("No new data after the last timestamp")
        return False

    ok, reason = validate_ohlcv(
        df_new_filtered,
        min_rows=2,
        expected_interval_seconds=60,
        require_minute_alignment=True,
        require_volume=True,
    )
    if not ok:
        print(f"New data rejected: {reason}")
        return False

    print(f"New data from CoinCheck: {len(df_new_filtered)} records")
    print(f"New data range: {df_new_filtered.index.min()} to {df_new_filtered.index.max()}")

    print("\nMerging data...")
    df_merged = merge_ohlcv(df_existing, df_new_filtered)

    print(f"Merged data: {len(df_merged)} total records")
    print(f"Merged range: {df_merged.index.min()} to {df_merged.index.max()}")

    print(f"\nSaving to {data_file}...")
    try:
        save_ohlcv_csv(data_file, df_merged)
        print(f"✓ Successfully updated {data_file}")
        print(f"  Added {len(df_new_filtered)} new records")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update BTC/JPY 1m data from CoinCheck API"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to fetch (default: 30)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output CSV file (default: auto-detect from data/)"
    )
    
    args = parser.parse_args()
    
    success = update_with_coincheck(
        data_file=args.output_file,
        days=args.days
    )
    
    if not success:
        print("\n" + "="*70)
        print("CoinCheck API から の取得に失敗しました")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
