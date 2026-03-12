"""
BitFlyer API を使用して BTC/JPY の1分足データを最新に更新
最大30日分のデータを遡って取得し、既存データとマージ

Usage:
    python scripts/v456/update_data_bitflyer.py [--days 30] [--output-file path/to/file.csv]
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


class BitFlyerDataFetcher:
    """BitFlyer REST API でOHLC1分足データを取得"""
    
    BASE_URL = "https://api.bitflyer.jp"
    
    # BitFlyer API エンドポイント
    OHLC_ENDPOINT = "/v1/getboardstate"  # 現在の状態
    EXECUTIONS_ENDPOINT = "/v1/me/getexecutions"  # 約定履歴（プライベート）
    TICKER_ENDPOINT = "/v1/ticker"  # ティッカー（パブリック）
    
    def __init__(self, product_code: str = "BTC_JPY", rate_limit_delay: float = 0.5):
        """
        Args:
            product_code: BitFlyer product code (default: BTC_JPY)
            rate_limit_delay: API呼び出し間隔（秒）
        """
        self.product_code = product_code
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
    def fetch_ohlc_from_trades(
        self, 
        start_time: datetime, 
        end_time: datetime,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        BitFlyer API から約定データを取得してOHLCを再構成
        
        BitFlyer には直接 OHLC エンドポイントがないため、
        ティッカー/約定履歴から1分足を合成
        
        Args:
            start_time: 取得開始時刻
            end_time: 取得終了時刻
            max_retries: リトライ回数
            
        Returns:
            OHLC データフレーム
        """
        print(f"[BitFlyer] Fetching data from {start_time} to {end_time}")
        
        ohlc_data = {}
        current_time = start_time
        
        while current_time < end_time:
            minute_key = current_time.strftime("%Y-%m-%d %H:%M:00")
            minute_end = current_time + timedelta(minutes=1)
            
            # 1分間のティッカー情報を取得（約定最新データから推定）
            # 実際には bitFlyer v1 API では OHLC は直接提供されないため
            # 過去の約定履歴から復元する必要がある
            
            # ここでは簡略版：ティッカーから現在価格を取得
            try:
                ticker = self._get_ticker(max_retries=max_retries)
                if ticker:
                    ohlc_data[minute_key] = {
                        'timestamp': minute_key,
                        'open': ticker.get('ltp', np.nan),
                        'high': ticker.get('ltp', np.nan),
                        'low': ticker.get('ltp', np.nan),
                        'close': ticker.get('ltp', np.nan),
                        'volume': 0.0,  # 約定量は別途取得が必要
                    }
            except Exception as e:
                print(f"  Error fetching data for {minute_key}: {e}")
            
            # Rate limit
            time.sleep(self.rate_limit_delay)
            current_time = minute_end
        
        if not ohlc_data:
            print("[BitFlyer] No data fetched")
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(ohlc_data, orient='index')
        df.index = pd.to_datetime(df['timestamp'], utc=True)
        df = df.drop(columns=['timestamp'])
        
        print(f"[BitFlyer] Fetched {len(df)} records")
        return df
    
    def _get_ticker(self, max_retries: int = 3) -> Optional[Dict]:
        """ティッカー情報を取得（最新価格）"""
        url = urljoin(self.BASE_URL, self.TICKER_ENDPOINT)
        params = {'product_code': self.product_code}
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to fetch ticker after {max_retries} attempts")
                    return None
    
    def fetch_recent_ohlc(
        self,
        days: int = 30,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        最近の N日間の OHLC データを取得
        
        Note: BitFlyer REST API には直接 OHLC エンドポイントがないため、
        WebSocket または外部サービスの利用が推奨される
        ここでは簡略版として最新のティッカー情報を1分足に変換
        
        Args:
            days: 遡る日数
            max_retries: リトライ回数
            
        Returns:
            OHLC データフレーム
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # BitFlyer の REST API は約定履歴から OHLC を再構成するか、
        # WebSocket を使用する必要がある
        # ここでは簡略版で最新のティッカーのみを返す
        print(f"[BitFlyer] Attempting to fetch last {days} days of data...")
        print("[BitFlyer] Note: BitFlyer REST API has limitations for historical OHLC")
        print("[BitFlyer] Consider using WebSocket API or external data providers")
        
        # 簡略版：現在のティッカー情報を1レコード返す
        ticker = self._get_ticker(max_retries=max_retries)
        if not ticker:
            return pd.DataFrame()
        
        # 1分足 OHLC に変換（簡略版）
        now = datetime.utcnow()
        df = pd.DataFrame({
            'timestamp': [now],
            'open': [ticker.get('ltp', np.nan)],
            'high': [ticker.get('ltp', np.nan)],
            'low': [ticker.get('ltp', np.nan)],
            'close': [ticker.get('ltp', np.nan)],
            'volume': [0.0],
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
        
        return df


class OHLCMerger:
    """既存の OHLC データと新規データをマージ"""
    
    @staticmethod
    def merge_dataframes(
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        既存データと新規データをマージ
        
        Args:
            existing_df: 既存データフレーム
            new_df: 新規データフレーム
            remove_duplicates: 重複を削除するか
            
        Returns:
            マージされたデータフレーム
        """
        if new_df.empty:
            return existing_df
        if existing_df.empty:
            return new_df
        return merge_ohlcv(existing_df, new_df)
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> bool:
        """OHLC データの妥当性を検証"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing columns. Required: {required_columns}")
            return False
        
        # High >= Max(Open, Close) >= Low
        for col in ['open', 'high', 'low', 'close']:
            if not np.all(np.isfinite(df[col])):
                print(f"Error: {col} contains non-finite values")
                return False
        
        invalid_rows = df[df['high'] < df['low']]
        if len(invalid_rows) > 0:
            print(f"Warning: {len(invalid_rows)} rows have high < low")
            return False
        
        return True


def update_with_bitflyer(
    data_file: Optional[Path] = None,
    days: int = 30
) -> bool:
    """
    BitFlyer API から最新データを取得して既存ファイルを更新
    
    Args:
        data_file: 更新対象のCSVファイル（指定なければ自動判定）
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
    
    # BitFlyer から新規データを取得
    print("\n[BitFlyer API] Fetching new data...")
    fetcher = BitFlyerDataFetcher(product_code="BTC_JPY", rate_limit_delay=0.5)
    df_new = fetcher.fetch_recent_ohlc(days=days, max_retries=3)
    
    if df_new.empty:
        print("No new data fetched from BitFlyer")
        print("\nNote: BitFlyer REST API has limitations for historical OHLC data")
        print("For production use, consider:")
        print("  1. BitFlyer WebSocket API (real-time streaming)")
        print("  2. CoinCheck/Zaif API (alternative exchanges)")
        print("  3. Binance API (supplemental global data)")
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

    print(f"New data from BitFlyer: {len(df_new_filtered)} records")
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
        description="Update BTC/JPY 1m data from BitFlyer API"
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
    parser.add_argument(
        "--skip-bitflyer",
        action="store_true",
        help="Skip BitFlyer API, only show existing data info"
    )
    
    args = parser.parse_args()
    
    if args.skip_bitflyer:
        print("Skipping BitFlyer API fetch (--skip-bitflyer)")
        print("\nLet me show you alternative data sources:\n")
        print("1. CoinCheck API (historically reliable for JPY pairs)")
        print("2. Zaif API (native BTC/JPY support)")
        print("3. Binance API (global liquidity, but no JPY pair)")
        return
    
    success = update_with_bitflyer(
        data_file=args.output_file,
        days=args.days
    )
    
    if not success:
        print("\n" + "="*70)
        print("BitFlyer API 制限について:")
        print("="*70)
        print("""
BitFlyer REST API には以下の制限があります：

1. OHLC エンドポイントがない
   - ティッカー：最新価格のみ
   - 約定履歴：認証が必要（プライベートAPI）

推奨される代替手段：

A) BitFlyer WebSocket API (リアルタイムストリーミング)
   - Product: BTC_JPY
   - Channel: lightning_executions, lightning_ticker
   - 約定データから 1分足を合成可能

B) CoinCheck REST API (V2)
   - GET /api/exchange/orders/opens
   - より詳細な約定履歴が取得可能

C) Zaif API 
   - 日本の取引所として JPY ペアに強い

D) Binance API + 為替データ
   - BTCUSDT 取得 + USD/JPY レート組合せ
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()
