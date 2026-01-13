"""
複数のデータソースから BTC/JPY 1分足データを更新
優先順位: CoinCheck > BitFlyer > YahooFinance

Usage:
    python scripts/v456/update_data_comprehensive.py [--source coincheck|bitflyer|yahoo|all] [--days 30]
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import numpy as np
import traceback

try:
    from ztb.utils.path_utils import get_project_root
    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))


class DataSourceManager:
    """複数のデータソースを管理"""
    
    SOURCES = ['coincheck', 'bitflyer', 'yahoo']
    DEFAULT_PRIORITY = ['coincheck', 'bitflyer', 'yahoo']
    
    def __init__(self):
        self.data_file = self._find_data_file()
        if not self.data_file:
            raise FileNotFoundError("No BTC/JPY data file found in data/")
    
    @staticmethod
    def _find_data_file() -> Optional[Path]:
        """データファイルを自動検出"""
        candidates = [
            project_root / "data" / "btc_jpy_1m_v456.csv",
            project_root / "data" / "btc_jpy_1m_v455.csv",
            project_root / "data" / "btc_jpy_1m_v454.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
    
    def get_existing_data(self) -> pd.DataFrame:
        """既存データを読み込む"""
        try:
            df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
            
            # タイムゾーン統一
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            return df
        except Exception as e:
            print(f"Error loading existing data: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, num_added: int) -> bool:
        """データを保存"""
        try:
            df.to_csv(self.data_file)
            print(f"✓ Successfully updated {self.data_file}")
            print(f"  Added {num_added} new records")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False


class CoinCheckUpdater:
    """CoinCheck 更新エンジン"""
    
    def __init__(self):
        self.module = None
        self._import_module()
    
    def _import_module(self):
        """CoinCheck モジュールをインポート"""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from update_data_coincheck import CoinCheckDataFetcher, OHLCMerger
            self.fetcher_class = CoinCheckDataFetcher
            self.merger_class = OHLCMerger
        except ImportError:
            print("  Note: CoinCheck module not available")
            self.fetcher_class = None
    
    def update(self, existing_df: pd.DataFrame, days: int = 30) -> Optional[pd.DataFrame]:
        """CoinCheck からデータを取得してマージ"""
        if self.fetcher_class is None:
            return None
        
        print("[CoinCheck] Attempting to fetch data...")
        
        try:
            fetcher = self.fetcher_class(pair="btc_jpy", rate_limit_delay=0.5)
            df_new = fetcher.fetch_recent_ohlc(days=days, max_retries=3)
            
            if df_new.empty:
                print("[CoinCheck] No data fetched")
                return None
            
            # フィルタ
            last_timestamp = existing_df.index.max()
            df_new_filtered = df_new[df_new.index > last_timestamp]
            
            if df_new_filtered.empty:
                print("[CoinCheck] No new data after last timestamp")
                return None
            
            print(f"[CoinCheck] ✓ Fetched {len(df_new_filtered)} new records")
            
            # マージ
            merger = self.merger_class()
            merged = merger.merge_dataframes(existing_df, df_new_filtered)
            
            return merged
            
        except Exception as e:
            print(f"[CoinCheck] ✗ Error: {e}")
            return None


class BitFlyerUpdater:
    """BitFlyer 更新エンジン"""
    
    def __init__(self):
        self.module = None
        self._import_module()
    
    def _import_module(self):
        """BitFlyer モジュールをインポート"""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from update_data_bitflyer import BitFlyerDataFetcher, OHLCMerger
            self.fetcher_class = BitFlyerDataFetcher
            self.merger_class = OHLCMerger
        except ImportError:
            print("  Note: BitFlyer module not available")
            self.fetcher_class = None
    
    def update(self, existing_df: pd.DataFrame, days: int = 30) -> Optional[pd.DataFrame]:
        """BitFlyer からデータを取得してマージ"""
        if self.fetcher_class is None:
            return None
        
        print("[BitFlyer] Attempting to fetch data...")
        print("[BitFlyer] Note: BitFlyer REST API has OHLC limitations")
        
        try:
            fetcher = self.fetcher_class(product_code="BTC_JPY", rate_limit_delay=0.5)
            df_new = fetcher.fetch_recent_ohlc(days=days, max_retries=3)
            
            if df_new.empty:
                print("[BitFlyer] No data fetched")
                return None
            
            # フィルタ
            last_timestamp = existing_df.index.max()
            df_new_filtered = df_new[df_new.index > last_timestamp]
            
            if df_new_filtered.empty:
                print("[BitFlyer] No new data after last timestamp")
                return None
            
            print(f"[BitFlyer] ✓ Fetched {len(df_new_filtered)} new records")
            
            # マージ
            merger = self.merger_class()
            merged = merger.merge_dataframes(existing_df, df_new_filtered)
            
            return merged
            
        except Exception as e:
            print(f"[BitFlyer] ✗ Error: {e}")
            return None


class YahooUpdater:
    """YahooFinance 更新エンジン"""
    
    def __init__(self):
        self.yf = None
        self._import_module()
    
    def _import_module(self):
        """yfinance をインポート"""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            print("  Note: yfinance module not available (install: pip install yfinance)")
            self.yf = None
    
    def update(self, existing_df: pd.DataFrame, days: int = 30) -> Optional[pd.DataFrame]:
        """YahooFinance からデータを取得してマージ"""
        if self.yf is None:
            return None
        
        print("[YahooFinance] Attempting to fetch data...")
        
        try:
            ticker = "BTC-JPY"
            interval = "1m"
            period = "7d"  # YahooFinance は通常 7日までしか遡れない
            
            print(f"  Downloading {ticker} ({period}, {interval})...")
            df_new = self.yf.download(ticker, interval=interval, period=period, progress=False)
            
            if df_new.empty:
                print("[YahooFinance] No data fetched")
                return None
            
            # 列名を小文字に統一
            df_new.columns = [c.lower() for c in df_new.columns]
            
            # タイムゾーン統一
            if df_new.index.tz is None:
                df_new.index = df_new.index.tz_localize("UTC")
            else:
                df_new.index = df_new.index.tz_convert("UTC")
            
            # フィルタ
            last_timestamp = existing_df.index.max()
            df_new_filtered = df_new[df_new.index > last_timestamp]
            
            if df_new_filtered.empty:
                print("[YahooFinance] No new data after last timestamp")
                return None
            
            print(f"[YahooFinance] ✓ Fetched {len(df_new_filtered)} new records")
            
            # マージ
            existing_df.index.name = 'timestamp'
            df_new_filtered.index.name = 'timestamp'
            
            merged = pd.concat([existing_df, df_new_filtered])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.sort_index()
            
            return merged
            
        except Exception as e:
            print(f"[YahooFinance] ✗ Error: {e}")
            return None


def update_from_sources(
    sources: list = None,
    days: int = 30,
    data_file: Optional[Path] = None
) -> bool:
    """
    複数のソースからデータを更新
    
    Args:
        sources: データソースのリスト （順序が優先順位）
        days: 遡る日数
        data_file: ターゲットファイル
        
    Returns:
        成功時 True
    """
    if sources is None:
        sources = DataSourceManager.DEFAULT_PRIORITY
    
    # データマネージャー初期化
    try:
        manager = DataSourceManager()
        if data_file is None:
            data_file = manager.data_file
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    print(f"Target file: {data_file}\n")
    
    # 既存データ読み込み
    print("Loading existing data...")
    try:
        df_existing = manager.get_existing_data()
    except Exception as e:
        return False
    
    print(f"  Existing range: {df_existing.index.min()} to {df_existing.index.max()}")
    print(f"  Existing rows: {len(df_existing)}\n")
    
    # 各ソースを試す
    updaters = {
        'coincheck': CoinCheckUpdater(),
        'bitflyer': BitFlyerUpdater(),
        'yahoo': YahooUpdater(),
    }
    
    merged_df = None
    
    for source in sources:
        if source not in updaters:
            print(f"Warning: Unknown source '{source}'")
            continue
        
        updater = updaters[source]
        
        print(f"\n{'='*70}")
        print(f"Trying source: {source.upper()}")
        print(f"{'='*70}")
        
        merged_df = updater.update(df_existing, days=days)
        
        if merged_df is not None:
            print(f"\n✓ Successfully fetched data from {source}")
            print(f"  New range: {merged_df.index.min()} to {merged_df.index.max()}")
            print(f"  Total rows: {len(merged_df)}")
            break
        else:
            print(f"✗ Could not fetch from {source}, trying next...")
    
    # 結果の保存
    if merged_df is None:
        print("\n" + "="*70)
        print("FAILED: No data source succeeded")
        print("="*70)
        print("\nAll available data sources have limitations:")
        print("  1. CoinCheck: API availability, rate limits")
        print("  2. BitFlyer: No direct OHLC endpoint (WebSocket needed)")
        print("  3. YahooFinance: Limited to last 7 days")
        print("\nRecommendation:")
        print("  For production, consider using WebSocket APIs or")
        print("  maintaining a local database with regular updates")
        return False
    
    num_added = len(merged_df) - len(df_existing)
    success = manager.save_data(merged_df, num_added)
    
    return success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update BTC/JPY 1m data from multiple sources"
    )
    parser.add_argument(
        "--source",
        choices=['coincheck', 'bitflyer', 'yahoo', 'all'],
        default='all',
        help="Data source (default: all - tries in priority order)"
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
        help="Output CSV file"
    )
    
    args = parser.parse_args()
    
    # ソース指定を処理
    if args.source == 'all':
        sources = DataSourceManager.DEFAULT_PRIORITY
    else:
        sources = [args.source]
    
    print("="*70)
    print("BTC/JPY 1分足データ更新 - マルチソース対応")
    print("="*70)
    print(f"\nSources (priority order): {' > '.join(sources)}")
    print(f"Days to fetch: {args.days}\n")
    
    success = update_from_sources(
        sources=sources,
        days=args.days,
        data_file=args.output_file
    )
    
    if not success:
        sys.exit(1)
    
    print("\n✓ Data update completed successfully")


if __name__ == "__main__":
    main()
