#!/usr/bin/env python3
"""
BTC/JPY 1分足データを yfinance で更新（シンプル版）

既存のデータを読み込み、最新の日付から現在までのデータを追加取得
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

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
    fetch_yahoo_ohlcv,
)


def update_btc_jpy_data(
    data_file: str = "data/btc_jpy_real_dataset.csv",
    days_back: int = 30
):
    """BTC/JPY データを更新"""
    
    data_path = resolve_data_file(project_root, Path(data_file))
    
    if data_path is None or not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return False
    
    print(f"📊 データ更新スクリプト")
    print(f"{'='*70}")
    
    # 既存データを読み込み
    print(f"\n📖 既存データを読み込み中...")
    df_existing = load_ohlcv_csv(data_path)
    print(f"  行数: {len(df_existing):,}")
    print(f"  最古: {df_existing.index.min()}")
    print(f"  最新: {df_existing.index.max()}")
    
    # yfinance で最新データを取得
    print(f"\n🌐 yfinance から最新データを取得中...")
    try:
        import yfinance as yf
        
        # 現在より30日前から現在まで取得
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"  期間: {start_date.date()} ～ {end_date.date()}")
        
        # BTC/JPY は BTC-JPY (yfinance の形式)
        ticker = "BTC-JPY"
        df_new = fetch_yahoo_ohlcv(
            ticker=ticker,
            interval="1m",
            start=start_date,
            end=end_date,
        )
        
        if df_new.empty:
            print(f"  ⚠️  新しいデータを取得できませんでした")
            return False
        
        print(f"  ✅ 取得しました: {len(df_new)} 行")
        
        df_new = prepare_new_ohlcv(df_new)
        df_new = filter_new_rows(df_existing, df_new)

        if len(df_new) == 0:
            print(f"  ℹ️  新しいデータがありません（最新状態）")
            return True
        
        print(f"  🔄 マージ対象: {len(df_new)} 行の新規データ")

        ok, reason = validate_ohlcv(
            df_new,
            min_rows=1,
            expected_interval_seconds=60,
            require_minute_alignment=True,
            require_volume=False,
        )
        if not ok:
            print(f"  ❌ データ品質チェック失敗: {reason}")
            return False

        df_merged = merge_ohlcv(df_existing, df_new)
        
        # 保存
        print(f"\n💾 更新データを保存中...")
        save_ohlcv_csv(data_path, df_merged)
        
        print(f"  ✅ 保存完了: {len(df_merged):,} 行")
        print(f"  新しい日付範囲:")
        print(f"    最古: {df_merged.index.min()}")
        print(f"    最新: {df_merged.index.max()}")
        print(f"    期間: {(df_merged.index.max() - df_merged.index.min()).days + 1} 日")
        
        return True
        
    except ImportError:
        print(f"  ❌ yfinance がインストールされていません")
        print(f"  pip install yfinance")
        return False
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return False


if __name__ == "__main__":
    success = update_btc_jpy_data(days_back=30)
    
    if success:
        print(f"\n✅ データ更新完了")
        sys.exit(0)
    else:
        print(f"\n❌ データ更新失敗")
        sys.exit(1)
