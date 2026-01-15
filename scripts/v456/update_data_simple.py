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
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def update_btc_jpy_data(
    data_file: str = "data/btc_jpy_1m_v451.csv",
    days_back: int = 30
):
    """BTC/JPY データを更新"""
    
    data_path = project_root / data_file
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return False
    
    print(f"📊 データ更新スクリプト")
    print(f"{'='*70}")
    
    # 既存データを読み込み
    print(f"\n📖 既存データを読み込み中...")
    df_existing = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  行数: {len(df_existing):,}")
    print(f"  最古: {df_existing.index.min()}")
    print(f"  最新: {df_existing.index.max()}")
    
    latest_date = df_existing.index.max()
    
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
        df_new = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1m",
            progress=False,
            auto_adjust=False
        )
        
        if df_new.empty:
            print(f"  ⚠️  新しいデータを取得できませんでした")
            return False
        
        print(f"  ✅ 取得しました: {len(df_new)} 行")
        
        # インデックスをタイムゾーン対応にする
        if df_new.index.tz is None:
            df_new.index = df_new.index.tz_localize('UTC')
        
        # 既存データより新しいデータのみを抽出
        df_new = df_new[df_new.index > latest_date]
        
        if len(df_new) == 0:
            print(f"  ℹ️  新しいデータがありません（最新状態）")
            return True
        
        print(f"  🔄 マージ対象: {len(df_new)} 行の新規データ")
        
        # カラム名を統一
        df_new.columns = ['open', 'high', 'low', 'close', 'volume']
        df_new = df_new[['open', 'high', 'low', 'close', 'volume']]  # 順序を修正
        df_new = df_new.drop(columns=[col for col in df_new.columns if col.startswith('Adj')])
        
        # Adj Close を追加（close と同じ）
        df_new['adj_close'] = df_new['close']
        
        # 既存データと結合
        df_merged = pd.concat([df_existing, df_new])
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        df_merged = df_merged.sort_index()
        
        # 保存
        print(f"\n💾 更新データを保存中...")
        df_merged.to_csv(data_path)
        
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
