#!/usr/bin/env python3
"""
BTC/JPY データ拡張スクリプト

既存の 12月8日のデータから、現在までのデータを
過去のボラティリティパターンを使用してシミュレート拡張
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def extend_btc_data(
    data_file: str = "data/btc_jpy_1m_v451.csv",
):
    """既存データから現在までをシミュレートで拡張"""
    
    data_path = project_root / data_file
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return False
    
    print(f"📊 BTC/JPY データ拡張スクリプト")
    print(f"{'='*70}")
    
    # 既存データを読み込み
    print(f"\n📖 既存データを読み込み中...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  行数: {len(df):,}")
    print(f"  最古: {df.index.min()}")
    print(f"  最新: {df.index.max()}")
    
    latest_date = df.index.max()
    current_date = datetime.now().replace(second=0, microsecond=0, tzinfo=df.index[-1].tz)
    
    # 既にほぼ現在の場合はスキップ
    minutes_gap = (current_date - latest_date).total_seconds() / 60
    print(f"  時間差: {minutes_gap:.0f} 分")
    
    if minutes_gap < 60:
        print(f"\n✅ データがほぼ現在までありますので拡張不要です")
        return True
    
    print(f"\n🔄 拡張データを生成中...")
    
    # リターンの統計量を計算
    close = df['close'].values
    returns = np.diff(close) / close[:-1]
    mean_return = returns.mean()
    std_return = returns.std()
    
    print(f"  過去のボラティリティ統計:")
    print(f"    平均リターン: {mean_return:.6f} ({mean_return*100:.4f}%)")
    print(f"    標準偏差: {std_return:.6f} ({std_return*100:.4f}%)")
    
    # シミュレーション用に拡張
    last_close = float(df['close'].iloc[-1])
    new_rows = []
    
    current = latest_date + timedelta(minutes=1)
    np.random.seed(42)  # 再現性
    
    minutes_to_add = int(minutes_gap)
    for i in range(minutes_to_add):
        # リターンをシミュレート（AR(1)過程）
        if i == 0:
            simulated_return = np.random.normal(mean_return, std_return)
        else:
            simulated_return = 0.5 * returns[-(i % 100)] + np.random.normal(mean_return * 0.5, std_return * 0.5)
        
        new_close = last_close * (1 + simulated_return)
        
        # OHLC を生成
        intra_vol = abs(simulated_return) * 0.5
        new_high = new_close + last_close * intra_vol * np.random.uniform(0.3, 0.7)
        new_low = new_close - last_close * intra_vol * np.random.uniform(0.3, 0.7)
        new_open = last_close
        
        # ボリューム（通常の80-120%の範囲）
        avg_volume = df['volume'].mean()
        new_volume = avg_volume * np.random.uniform(0.8, 1.2)
        
        new_rows.append({
            'timestamp': current,
            'open': float(new_open),
            'high': float(max(new_open, new_close, new_high)),
            'low': float(min(new_open, new_close, new_low)),
            'close': float(new_close),
            'volume': float(new_volume),
            'adj_close': float(new_close),
        })
        
        last_close = new_close
        current += timedelta(minutes=1)
    
    df_new = pd.DataFrame(new_rows)
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
    df_new.set_index('timestamp', inplace=True)
    
    print(f"  ✅ {len(df_new):,} 行の拡張データを生成")
    
    # マージ（既存データ + 新規シミュレーション）
    df_extended = pd.concat([df, df_new])
    df_extended = df_extended[~df_extended.index.duplicated(keep='last')]
    df_extended = df_extended.sort_index()
    
    # 既存カラムは保持（基本的なOHLCV のみ）
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
    if len(df.columns) > len(base_cols):
        print(f"\n⚠️  注意: 拡張データは基本的な OHLCV のみです")
        print(f"  元のデータには {len(df.columns)} カラムありますが、")
        print(f"  シミュレーション後は {len(base_cols)} カラムのみです")
        print(f"\n  追加特徴量（RSI, MACD等）は別途計算してください")
        
        # 既存の追加カラムは削除
        df_extended = df_extended[base_cols]
    
    # 保存
    print(f"\n💾 拡張データを保存中...")
    df_extended.to_csv(data_path)
    
    print(f"  ✅ 保存完了")
    print(f"  新しいデータ範囲:")
    print(f"    最古: {df_extended.index.min()}")
    print(f"    最新: {df_extended.index.max()}")
    print(f"    行数: {len(df_extended):,}")
    print(f"    期間: {(df_extended.index.max() - df_extended.index.min()).days} 日")
    
    return True


if __name__ == "__main__":
    success = extend_btc_data()
    
    if success:
        print(f"\n✅ データ拡張完了")
        print(f"\n📝 注意: 拡張データはシミュレーション値です")
        print(f"   本番運用前には、以下のいずれかの方法で")
        print(f"   実際のマーケットデータに差し替えてください:")
        print(f"   1. Zaif API から直接取得")
        print(f"   2. BitFlyer WebSocket で取得")
        print(f"   3. 外部データプロバイダから購入")
        sys.exit(0)
    else:
        print(f"\n❌ データ拡張失敗")
        sys.exit(1)
