#!/usr/bin/env python3
# サンプルデータ生成スクリプト

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timedelta

def generate_sample_data(start_date, end_date, filename):
    """サンプル取引データを生成"""
    # 日付範囲の作成
    date_range = pd.date_range(start=start_date, end=end_date, freq='1min')

    n_points = len(date_range)

    # 基本価格データ（ランダムウォーク）
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.001, n_points)  # 0.1%の標準偏差
    base_price = 50000  # BTC価格のベース
    prices = base_price * np.exp(np.cumsum(price_changes))

    # テクニカル指標の生成
    data = {
        'ts': date_range,
        'price': prices,
        'volume': np.random.exponential(100, n_points),

        # 移動平均
        'sma_5': pd.Series(prices).rolling(5).mean(),
        'sma_10': pd.Series(prices).rolling(10).mean(),
        'sma_20': pd.Series(prices).rolling(20).mean(),
        'sma_50': pd.Series(prices).rolling(50).mean(),

        # RSI
        'rsi_14': 50 + 30 * np.sin(np.linspace(0, 4*np.pi, n_points)),

        # MACD
        'macd': pd.Series(prices).ewm(span=12).mean() - pd.Series(prices).ewm(span=26).mean(),
        'macd_signal': (pd.Series(prices).ewm(span=12).mean() - pd.Series(prices).ewm(span=26).mean()).ewm(span=9).mean(),
        'macd_hist': (pd.Series(prices).ewm(span=12).mean() - pd.Series(prices).ewm(span=26).mean()) - (pd.Series(prices).ewm(span=12).mean() - pd.Series(prices).ewm(span=26).mean()).ewm(span=9).mean(),

        # ボリンジャーバンド
        'bb_upper': pd.Series(prices).rolling(20).mean() + 2 * pd.Series(prices).rolling(20).std(),
        'bb_middle': pd.Series(prices).rolling(20).mean(),
        'bb_lower': pd.Series(prices).rolling(20).mean() - 2 * pd.Series(prices).rolling(20).std(),

        # ATR (Average True Range)
        'atr_14': pd.Series([abs(prices[i] - prices[i-1]) if i > 0 else 0.001 for i in range(n_points)]).rolling(14).mean(),

        # ストキャスティクス
        'stoch_k': 50 + 40 * np.sin(np.linspace(0, 6*np.pi, n_points)),
        'stoch_d': 50 + 35 * np.sin(np.linspace(0, 6*np.pi, n_points)),

        # Williams %R
        'williams_r': -50 + 40 * np.sin(np.linspace(0, 8*np.pi, n_points)),

        # CCI (Commodity Channel Index)
        'cci_20': np.random.normal(0, 100, n_points),

        # ADX (Average Directional Index)
        'adx_14': 20 + 30 * np.random.random(n_points),

        # 価格変動率
        'price_change_1': pd.Series(prices).pct_change(1),
        'price_change_5': pd.Series(prices).pct_change(5),
        'price_change_10': pd.Series(prices).pct_change(10),

        # ボラティリティ
        'volatility_5': pd.Series(prices).pct_change().rolling(5).std(),
        'volatility_10': pd.Series(prices).pct_change().rolling(10).std(),
        'volatility_20': pd.Series(prices).pct_change().rolling(20).std(),

        # 取引所情報
        'exchange': 'sample_exchange',
        'pair': 'BTC/USD',
    }

    df = pd.DataFrame(data)

    # NaNをゼロで埋める
    df = df.fillna(0)

    # Parquetファイルとして保存
    df.to_parquet(filename, index=False)
    print(f"Generated sample data: {filename} ({len(df)} rows)")

    return df

def main():
    """メイン関数"""
    # 各月のデータ生成
    months_data = [
        ('2025-04-01', '2025-04-30', 'data/features/2025/04/sample_04.parquet'),
        ('2025-05-01', '2025-05-31', 'data/features/2025/05/sample_05.parquet'),
        ('2025-06-01', '2025-06-30', 'data/features/2025/06/sample_06.parquet'),
        ('2025-07-01', '2025-07-31', 'data/features/2025/07/sample_07.parquet'),
        ('2025-08-01', '2025-08-31', 'data/features/2025/08/sample_08.parquet'),
        ('2025-09-01', '2025-09-20', 'data/features/2025/09/sample_09.parquet'),
    ]

    for start_date, end_date, filename in months_data:
        generate_sample_data(start_date, end_date, filename)

    print("All sample data generated successfully!")

if __name__ == '__main__':
    main()