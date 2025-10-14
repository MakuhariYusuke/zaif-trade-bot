#!/usr/bin/env python3
"""
特徴量エンジニアリングスクリプト

btc_jpy_real_dataset.csvに基本的なテクニカル指標を追加
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD"""
    fast_ema = data.ewm(span=fast).mean()
    slow_ema = data.ewm(span=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line, macd - signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def add_technical_features(df):
    """データフレームにテクニカル指標を追加"""
    # 価格変化率
    df['returns'] = df['close'].pct_change()

    # 移動平均
    df['sma_5'] = calculate_sma(df['close'], 5)
    df['sma_10'] = calculate_sma(df['close'], 10)
    df['sma_20'] = calculate_sma(df['close'], 20)

    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)

    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist

    # ボリンジャーバンド
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower

    # ボリンジャーバンドの位置
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

    # ボラティリティ (標準偏差)
    df['volatility_10'] = df['returns'].rolling(window=10).std()

    # 出来高の移動平均
    df['volume_sma_5'] = calculate_sma(df['volume'], 5)

    # 高値-安値の範囲
    df['high_low_range'] = df['high'] - df['low']

    # 価格の対数変化
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    return df

def main():
    # データ読み込み
    data_path = Path("btc_jpy_real_dataset.csv")
    if not data_path.exists():
        print(f"データファイルが見つかりません: {data_path}")
        return

    print("データを読み込み中...")
    df = pd.read_csv(data_path)

    print(f"元のデータ形状: {df.shape}")
    print(f"元のカラム: {list(df.columns)}")

    # 特徴量追加
    print("特徴量を計算中...")
    df_featured = add_technical_features(df)

    print(f"df.shape after add_technical_features: {df.shape}")
    print(f"df_featured.shape: {df_featured.shape}")
    print(f"df.columns: {list(df.columns)}")
    print(f"df_featured.columns: {list(df_featured.columns)}")

    # NaNを削除（最初の数行は計算できない）
    df_featured = df_featured.dropna()

    print(f"特徴量追加後のデータ形状: {df_featured.shape}")
    original_cols = set(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    new_cols = set(df_featured.columns)
    new_features = list(new_cols - original_cols)
    print(f"新しい特徴量: {new_features}")

    # 保存
    output_path = Path("btc_jpy_featured_dataset.csv")
    df_featured.to_csv(output_path, index=False)
    print(f"特徴量付きデータを保存しました: {output_path}")

    # 統計情報表示
    print("\n特徴量の統計情報:")
    feature_cols = [col for col in df_featured.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    print(df_featured[feature_cols].describe())

if __name__ == "__main__":
    main()