#!/usr/bin/env python3
"""
Data augmentation script to add missing features for v434.2 backtest
"""

import numpy as np
import pandas as pd

from ztb.utils.file_utils import save_csv_data

def add_missing_features(df):
    """Add missing features required by v434.1 schema"""

    # Required features from schema
    required_features = [
        "close",
        "volume",
        "price_change",
        "volume_change",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "atr_14",
        "cci_14",
        "mfi_14",
        "roc_12",
        "mom_10",
    ]

    # Add basic features if missing
    if "price_change" not in df.columns and "close" in df.columns:
        df["price_change"] = df["close"].pct_change().fillna(0)

    if "volume_change" not in df.columns and "volume" in df.columns:
        df["volume_change"] = df["volume"].pct_change().fillna(0)

    # Add technical indicators if missing
    if "rsi_14" not in df.columns and "close" in df.columns:
        # Simple RSI approximation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # Add MACD if missing
    if "macd" not in df.columns and "close" in df.columns:
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Rename macd_histogram to macd_hist if it exists
    if "macd_histogram" in df.columns and "macd_hist" not in df.columns:
        df["macd_hist"] = df["macd_histogram"]

    # Add Bollinger Bands if missing
    if "bb_middle" not in df.columns and "close" in df.columns:
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (std * 2)
        df["bb_lower"] = df["bb_middle"] - (std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # Add Stochastic if missing
    if (
        "stoch_k" not in df.columns
        and "high" in df.columns
        and "low" in df.columns
        and "close" in df.columns
    ):
        lowest_low = df["low"].rolling(window=14).min()
        highest_high = df["high"].rolling(window=14).max()
        df["stoch_k"] = ((df["close"] - lowest_low) / (highest_high - lowest_low)) * 100
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # Add Williams %R if missing
    if (
        "williams_r" not in df.columns
        and "high" in df.columns
        and "low" in df.columns
        and "close" in df.columns
    ):
        highest_high = df["high"].rolling(window=14).max()
        lowest_low = df["low"].rolling(window=14).min()
        df["williams_r"] = (
            (highest_high - df["close"]) / (highest_high - lowest_low)
        ) * -100

    # Add Ichimoku if missing
    if (
        "ichimoku_tenkan" not in df.columns
        and "high" in df.columns
        and "low" in df.columns
    ):
        df["ichimoku_tenkan"] = (
            df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()
        ) / 2
        df["ichimoku_kijun"] = (
            df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()
        ) / 2
        df["ichimoku_senkou_a"] = (
            (df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2
        ).shift(26)
        df["ichimoku_senkou_b"] = (
            (df["high"].rolling(window=52).max() + df["low"].rolling(window=52).min())
            / 2
        ).shift(26)

    # Add ATR if missing
    if (
        "atr_14" not in df.columns
        and "high" in df.columns
        and "low" in df.columns
        and "close" in df.columns
    ):
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean()

    # Add CCI if missing
    if (
        "cci_14" not in df.columns
        and "high" in df.columns
        and "low" in df.columns
        and "close" in df.columns
    ):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(window=14).mean()
        mad = (tp - sma_tp).abs().rolling(window=14).mean()
        df["cci_14"] = (tp - sma_tp) / (0.015 * mad)

    # Add MFI if missing
    if (
        "mfi_14" not in df.columns
        and "high" in df.columns
        and "low" in df.columns
        and "close" in df.columns
        and "volume" in df.columns
    ):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        mf = tp * df["volume"]
        positive_mf = mf.where(tp > tp.shift(), 0)
        negative_mf = mf.where(tp < tp.shift(), 0)
        positive_mf_sum = positive_mf.rolling(window=14).sum()
        negative_mf_sum = negative_mf.rolling(window=14).sum()
        mfr = positive_mf_sum / negative_mf_sum
        df["mfi_14"] = 100 - (100 / (1 + mfr))

    # Add ROC if missing
    if "roc_12" not in df.columns and "close" in df.columns:
        df["roc_12"] = (
            (df["close"] - df["close"].shift(12)) / df["close"].shift(12)
        ) * 100

    # Add Momentum if missing
    if "mom_10" not in df.columns and "close" in df.columns:
        df["mom_10"] = df["close"] - df["close"].shift(10)

    # Add feature_XXX columns (dummy features)
    for i in range(25, 157):  # feature_25 to feature_156
        col_name = f"feature_{i}"
        if col_name not in df.columns:
            # Create dummy features based on existing data
            if "close" in df.columns:
                df[col_name] = df["close"].pct_change(i % 10 + 1).fillna(
                    0
                ) * np.random.normal(0, 0.1, len(df))
            else:
                df[col_name] = np.random.normal(0, 1, len(df))

    # Fill NaN values
    df = df.fillna(0)

    return df

def main():
    # Load existing dataset
    input_file = "data/btc_jpy_yahoo_real_20251021_featured.csv"
    output_file = "data/btc_jpy_v434_extended.csv"

    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # Add missing features
    print("Adding missing features...")
    df_extended = add_missing_features(df)
    print(f"Extended shape: {df_extended.shape}")

    # Save extended dataset
    save_csv_data(df_extended, output_file, index=False)
    print(f"Saved extended dataset to {output_file}")

if __name__ == "__main__":
    main()
