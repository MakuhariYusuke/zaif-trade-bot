#!/usr/bin/env python3
"""
データ分析スクリプト
トレーニングデータの基本統計を分析します
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ztb.utils.data_utils import load_csv_data


def analyze_training_data() -> None:
    """トレーニングデータを分析"""

    data_path = Path("ml-dataset-enhanced.csv")
    if not data_path.exists():
        print(f"データファイルが見つかりません: {data_path}")
        return

    print("=== トレーニングデータ分析 ===")

    # データ読み込み（メモリ節約のため制限）
    print("データを読み込み中...")
    df = load_csv_data(data_path, nrows=50000)

    print("\n=== データ基本情報 ===")
    print(f"総行数: {len(df):,}")
    print(f"カラム数: {len(df.columns)}")
    print(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"データ型数: {df.dtypes.nunique()}")

    print("\n=== 時間範囲 ===")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
        valid_ts = df["ts"].dropna()
        if len(valid_ts) > 0:
            print(f"開始: {valid_ts.min()}")
            print(f"終了: {valid_ts.max()}")
            print(f"期間: {(valid_ts.max() - valid_ts.min()).days} 日")

    print("\n=== 取引統計 ===")
    if "win" in df.columns:
        win_rate = df["win"].mean() * 100
        print(f"勝率: {win_rate:.1f}%")
    if "pnl" in df.columns:
        print(f"平均PnL: {df['pnl'].mean():.4f}")
        print(f"総PnL: {df['pnl'].sum():.2f}")
        print(f"PnL標準偏差: {df['pnl'].std():.4f}")
        print(f"最大PnL: {df['pnl'].max():.4f}")

    print("\n=== 特徴量サンプル ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"数値特徴量数: {len(numeric_cols)}")
    print("サンプル特徴量:", list(numeric_cols[:10]))

    print("\n=== 欠損値分析 ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print("欠損値のあるカラム:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({missing_pct[col]}%)")  # type: ignore
    else:
        print("欠損値なし")

    print("\n=== 分析完了 ===")


if __name__ == "__main__":
    analyze_training_data()  # type: ignore
