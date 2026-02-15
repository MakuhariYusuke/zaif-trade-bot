#!/usr/bin/env python3
"""
medium parquet データ更新スクリプト

yfinance から直近7日の BTC-JPY 1分足を取得し、
既存 medium_features.parquet に追記する。
特徴量は FeatureRegistry で計算。

Usage:
    python scripts/v460/update_medium_parquet.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yfinance as yf

from ztb.features.core.registry import FeatureRegistry

# Feature modules を import して register
import ztb.features.scalping  # noqa: F401
import ztb.features.time.time_features  # noqa: F401

try:
    import ztb.features.generators.technical.volume.chaikin_ad  # noqa: F401
except ImportError:
    pass
try:
    import ztb.features.volatility.normalized_atr  # noqa: F401
except ImportError:
    pass

# Medium parquet の25特徴量
MEDIUM_FEATURES = [
    "RSI", "RSI_M1", "RSI_M5", "RSI_M15", "RSI_H1", "RSI_H4", "RSI_D1",
    "ReturnStdDev",
    "Normalized_ATR",
    "price_velocity", "price_acceleration", "micro_trend", "micro_volatility",
    "momentum_burst", "momentum_divergence",
    "volume_surge", "liquidity_surge",
    "realized_volatility", "spread_pressure",
    "tick_volume_ratio", "order_flow_imbalance",
    "Chaikin_AD",
    "Time_Hour_of_Day", "Time_Session", "Time_Volatility_Adjustment",
]

PARQUET_PATH = project_root / "data" / "btc_jpy_1m_medium_features.parquet"
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp列をtz-naiveに統一"""
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df


def download_recent_data(period: str = "7d") -> pd.DataFrame:
    """yfinance から BTC-JPY 1分足をダウンロード"""
    print(f"[1/4] yfinance BTC-JPY 1m ({period}) ダウンロード中...")
    ticker = yf.Ticker("BTC-JPY")
    hist = ticker.history(period=period, interval="1m")

    if hist.empty:
        raise RuntimeError("yfinance からデータ取得できませんでした")

    ts = hist.index
    if ts.tz is not None:
        ts = ts.tz_localize(None)

    df = pd.DataFrame({
        "timestamp": ts,
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values,
    })
    df = df.reset_index(drop=True)
    print(f"  取得: {len(df)} 行, {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """FeatureRegistry を使って25特徴量を計算"""
    print(f"[2/4] 特徴量計算中 ({len(MEDIUM_FEATURES)} features)...")

    features_dict = {}
    # Time feature用にDatetimeIndexをセット
    df_indexed = df.copy()
    df_indexed.index = pd.DatetimeIndex(df_indexed["timestamp"])
    for feat_name in MEDIUM_FEATURES:
        try:
            func = FeatureRegistry.get(feat_name)
            series = func(df_indexed)
            series.index = df.index  # 元のinteger indexに戻す
            features_dict[feat_name] = series
        except Exception as e:
            print(f"  ⚠️ {feat_name} 計算失敗: {e}")
            features_dict[feat_name] = pd.Series(
                np.zeros(len(df)), index=df.index, dtype=np.float32
            )

    features_df = pd.DataFrame(features_dict)
    result = pd.concat([df[OHLCV_COLS], features_df], axis=1)

    # float32 に統一 (timestamp除く)
    for col in result.columns:
        if col != "timestamp" and result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    print(f"  計算完了: {result.shape}")
    return result


def update_parquet(new_data: pd.DataFrame) -> None:
    """既存parquetに新データを追記"""
    print(f"[3/4] 既存parquet読み込み・マージ中...")

    if not PARQUET_PATH.exists():
        print(f"  ⚠️ 既存ファイルなし、新規作成: {PARQUET_PATH}")
        existing = pd.DataFrame()
    else:
        existing = pd.read_parquet(PARQUET_PATH)
        existing = _normalize_timestamps(existing)
        print(f"  既存: {len(existing)} 行, {existing['timestamp'].min()} ~ {existing['timestamp'].max()}")

    # カラム一致確認
    if not existing.empty:
        existing_cols = set(existing.columns)
        new_cols = set(new_data.columns)
        if existing_cols != new_cols:
            missing = existing_cols - new_cols
            extra = new_cols - existing_cols
            if missing:
                print(f"  ⚠️ 新データに不足列: {missing}")
            if extra:
                print(f"  ⚠️ 新データに余剰列: {extra}")
            # 列を合わせる
            for col in missing:
                new_data[col] = 0.0
            new_data = new_data[[c for c in existing.columns if c in new_data.columns]]

    # 結合
    combined = pd.concat([existing, new_data], ignore_index=True)

    # timestamp でソート & 重複削除
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    ).reset_index(drop=True)

    print(f"  マージ後: {len(combined)} 行, {combined['timestamp'].min()} ~ {combined['timestamp'].max()}")
    added = len(combined) - len(existing)
    print(f"  追加行数: {added}")

    # 保存
    print(f"[4/4] Parquet 保存中...")
    combined.to_parquet(PARQUET_PATH, index=False, compression="snappy")
    size_mb = PARQUET_PATH.stat().st_size / 1e6
    print(f"  保存完了: {size_mb:.1f} MB")


def main() -> None:
    """メイン処理"""
    print("=" * 60)
    print("Medium Parquet データ更新")
    print("=" * 60)

    # ダウンロード
    new_ohlcv = download_recent_data("7d")

    # ウォームアップ用に先頭に既存データの終端を追加
    # RSI等の計算にはウォームアップ期間が必要
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH, columns=OHLCV_COLS)
        existing = _normalize_timestamps(existing)
        # 最後 500行をウォームアップとして使用
        warmup = existing.tail(500).copy()
        warmup_with_new = pd.concat([warmup, new_ohlcv], ignore_index=True)
        # 特徴量計算
        full_features = compute_features(warmup_with_new)
        # ウォームアップ部分を除去
        new_features = full_features.iloc[len(warmup):].reset_index(drop=True)
    else:
        new_features = compute_features(new_ohlcv)

    # 更新
    update_parquet(new_features)

    print("\n✅ 更新完了!")


if __name__ == "__main__":
    main()
