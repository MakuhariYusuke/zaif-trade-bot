#!/usr/bin/env python3
"""
Resample OHLCV dataset to target timeframe(s).

This script leverages pandas resample and mapping to create target timeframes from
higher frequency data (e.g., 1m -> 5m, 15m, 1h) and saves them to disk (CSV).
It uses the common utilities in `ztb.trading.signal.common.utilities` when possible.
"""
import argparse
from pathlib import Path
import pandas as pd
from ztb.trading.signal.common.utilities import resample_data

# Map user frequencies to pandas offsets
FREQ_MAP = {
    "1m": "1T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


def _validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # parse
    # Sort and drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])  
    df = df.set_index("timestamp")
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate open/high/low/close/volume to target freq.

    Args:
        df: DataFrame indexed by timestamp containing open/high/low/close/volume
        freq: pandas frequency string (e.g., '5T')

    Returns:
        resampled DataFrame with 'open','high','low','close','volume'
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    required = [c for c in agg.keys()]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for OHLCV resampling: {c}")

    resampled = df.resample(freq).agg(agg)
    # Drop rows with NaN in the close (partial aggregation for last bin). If you want to keep, change behavior.
    resampled = resampled.dropna(subset=["close"])  
    resampled = resampled.reset_index()
    return resampled


def map_freq(freq: str) -> str:
    if freq.lower() in FREQ_MAP:
        return FREQ_MAP[freq.lower()]
    return freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/btc_jpy_1m_dataset.csv")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["5m"],
        help="Target timeframes to create. e.g. 5m 15m 1h",
    )
    parser.add_argument("--method", choices=["ohlc", "last", "mean"], default="ohlc")
    parser.add_argument("--outdir", default="data", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    df = _validate_and_sort(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    created = []
    for t in args.targets:
        pandas_freq = map_freq(t)
        if args.method == "ohlc":
            new_df = resample_ohlcv(df, pandas_freq)
        else:
            # Fallback to a simple resample based on mean or last (use existing utility)
            if args.method == "last":
                # reindex to using resample_data which will call .resample internally
                new_df = resample_data(df, pandas_freq, method='last').reset_index()
            else:
                new_df = resample_data(df, pandas_freq, method='mean').reset_index()

        out_file = outdir / f"btc_jpy_{t}_from_{input_path.stem}.csv"
        new_df.to_csv(out_file, index=False)
        created.append(str(out_file))
        print(f"Saved resampled {t} to {out_file}")

    print("✅ Completed conversion. Files created:")
    for p in created:
        print(f" - {p}")


if __name__ == "__main__":
    main()
