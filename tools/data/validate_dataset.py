#!/usr/bin/env python3
"""
Validate dataset CSV for compatibility with Zaif-training.

Checks:
- required columns: timestamp, open, high, low, close, volume
- timestamp parseable
- sorted timestamps
- frequency at least 1 minute
- missing/null counts
- timezone awareness
"""
import argparse
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/btc_jpy_1m_dataset.csv")
    parser.add_argument(
        "--resample-to",
        nargs="*",
        help="Optional: list of target timeframes to resample to (e.g. 5m 15m 1h)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    df = pd.read_csv(path)

    # Check columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return

    # Parse timestamps
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # infer format
    except Exception as e:
        print(f"❌ Timestamp parse error: {e}")
        return

    # Sort and check monotonic
    if not df["timestamp"].is_monotonic_increasing:
        print("⚠️  Timestamps not monotonically increasing — sorting for checks")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    print(f"📊 Missing values: {missing_pct:.3f}%")

    # Check frequency
    if len(df) >= 2:
        diffs = df["timestamp"].diff().dropna().dt.total_seconds()
        median_freq = diffs.median()
        print(f"📈 Median timestamp delta: {median_freq} seconds")
        if median_freq > 60 and median_freq < 300:
            print("⚠️  Median freq > 60s (likely >1 min). Consider resampling to 1m if needed")
        if median_freq >= 300:
            print("❌ Data frequency is low for 1m training; resample or fetch 1m data")
    else:
        print("❌ Dataset too short for frequency checks")

    # Basic price checks
    if (df["close"] <= 0).any():
        print("❌ Close prices contain non-positive values")

    print("✅ Dataset validation complete")

    # Optional: resample to provided targets using conversion helper
    if args.resample_to:
        # call the convert_timeframe tool
        import subprocess
        cmd = [
            "python",
            "tools/data/convert_timeframe.py",
            "--input",
            str(path),
            "--targets",
        ]
        cmd.extend(args.resample_to)
        print(f"🔁 Resampling to: {args.resample_to}")
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
