import pandas as pd


def merge_long_history():
    long_path = "data/btc_jpy_5m_yfinance.csv"
    existing_path = "data/btc_jpy_1m_dataset.csv"
    output_path = "data/btc_jpy_1m_dataset.csv"
    backup_path = "data/btc_jpy_1m_dataset_pre_long.bak"

    print("Loading datasets...")
    df_long = pd.read_csv(long_path)
    df_existing = pd.read_csv(existing_path)

    # Backup
    df_existing.to_csv(backup_path, index=False)
    print(f"Backed up existing data to {backup_path}")

    # Process Long Data (5m)
    df_long["timestamp"] = pd.to_datetime(df_long["timestamp"], utc=True)
    df_long.set_index("timestamp", inplace=True)

    # Resample to 1m
    # Upsample and interpolate
    # We want to fill the gaps between 5m points
    print("Upsampling 5m data to 1m...")

    # Create a full 1m index
    full_idx = pd.date_range(
        start=df_long.index.min(), end=df_long.index.max(), freq="1min"
    )
    df_long_1m = df_long.reindex(full_idx)

    # Interpolate prices
    cols_to_interp = ["open", "high", "low", "close"]
    for col in cols_to_interp:
        df_long_1m[col] = df_long_1m[col].interpolate(method="linear")

    # Distribute volume
    # First fill NaNs with 0? No, we want to distribute the 5m volume across the 5 minutes.
    # Actually, reindex puts the value at the timestamp (e.g. 00:00), and NaNs at 00:01, 00:02, 00:03, 00:04.
    # So we should divide the value at 00:00 by 5 and spread it?
    # Or just forward fill and divide by 5?
    # Simple approach: fillna(0) then rolling mean? No.
    # Let's just divide the non-NaN volume by 5, then ffill?
    # Or simply: volume is usually "volume during the interval".
    # If 5m volume is V, then 1m volume is V/5 (avg).
    df_long_1m["volume"] = df_long_1m["volume"].fillna(
        0
    )  # This is wrong if we want to distribute.

    # Correct way:
    # The value at T is the volume for T to T+5m? Or T-5m to T?
    # Yahoo timestamp is usually start of interval.
    # So volume at 00:00 is for 00:00-00:05.
    # We want to put V/5 at 00:00, 00:01, 00:02, 00:03, 00:04.

    # Let's reload to handle volume correctly
    df_long_1m_vol = df_long[["volume"]].reindex(full_idx)
    df_long_1m_vol["volume"] = df_long_1m_vol["volume"].ffill(limit=4) / 5
    df_long_1m["volume"] = df_long_1m_vol["volume"]

    # Reset index
    df_long_1m.reset_index(inplace=True)
    df_long_1m.rename(columns={"index": "timestamp"}, inplace=True)

    # Normalize volume (JPY to BTC)
    # Yahoo 5m volume is likely JPY.
    def normalize_volume(row):
        vol = row["volume"]
        price = row["close"]
        if pd.isna(vol) or pd.isna(price) or price == 0:
            return 0.0
        if vol > 1000:  # Threshold
            return vol / price
        return vol

    print("Normalizing volume in long dataset...")
    df_long_1m["volume"] = df_long_1m.apply(normalize_volume, axis=1)

    # Process Existing Data
    df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)

    # Merge
    # We want to keep existing data where available, and use long data to fill gaps.
    # Combine both
    print("Merging...")
    combined = pd.concat([df_existing, df_long_1m])

    # Sort
    combined.sort_values("timestamp", inplace=True)

    # Drop duplicates
    # Keep 'first' (existing data was first in concat? No, existing is first in list)
    # But we want to prioritize existing data (real 1m) over interpolated data.
    # So we should ensure existing data comes *before* interpolated data for the same timestamp?
    # Actually, if we concat [existing, long], and existing has a row for T, and long has a row for T.
    # drop_duplicates(keep='first') will keep existing.
    # But wait, long data covers the whole range.
    # So for every T in existing, there is a T in long.
    # We want to keep existing.
    # So [existing, long] order is correct with keep='first'.

    before_dedup = len(combined)
    combined.drop_duplicates(subset="timestamp", keep="first", inplace=True)
    after_dedup = len(combined)

    print(
        f"Dropped {before_dedup - after_dedup} duplicates (interpolated points replaced by real data)."
    )
    print(f"Total rows after merge: {len(combined)}")

    # Save
    combined.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")


if __name__ == "__main__":
    merge_long_history()
