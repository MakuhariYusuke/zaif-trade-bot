import os

import pandas as pd


def merge_datasets():
    existing_path = "data/btc_jpy_1m_dataset.csv"
    new_path = "data/btc_jpy_1m_yfinance.csv"
    output_path = (
        "data/btc_jpy_1m_dataset.csv"  # Overwrite directly? Or create new one first.
    )
    # Let's overwrite directly as the user wants to use it.
    # But for safety, let's backup first.
    backup_path = "data/btc_jpy_1m_dataset.bak"

    print("Loading datasets...")
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        # Backup
        df_existing.to_csv(backup_path, index=False)
        print(f"Backed up existing data to {backup_path}")
    else:
        df_existing = pd.DataFrame()

    df_new = pd.read_csv(new_path)

    print(f"Existing rows: {len(df_existing)}")
    print(f"New rows: {len(df_new)}")

    # Convert timestamps to datetime and UTC
    if not df_existing.empty:
        df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)

    # Normalize volume to BTC
    # Heuristic: If volume > 1000, it's likely JPY volume. Convert to BTC.
    def normalize_volume(row):
        vol = row["volume"]
        price = row["close"]
        if pd.isna(vol) or pd.isna(price) or price == 0:
            return 0.0
        if vol > 1000:
            return vol / price
        return vol

    if not df_existing.empty:
        print("Normalizing volume in existing dataset...")
        df_existing["volume"] = df_existing.apply(normalize_volume, axis=1)

    print("Normalizing volume in new dataset...")
    df_new["volume"] = df_new.apply(normalize_volume, axis=1)

    # Combine
    print("Merging...")
    combined = pd.concat([df_existing, df_new])

    # Sort by timestamp
    combined.sort_values("timestamp", inplace=True)

    # Drop duplicates
    before_dedup = len(combined)
    combined.drop_duplicates(subset="timestamp", keep="first", inplace=True)
    after_dedup = len(combined)

    print(f"Dropped {before_dedup - after_dedup} duplicates.")
    print(f"Total rows after merge: {len(combined)}")

    # Save
    combined.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")


if __name__ == "__main__":
    merge_datasets()
