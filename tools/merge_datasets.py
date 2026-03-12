import glob
import os
import shutil

import pandas as pd


def merge_datasets():
    data_dir = os.path.join(os.getcwd(), "data")
    original_file = os.path.join(data_dir, "btc_jpy_1m_dataset.csv")

    # Find the latest yahoo download
    yahoo_files = glob.glob(os.path.join(data_dir, "btc_jpy_1m_yahoo_*.csv"))
    if not yahoo_files:
        print("No Yahoo data files found.")
        return

    latest_yahoo_file = max(yahoo_files, key=os.path.getctime)
    print(f"Merging {original_file} with {latest_yahoo_file}...")

    try:
        # Load original data
        if os.path.exists(original_file):
            df_orig = pd.read_csv(original_file)
            print(f"Original data: {len(df_orig)} rows")
        else:
            df_orig = pd.DataFrame()
            print("Original file not found, starting fresh.")

        # Load new data
        df_new = pd.read_csv(latest_yahoo_file)
        print(f"New data: {len(df_new)} rows")

        # Concatenate
        df_combined = pd.concat([df_orig, df_new])

        # Convert timestamp to datetime for sorting and deduplication
        df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])

        # Remove duplicates
        df_combined = df_combined.drop_duplicates(subset=["timestamp"])

        # Sort
        df_combined = df_combined.sort_values("timestamp")

        print(f"Combined data: {len(df_combined)} rows")

        # Save to expanded file
        expanded_file = os.path.join(data_dir, "btc_jpy_1m_dataset_expanded.csv")
        df_combined.to_csv(expanded_file, index=False)
        print(f"Saved expanded dataset to {expanded_file}")

        # Backup original and overwrite
        if os.path.exists(original_file):
            backup_file = original_file + ".bak"
            shutil.copy2(original_file, backup_file)
            print(f"Backed up original file to {backup_file}")

        df_combined.to_csv(original_file, index=False)
        print(f"Updated {original_file} with expanded data.")

    except Exception as e:
        print(f"An error occurred during merge: {e}")


if __name__ == "__main__":
    print("Starting merge script...")
    merge_datasets()
