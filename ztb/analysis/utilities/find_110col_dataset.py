from pathlib import Path

import pandas as pd

# Check all CSV files for column count
csv_files = [
    "ml-dataset-enhanced.csv",
    "btc_jpy_real_dataset.csv",
    "btc_jpy_yahoo_real_dataset.csv",
    "test_synthetic_dataset.csv",
]

print("Searching for 110-column datasets...\n")

for csv_file in csv_files:
    if Path(csv_file).exists():
        df = pd.read_csv(csv_file)
        print(f"{csv_file}:")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Rows: {len(df)}")
        if len(df.columns) >= 110:
            print("  ✅ MATCHES! (110+ columns)")
        print()
