#!/usr/bin/env python3
"""Check available datasets"""
from pathlib import Path

import pandas as pd

datasets = [
    "btc_jpy_real_dataset.csv",
    "btc_jpy_yahoo_real_dataset.csv",
    "ml-dataset-enhanced.csv",
]

for ds in datasets:
    if Path(ds).exists():
        df = pd.read_csv(ds)
        print(f"{ds}: {len(df)} rows")
        print(f"  Columns: {list(df.columns[:5])}...")
        print(
            f"  Date range: {df.iloc[0].get('timestamp', df.iloc[0].get('date', 'N/A'))} to {df.iloc[-1].get('timestamp', df.iloc[-1].get('date', 'N/A'))}"
        )
        print()
    else:
        print(f"{ds}: NOT FOUND")
        print()
