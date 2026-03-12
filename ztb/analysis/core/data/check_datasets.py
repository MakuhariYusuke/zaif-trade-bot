#!/usr/bin/env python3
"""Check available datasets"""
from pathlib import Path

from ztb.io.data_loader import DataLoader

def check_dataset_quality(dataset_path: str = None) -> dict[str, any]:
    """Check dataset quality and return information.

    Args:
        dataset_path: Path to specific dataset to check, or None to check all

    Returns:
        Dictionary with dataset information
    """
    datasets = [
        "btc_jpy_real_dataset.csv",
        "btc_jpy_yahoo_real_dataset.csv",
        "ml-dataset-enhanced.csv",
    ]

    if dataset_path:
        datasets = [dataset_path]

    results = {}

    for ds in datasets:
        if Path(ds).exists():
            df = DataLoader.load_csv_strict(ds)
            info = {
                "exists": True,
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": df.iloc[0].get("timestamp", df.iloc[0].get("date", "N/A")),
                    "end": df.iloc[-1].get("timestamp", df.iloc[-1].get("date", "N/A")),
                },
            }
            results[ds] = info
            print(f"{ds}: {len(df)} rows")
            print(f"  Columns: {list(df.columns[:5])}...")
            print(
                f"  Date range: {info['date_range']['start']} to {info['date_range']['end']}"
            )
            print()
        else:
            results[ds] = {"exists": False}
            print(f"{ds}: NOT FOUND")
            print()

    return results

if __name__ == "__main__":
    check_dataset_quality()
