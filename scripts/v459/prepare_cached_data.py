#!/usr/bin/env python3
"""
Pre-cache data files to avoid SIGINT issues during training.

This script converts CSV files to cached formats (feather/parquet) with
timestamps already converted, avoiding pandas C extension issues on Windows.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import os

os.environ.setdefault(
    "ZTB_SIGINT_POLICY", "ignore" if os.name == "nt" else "default"
)

from ztb.utils.data_utils import load_csv_data_cached

def main():
    """Pre-cache all data files."""
    data_dir = project_root / "data"
    
    # Main data file
    data_files = [
        data_dir / "btc_jpy_1m_v451.csv",
    ]
    
    print("=" * 70)
    print("Data Pre-Caching Tool")
    print("=" * 70)
    
    for data_file in data_files:
        if not data_file.exists():
            print(f"⚠️  Skipping {data_file.name} (not found)")
            continue
        
        print(f"\n📄 Processing: {data_file.name}")
        try:
            df = load_csv_data_cached(data_file, force_refresh=True)
            print(f"✅ Cached: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Show timestamp info if present
            if "timestamp" in df.columns:
                print(f"   Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            return 1
    
    print("\n" + "=" * 70)
    print("✅ All data files cached successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
