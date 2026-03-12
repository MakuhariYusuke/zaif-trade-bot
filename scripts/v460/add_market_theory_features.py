#!/usr/bin/env python3
"""379# P3-B3: full_registry Parquet に 5 市場理論特徴量を追加.

既存 data/btc_jpy_1m_full_registry_features.parquet (77列) に
market_theory.py の 5 FeatureRegistry 特徴量を追加 → 82列化.

Usage:
    python scripts/v460/add_market_theory_features.py
"""

import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Feature modules を import して register
import ztb.features.market_theory  # noqa: F401
from ztb.features.core.registry import FeatureRegistry

PARQUET_PATH = project_root / "data" / "btc_jpy_1m_full_registry_features.parquet"

NEW_FEATURES = [
    "parkinson_sigma",
    "vpin_proxy",
    "kyle_lambda_proxy",
    "amihud_illiq",
    "ema_velocity_bps",
]


def main() -> None:
    print("=" * 60)
    print("379# P3-B3: Market Theory Features → full_registry Parquet")
    print("=" * 60)

    if not PARQUET_PATH.exists():
        print(f"ERROR: {PARQUET_PATH} not found")
        sys.exit(1)

    print(f"Loading {PARQUET_PATH}...")
    t0 = time.time()
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Loaded: {df.shape} in {time.time() - t0:.1f}s")

    force = "--force" in sys.argv
    added = 0
    for feat_name in NEW_FEATURES:
        if feat_name in df.columns and not force:
            print(f"  {feat_name}: already exists, skipping (use --force to recalculate)")
            continue
        if feat_name in df.columns:
            print(f"  {feat_name}: recalculating (--force)...")
        else:
            print(f"  {feat_name}: not found, computing...")
        print(f"  Computing {feat_name} (new)...")
        func = FeatureRegistry.get(feat_name)
        series = func(df)
        df[feat_name] = series.astype(np.float32)
        nan_count = int(df[feat_name].isna().sum())
        print(
            f"    mean={df[feat_name].mean():.6f}, "
            f"std={df[feat_name].std():.6f}, "
            f"nan={nan_count}"
        )
        added += 1

    if added == 0:
        print("No new features to add. Exiting.")
        return

    print(f"\nFinal shape: {df.shape} (+{added} features)")
    print("Saving...")
    t1 = time.time()
    df.to_parquet(PARQUET_PATH, index=False)
    size_mb = os.path.getsize(PARQUET_PATH) / 1e6
    print(f"  Saved in {time.time() - t1:.1f}s ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
