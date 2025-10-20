#!/usr/bin/env python3
"""
Mirror Augmentation for SELL Bias Mitigation.

Sign-flips return/roc/momentum features to create synthetic downtrend samples.
Swaps BUY/SELL labels to increase SELL training signal.

Usage:
    python scripts/mirror_augment.py \
        --input ml-dataset-enhanced.csv \
        --output ml-dataset-mirrored.csv \
        --ratio 0.3
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ztb.utils.data_utils import load_csv_data_optimized

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Features that should be sign-flipped for mirror augmentation
# These are typically return/momentum/trend features
MIRROR_FEATURES = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "roc_3",
    "roc_5",
    "roc_10",
    "momentum_3",
    "momentum_5",
    "momentum_10",
    "trend_ratio",
    "price_position",
    # Add more return/momentum features as needed
]


def identify_mirror_features(df: pd.DataFrame) -> List[str]:
    """
    Identify features that should be sign-flipped.

    Args:
        df: Input dataframe

    Returns:
        List of column names to mirror
    """
    available = []

    for feature in MIRROR_FEATURES:
        if feature in df.columns:
            available.append(feature)

    # Also detect features by pattern
    for col in df.columns:
        if any(
            pattern in col.lower() for pattern in ["return", "roc", "momentum", "trend"]
        ):
            if col not in available:
                available.append(col)

    return available


def mirror_augment(
    df: pd.DataFrame,
    mirror_features: List[str],
    ratio: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create mirror-augmented dataset.

    Args:
        df: Original dataset
        mirror_features: Features to sign-flip
        ratio: Fraction of data to augment (0.3 = 30% augmentation)
        seed: Random seed

    Returns:
        Augmented dataframe (original + mirrored samples)
    """
    np.random.seed(seed)

    # Sample rows to mirror
    n_mirror = int(len(df) * ratio)
    mirror_indices = np.random.choice(len(df), size=n_mirror, replace=False)

    # Create mirrored subset
    df_mirror = df.iloc[mirror_indices].copy()

    # Sign-flip specified features
    for feature in mirror_features:
        if feature in df_mirror.columns:
            df_mirror[feature] = -df_mirror[feature]

    # Swap BUY/SELL labels (action column)
    if "action" in df_mirror.columns:
        # 0=HOLD, 1=BUY, 2=SELL
        # Swap 1 <-> 2
        action_map = {0: 0, 1: 2, 2: 1}
        df_mirror["action"] = df_mirror["action"].map(action_map)

    # Concatenate original + mirrored
    df_augmented = pd.concat([df, df_mirror], ignore_index=True)

    # Shuffle
    df_augmented = df_augmented.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_augmented


def main():
    parser = argparse.ArgumentParser(
        description="Mirror augmentation for SELL bias mitigation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output augmented dataset CSV",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Augmentation ratio (default: 0.3 = 30%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Mirror Augmentation for SELL Bias Mitigation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Augmentation ratio: {args.ratio:.1%}")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    print("Loading dataset...")
    df = load_csv_data_optimized(args.input)
    print(f"  Original size: {len(df)} rows, {len(df.columns)} columns")

    # Check action distribution (before)
    if "action" in df.columns:
        action_counts_before = df["action"].value_counts().sort_index()
        print("  Action distribution (before):")
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        for action, count in action_counts_before.items():
            name = action_names.get(action, f"Unknown({action})")
            pct = count / len(df) * 100
            print(f"    {name}: {count} ({pct:.1f}%)")
    print()

    # Identify mirror features
    print("Identifying mirror features...")
    mirror_features = identify_mirror_features(df)
    print(f"  Found {len(mirror_features)} features to mirror:")
    for feature in mirror_features[:10]:  # Show first 10
        print(f"    - {feature}")
    if len(mirror_features) > 10:
        print(f"    ... and {len(mirror_features) - 10} more")
    print()

    # Apply augmentation
    print("Applying mirror augmentation...")
    df_augmented = mirror_augment(
        df=df,
        mirror_features=mirror_features,
        ratio=args.ratio,
        seed=args.seed,
    )
    print(
        f"  Augmented size: {len(df_augmented)} rows (+{len(df_augmented) - len(df)})"
    )

    # Check action distribution (after)
    if "action" in df_augmented.columns:
        action_counts_after = df_augmented["action"].value_counts().sort_index()
        print("  Action distribution (after):")
        for action, count in action_counts_after.items():
            name = action_names.get(action, f"Unknown({action})")
            pct = count / len(df_augmented) * 100
            delta = count - action_counts_before.get(action, 0)
            print(f"    {name}: {count} ({pct:.1f}%, +{delta})")
    print()

    # Save
    print("Saving augmented dataset...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_csv(args.output, index=False)
    print(f"✅ Saved to: {args.output}")
    print()

    # Summary
    if "action" in df_augmented.columns:
        sell_improvement = (
            action_counts_after.get(2, 0) / len(df_augmented)
            - action_counts_before.get(2, 0) / len(df)
        ) * 100

        print("Summary:")
        print(
            f"  SELL representation improved by: {sell_improvement:+.1f} percentage points"
        )
        print("  Ready for BC warmstart training")


if __name__ == "__main__":
    main()
