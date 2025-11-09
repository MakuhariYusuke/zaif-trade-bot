#!/usr/bin/env python3
"""
Check curated features
"""

from ztb.features.curated_features import CURATED_FEATURES


def main():
    print(f"CURATED_FEATURESの総数: {len(CURATED_FEATURES)}個")
    print("Optimizer特徴量:")
    optimizer_features = [f for f in CURATED_FEATURES if f.startswith("optimizer_")]
    for feature in sorted(optimizer_features):
        print(f"  - {feature}")


if __name__ == "__main__":
    main()
