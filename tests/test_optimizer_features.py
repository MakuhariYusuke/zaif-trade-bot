#!/usr/bin/env python3
"""
Test optimizer features
"""

from ztb.features.optimizer_features import (
    get_optimizer_tracker,
    update_optimizer_features,
)
from ztb.features.registry import FeatureRegistry


def main():
    # FeatureRegistryを初期化
    FeatureRegistry.initialize()

    # Optimizer特徴量が登録されているか確認
    optimizer_features = [
        f for f in FeatureRegistry.list() if f.startswith("optimizer_")
    ]
    print(f"登録されたoptimizer特徴量: {len(optimizer_features)}個")
    for feature in sorted(optimizer_features):
        print(f"  - {feature}")

    # テストデータを更新
    tracker = get_optimizer_tracker()
    update_optimizer_features(
        learning_rate=0.001,
        gradient_norm=0.5,
        step_size=0.01,
        momentum=0.9,
        loss=-2.0,
        update_frequency=1.0,
    )

    # 特徴量ベクトルを取得
    features = tracker.get_feature_vector()
    print(f"\n特徴量ベクトル: {len(features)}個")
    for name, value in sorted(features.items()):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
