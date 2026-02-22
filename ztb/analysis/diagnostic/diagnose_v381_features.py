#!/usr/bin/env python3
"""
v381特徴量問題診断スクリプト

v381モデル（110特徴量）がバックテストで動作しない問題を診断します。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.io.data_loader import DataLoader
from ztb.training.core.feature_schema_manager import FeatureSchemaManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def diagnose_v381_features() -> None:
    """v381の特徴量問題を診断"""

    print("=" * 80)
    print("v381 Feature Diagnosis")
    print("=" * 80)

    # 1. v381スキーマ読み込み
    try:
        manager = FeatureSchemaManager("ppo_reward_v381_revised_profit_focused")
        metadata = manager.load_schema()

        print("\n✅ v381 Schema loaded")
        print(f"   Model expects: {metadata.num_features} features")
        print(f"   Created at: {metadata.created_at}")
        print(f"   Schema hash: {metadata.schema_hash}")

    except FileNotFoundError as e:
        print("\n❌ v381 Schema NOT found")
        print(f"   Error: {e}")
        print("\n   Solution: Run migration script")
        print(
            "   Command: python scripts/migrate_legacy_schemas.py --model ppo_reward_v381_revised_profit_focused"
        )
        return

    # 2. データセット読み込み
    data_path = "ml-dataset-enhanced.csv"
    try:
        df = DataLoader.load_csv_strict(data_path)
        print(f"\n✅ Dataset loaded: {data_path}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")

    except FileNotFoundError:
        print(f"\n❌ Dataset NOT found: {data_path}")
        return

    # 3. 特徴量比較
    print("\n" + "=" * 80)
    print("Feature Comparison")
    print("=" * 80)

    model_features = set(metadata.feature_names)
    data_features = set(df.columns)

    # 不足している特徴量
    missing_in_data = model_features - data_features
    # 余分な特徴量
    extra_in_data = data_features - model_features
    # 共通の特徴量
    common_features = model_features & data_features

    print("\n📊 Summary:")
    print(f"   Model requires: {len(model_features)} features")
    print(f"   Data provides: {len(data_features)} features")
    print(f"   Common: {len(common_features)} features")
    print(f"   Missing in data: {len(missing_in_data)} features")
    print(f"   Extra in data: {len(extra_in_data)} features")

    # 4. 不足している特徴量の詳細
    if missing_in_data:
        print(f"\n⚠️  PROBLEM: Data is missing {len(missing_in_data)} features!")
        print("\n   Missing features:")
        for i, feat in enumerate(sorted(missing_in_data), 1):
            print(f"   {i:3d}. {feat}")

        print("\n" + "=" * 80)
        print("Root Cause Analysis")
        print("=" * 80)

        print(f"\n📌 v381 was trained with {metadata.num_features} features")
        print(f"📌 Current dataset only has {len(df.columns)} features")
        print("📌 This is why v381 backtest fails with dimension mismatch")

        print("\n" + "=" * 80)
        print("Solutions")
        print("=" * 80)

        print("\n🔧 Option 1: Use Full Feature Dataset")
        print(f"   - Find/create dataset with all {metadata.num_features} features")
        print("   - This is the original dataset used for v381 training")

        print("\n🔧 Option 2: Retrain v381 with Curated Features")
        print(f"   - Train a new v386 model with {len(data_features)} features")
        print(
            '   - Use config: curated_features_list="curated_features.py::CURATED_FEATURES"'
        )

        print("\n🔧 Option 3: Generate Missing Features")
        print("   - Implement feature engineering to create missing features")
        print("   - Add feature generation logic to environment")

    else:
        print("\n✅ SUCCESS: All required features are present!")
        print("   v381 should work with this dataset")

    # 5. v384/v385との比較
    print("\n" + "=" * 80)
    print("Comparison with v384/v385")
    print("=" * 80)

    try:
        manager_v385 = FeatureSchemaManager("ppo_reward_v385_curated")
        metadata_v385 = manager_v385.load_schema()

        v385_features = set(metadata_v385.feature_names)
        v385_missing = v385_features - data_features

        print("\n📊 v385 (curated features):")
        print(f"   Requires: {metadata_v385.num_features} features")
        print(f"   Missing: {len(v385_missing)} features")
        print(f"   Status: {'✅ Works' if len(v385_missing) == 0 else '❌ Fails'}")

        print("\n💡 Insight:")
        print(
            f"   v385 uses curated features → {metadata_v385.num_features} features → Works"
        )
        print(f"   v381 uses all features → {metadata.num_features} features → Fails")
        print("   Recommendation: Use curated feature set for new models")

    except FileNotFoundError:
        pass

    print("\n" + "=" * 80)
    print("End of Diagnosis")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    diagnose_v381_features()
