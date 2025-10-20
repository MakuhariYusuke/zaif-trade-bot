#!/usr/bin/env python3
"""
Legacy Schema Migration Tool

既存のモデルのスキーマ情報を新しい管理システムに移行します。
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from ztb.training.core.feature_schema_manager import (
    FeatureSchemaManager,
    migrate_legacy_schema,
)
from ztb.utils.config import ZTBConfig
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


# 既存モデルの設定情報（手動で定義）
KNOWN_MODELS = {
    "ppo_reward_v381_revised_profit_focused": {
        "num_features": 68,
        "config": {
            "curated_features_list": None,  # 全特徴量
            "enable_feature_filtering": False,
            "total_timesteps": 100000,  # 推定
            "learning_rate": 0.003,
            "vf_coef": 0.3,
            "target_kl": 0.01,
        },
    },
    "ppo_reward_v384_curated_60": {
        "num_features": 68,
        "config": {
            "curated_features_list": "curated_features.py::CURATED_FEATURES",
            "enable_feature_filtering": True,
            "feature_filter_mode": "whitelist",
            "total_timesteps": 50000,
            "learning_rate": 0.003,
            "vf_coef": 0.3,
            "target_kl": 0.01,
        },
    },
}


def migrate_model(model_name: str, force: bool = False):
    """単一モデルのスキーマを移行"""
    logger.info(f"Migrating schema for: {model_name}")

    # スキーマがすでに存在するか確認
    manager = FeatureSchemaManager(model_name)
    try:
        existing = manager.load_schema()
        if not force:
            logger.warning(f"Schema already exists for {model_name}")
            logger.warning(f"  Features: {existing.num_features}")
            logger.warning("  Use --force to overwrite")
            return False
    except FileNotFoundError:
        pass  # スキーマが存在しない（正常）

    # レガシースキーマを移行
    config = ZTBConfig()
    legacy_schema_path = Path(f"{config.get_model_dir()}/features_schema.json")
    legacy_scaler_path = Path(f"{config.get_model_dir()}/scaler.npz")

    if not legacy_schema_path.exists():
        logger.error(f"Legacy schema not found: {legacy_schema_path}")
        return False

    # 既知の設定を取得
    config = KNOWN_MODELS.get(model_name, {}).get("config", {})

    # 移行実行
    migrate_legacy_schema(
        model_name=model_name,
        legacy_schema_path=legacy_schema_path,
        legacy_scaler_path=legacy_scaler_path,
        config=config,
    )

    logger.info(f"✅ Migration completed for {model_name}")
    return True


def migrate_all_models(force: bool = False):
    """すべての既知モデルを移行"""
    logger.info("=" * 80)
    logger.info("Migrating all known models")
    logger.info("=" * 80)

    success_count = 0
    for model_name in KNOWN_MODELS:
        if migrate_model(model_name, force):
            success_count += 1

    logger.info(
        f"\n✅ Successfully migrated {success_count}/{len(KNOWN_MODELS)} models"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate legacy schemas")
    parser.add_argument("--model", help="Specific model to migrate")
    parser.add_argument("--all", action="store_true", help="Migrate all known models")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing schemas"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available schemas"
    )

    args = parser.parse_args()

    if args.list:
        # 利用可能なスキーマをリスト
        FeatureSchemaManager.print_schema_summary()
        return 0

    if args.all:
        migrate_all_models(args.force)
    elif args.model:
        migrate_model(args.model, args.force)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
