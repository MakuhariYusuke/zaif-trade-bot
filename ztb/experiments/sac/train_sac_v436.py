#!/usr/bin/env python3
"""
SAC v436 Signal Guided Training - Classical Technical Signal Integration
古典的テクニカルシグナル統合による学習加速
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.utils.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="SAC v436 Signal Guided Training")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps from config",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v436_signal_guided_config.json",
        help="Path to configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    print("🚀 SAC v436 Signal Guided Training")
    print("=" * 50)

    # 設定ファイル読み込み
    config_path = Path(args.config)
    print(f"🔍 Looking for config at: {config_path.resolve()}")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # コマンドライン引数でtimestepsを上書き
    if args.timesteps is not None:
        original_timesteps = config["training"]["total_timesteps"]
        config["training"]["total_timesteps"] = args.timesteps
        print(f"📋 Timesteps overridden: {original_timesteps:,} → {args.timesteps:,}")

    print("📋 Configuration loaded:")
    print(f"  - Model: {config['training']['model_name']}")
    print(f"  - Timesteps: {config['training']['total_timesteps']:,}")
    print(f"  - Algorithm: {config['training']['algorithm']}")
    print(
        f"  - Guidance Mode: {config['training']['environment']['reward_settings']['signal_guidance']['guidance_mode']}"
    )
    print(
        f"  - Signal Bonus Weight: {config['training']['environment']['reward_settings']['signal_guidance']['signal_bonus_weight']}"
    )
    print(
        f"  - Signal Penalty Weight: {config['training']['environment']['reward_settings']['signal_guidance']['signal_penalty_weight']}"
    )
    print(
        f"  - Curriculum Stage: {config['training']['environment']['curriculum_stage']}"
    )
    print(f"  - Ent Coef: {config['training']['sac_hyperparameters']['ent_coef']}")

    print("\n🚀 Starting training...")
    trainer = create_algorithm_trainer("sac", config)
    trainer.train()

    print("\n✅ Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
