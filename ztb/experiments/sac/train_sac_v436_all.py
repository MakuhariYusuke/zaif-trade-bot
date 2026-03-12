#!/usr/bin/env python3
"""
SAC v436 All Variants Training - Train all three signal guidance variants
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.utils.logging_utils import setup_logging

def train_variant(config_path: str, variant_name: str) -> bool:
    """Train a single variant."""
    print(f"\n🚀 Training {variant_name}...")
    print("-" * 40)

    try:
        # 設定ファイル読み込み
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False

        with open(config_path_obj, "r", encoding="utf-8") as f:
            config = json.load(f)

        print("📋 Configuration loaded:")
        print(f"  - Model: {config['training']['model_name']}")
        print(f"  - Timesteps: {config['training']['total_timesteps']:,}")
        print(
            f"  - Signal Guidance: {config['training']['environment']['reward_settings']['signal_guidance']['enabled']}"
        )
        print(
            f"  - Guidance Mode: {config['training']['environment']['reward_settings']['signal_guidance']['guidance_mode']}"
        )

        # 訓練実行
        trainer = create_algorithm_trainer("sac", config)
        trainer.train()

        print(f"✅ {variant_name} training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ {variant_name} training failed: {e}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser(description="SAC v436 All Variants Training")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    print("🚀 SAC v436 All Variants Training")
    print("=" * 50)
    print("Training all three signal guidance variants:")
    print("1. full_guidance - Strong signal guidance throughout")
    print("2. no_guidance - Pure RL without signal guidance")
    print("3. fade_out - Signal guidance that fades over time")
    print()

    # 設定ファイルのパス
    configs = [
        ("config/sac_v436_signal_guided_config.json", "full_guidance"),
        ("config/sac_v436_no_guidance_config.json", "no_guidance"),
        ("config/sac_v436_fade_out_config.json", "fade_out"),
    ]

    results = []
    for config_path, variant_name in configs:
        success = train_variant(config_path, variant_name)
        results.append((variant_name, success))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("🎯 Training Summary:")
    for variant_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {variant_name}: {status}")

    successful_count = sum(1 for _, success in results if success)
    print(f"\n📊 {successful_count}/{len(results)} variants trained successfully")

    if successful_count == len(results):
        print("🎉 All variants completed successfully!")
        return 0
    else:
        print("⚠️  Some variants failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
