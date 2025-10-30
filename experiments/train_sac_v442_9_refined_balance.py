#!/usr/bin/env python3
"""
SAC v442.9 Refined Balance Trading Training

SELLバイアスを修正したバランスメカニズム:
- consistency_penaltyを0.04に低減 (v442.8: 0.08)
- entropy_regularizationを0.01に低減 (v442.8: 0.02)
- action_balance_targetを0.6に調整 (v442.8: 0.8)
- stability_optimizedステージを維持
- よりバランスの取れたアクション分布を目指す
"""

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAC v442.9 Refined Balance Trading Training"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps from config",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v442_9_refined_balance_config.json",
        help="Path to configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🚀 SAC v442.9 Refined Balance Trading Training")
    print("=" * 60)
    print("SELLバイアスを修正したバランスメカニズム:")
    print("- consistency_penalty: 0.04 (v442.8: 0.08)")
    print("- entropy_regularization: 0.01 (v442.8: 0.02)")
    print("- action_balance_target: 0.6 (v442.8: 0.8)")
    print("- stability_optimizedカリキュラムステージ維持")
    print("- よりバランスの取れたアクション分布を目指す")
    print("=" * 60)

    try:
        # 設定ファイル読み込み
        config_path = project_root / "config" / "sac_v442_9_refined_balance_config.json"
        print(f"🔍 Loading config from: {config_path.resolve()}")

        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return 1

        # Unified Trainerで学習実行
        trainer = V4XXUnifiedTrainer(str(config_path))

        # タイムステップの上書き
        if args.timesteps:
            trainer.config["training"]["total_timesteps"] = args.timesteps
            print(f"📊 Overriding timesteps to: {args.timesteps:,}")

        print(f"🎯 Starting training for {trainer.config['model_name']}")
        print(f"📈 Total timesteps: {trainer.config['training']['total_timesteps']:,}")

        # トレーニング実行
        trainer.train()

        print("✅ Training completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
