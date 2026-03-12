#!/usr/bin/env python3
"""
SAC v442.4 Balanced Trading Training

売買アクションのバランスを重視した学習:
- シングルタイムフレームのみ
- 基本的なテクニカル指標のみを使用
- Long/Short報酬をさらに調整して売買バランスを改善 (long: 1.3, short: 1.2)
- 安定性重視の報酬関数
- Optimizer features統合
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
    parser = argparse.ArgumentParser(description="SAC v442.3 Balanced Trading Training")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps from config",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v442_4_balanced_config.json",
        help="Path to configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🚀 SAC v442.1 Single Timeframe Improved Training")
    print("=" * 60)
    print("v441反省点の反映:")
    print("- シングルタイムフレームのみ")
    print("- 基本テクニカル指標のみ（10個）")
    print("- v441のLong/Short報酬設定を統合")
    print("- 安定性重視の報酬関数")
    print("- 保守的学習パラメータ")
    print("- Optimizer features統合")
    print("=" * 60)

    try:
        # 設定ファイル読み込み
        config_path = Path(args.config)
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
