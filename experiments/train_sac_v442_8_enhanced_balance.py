#!/usr/bin/env python3
"""
SAC v442.8 Enhanced Balance Trading Training

安定性重視のバランスメカニズムを統合:
- stability_optimizedステージを使用
- v441のbehavior_optimizationパラメータを統合
- エントロピー正則化と一貫性ペナルティを適用
- アクション分布の安定化を目指す
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
        description="SAC v442.8 Enhanced Balance Trading Training"
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
        default="config/sac_v442_8_enhanced_balance_config.json",
        help="Path to configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🚀 SAC v442.8 Enhanced Balance Trading Training")
    print("=" * 60)
    print("v441の高度なバランスメカニズムを統合:")
    print("- stability_optimizedカリキュラムステージ")
    print("- エントロピー正則化 (entropy_regularization)")
    print("- 一貫性ペナルティ (consistency_penalty)")
    print("- アクション平滑化 (action_smoothing)")
    print("- 強化されたバランス目標 (action_balance_target)")
    print("- 報酬乗数の最適化バランス")
    print("=" * 60)

    try:
        # 設定ファイル読み込み
        config_path = (
            project_root / "config" / "sac_v442_8_enhanced_balance_config.json"
        )
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
