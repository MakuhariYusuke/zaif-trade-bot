#!/usr/bin/env python3
"""
SAC v435.7c Training - Enhanced victory bonuses with symmetric thresholds
勝利ボーナス増大 + 対称閾値による値の焦げ付き防止
"""
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.utils.logging_utils import setup_logging


def main():
    # ログレベルをデバッグに設定
    setup_logging(level=logging.DEBUG)

    print("🚀 SAC v435.7c Training - Enhanced victory bonuses")
    print("=" * 60)

    # 設定ファイルのパス
    config_dir = Path("v435/v435.7")
    config_path = config_dir / "sac_v435_7c_config.json"
    env_config_path = config_dir / "sac_v435_environment_config.json"
    reward_config_path = config_dir / "sac_v435_7c_config.json"

    # メイン設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("📋 Configuration loaded:")
    print(f"  - Model: {config['model_name']}")
    print(f"  - Timesteps: {config['training']['total_timesteps']:,}")
    print(
        f"  - Transaction cost: {config['training']['environment']['transaction_cost']}"
    )
    print(
        f"  - Max position size: {config['training']['environment']['max_position_size']}"
    )
    print(
        f"  - Profit bonus ATR: {config['training']['reward_function']['base_profit_bonus_atr_coeff']}"
    )
    print(
        f"  - Profit bonus Portfolio: {config['training']['reward_function']['base_profit_bonus_portfolio_coeff']}"
    )
    print(
        f"  - Symmetric thresholds: {config['training']['environment']['symmetric_thresholds']}"
    )

    print("\n🚀 Starting training...")
    trainer = create_algorithm_trainer("sac", config)
    trainer.train()

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
