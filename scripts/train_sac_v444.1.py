#!/usr/bin/env python3
"""
SAC v444.1 Training Script
10エピソードで学習を行う
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.training.unified_trainer import UnifiedTrainer
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """SAC v444.1トレーニング実行"""
    config_path = "configs/sac_v444.1_config.json"

    # 設定ファイルの読み込み
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info("Starting SAC v444.1 training with 10 episodes...")

    # UnifiedTrainerの初期化
    trainer = UnifiedTrainer(config=config)

    # トレーニング実行
    try:
        stats = trainer.train()
        logger.info(f"Training completed. Stats: {stats}")

        # モデル保存
        trainer.save_model("models/sac_v444.1_final.zip")
        logger.info("Model saved to models/sac_v444.1_final.zip")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()