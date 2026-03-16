#!/usr/bin/env python3
"""
簡素化されたv456訓練スクリプト
マルチプロセッシング関連の複雑性を排除
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# プロジェクトパス
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTrainingCallback(BaseCallback):
    """シンプルなコールバック"""
    
    def __init__(self, log_freq: int = 5000):
        super().__init__()
        self.log_freq = log_freq
    
    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.log_freq == 0:
            logger.info(f"Milestone {self.model.num_timesteps:,} timesteps")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--csv-path', type=str, default='data/btc_jpy_1m_v451.csv')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("v456 SIMPLE Training Pipeline")
    logger.info("="*70)
    
    # データ読み込み
    logger.info(f"📥 Loading data from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    logger.info(f"✓ Loaded {len(df):,} bars")
    df = calculate_base_features(df, copy=False)
    
    # 環境作成
    logger.info("Creating training environment...")
    try:
        env = create_fast_intraday_env_v456(
            df=df,
            env_config={
                "reward_settings": {
                    "alpha": 0.0,
                    "beta": 0.0,
                    "gamma": 0.0,
                    "edge_penalty_rate": 0.0,
                    "vol_floor_penalty": 0.0,
                    "hold_ramp": 0.0,
                }
            },
        )
        if env is None:
            raise RuntimeError("Failed to create environment")
        del df
        
        logger.info(f"✓ Environment created: obs_shape={env.observation_space.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to create environment: {e}", exc_info=True)
        return 1
    
    # SAC モデル作成
    logger.info("Creating SAC model...")
    try:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            verbose=0,
            device="cpu",
        )
        logger.info("✓ SAC model created")
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}", exc_info=True)
        return 1
    
    # 訓練実行
    logger.info(f"\n🚀 Starting training: {args.timesteps:,} timesteps")
    start_time = datetime.utcnow()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=SimpleTrainingCallback(log_freq=10000),
            progress_bar=True,
        )
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1
    finally:
        env.close()
    
    # モデル保存
    timestamp = int(pd.Timestamp.now().timestamp())
    output_dir = Path("models/v456/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"v456_simplified_{timestamp}"
    
    try:
        model.save(str(model_path))
        logger.info(f"✓ Model saved: {model_path}")
        logger.info(f"Timesteps: {args.timesteps:,}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}", exc_info=True)
        return 1
    
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
