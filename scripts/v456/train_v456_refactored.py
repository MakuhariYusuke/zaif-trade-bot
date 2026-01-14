#!/usr/bin/env python3
"""
v456 訓練スクリプト（リファクタリング版）

型安全な環境初期化ファクトリー + Phase 1-3 最適化統合
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import argparse
import time
import json

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from ztb.trading.environment.factory_v456 import EnvironmentFactory
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.utils.error_utils import safe_operation
from ztb.utils.checkpoint import CheckpointManager
from ztb.utils.cache_coordination import CacheCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class V456TrainingCallback(BaseCallback):
    """訓練コールバック（Phase 1-3 統合）"""
    
    def __init__(
        self,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        cache_coord: Optional[CacheCoordinator] = None,
    ) -> None:
        super().__init__()
        self.checkpoint_mgr = checkpoint_mgr
        self.cache_coord = cache_coord
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int = 0
        self.milestone_steps: list[int] = [1000, 5000, 10000, 25000, 50000, 100000]
        self.next_milestone_idx: int = 0
        self.start_time: float = time.time()
    
    def _on_step(self) -> bool:
        """訓練ステップごとのコールバック"""
        reward: float = float(self.locals.get("rewards", [0.0])[0])
        done: bool = bool(self.locals.get("dones", [False])[0])
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Milestone 処理
        while (self.next_milestone_idx < len(self.milestone_steps) and
               self.num_timesteps >= self.milestone_steps[self.next_milestone_idx]):
            milestone: int = self.milestone_steps[self.next_milestone_idx]
            elapsed: float = time.time() - self.start_time
            
            avg_reward: float = (
                float(np.mean(self.episode_rewards[-100:]))
                if self.episode_rewards else 0.0
            )
            
            logger.info(
                f"⏱️  Milestone {milestone:,} steps | "
                f"Avg Reward (last 100): {avg_reward:.4f} | "
                f"Episodes: {len(self.episode_rewards)} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            
            # Phase 1-A: Checkpoint 保存
            if self.checkpoint_mgr:
                self._save_checkpoint(milestone, avg_reward)
            
            # Phase 3: キャッシュ統計
            if self.cache_coord:
                self._log_cache_stats()
            
            self.next_milestone_idx += 1
        
        return True
    
    def _save_checkpoint(self, milestone: int, avg_reward: float) -> None:
        """Checkpoint 保存"""
        if not self.checkpoint_mgr:
            return
        
        try:
            checkpoint_path = self.checkpoint_mgr.save_sync(
                self.model,
                step=milestone,
                metadata={
                    "avg_reward": avg_reward,
                    "total_timesteps": self.num_timesteps,
                    "episodes": len(self.episode_rewards),
                }
            )
            logger.info(f"  ✓ Checkpoint saved: {Path(checkpoint_path).name}")
        except Exception as e:
            logger.error(f"  ❌ Checkpoint save failed: {e}")
    
    def _log_cache_stats(self) -> None:
        """キャッシュ統計をログ"""
        if not self.cache_coord:
            return
        
        stats: Dict[str, float] = self.cache_coord.get_stats()
        logger.info(
            f"  📊 Cache: hit_rate={stats['hit_rate']:.1%}, "
            f"items={stats['items']}/{stats['max_items']}, "
            f"size={stats['size_mb']:.2f}MB"
        )


class V456TrainingPipeline:
    """v456 訓練パイプライン（型安全）"""
    
    def __init__(
        self,
        timesteps: int = 50000,
        batch_size: int = 256,
        learning_rate: float = 0.0003,
        use_checkpoint: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_checkpoint = use_checkpoint
        self.use_cache = use_cache
        
        # Phase 1-3 最適化
        self.checkpoint_mgr: Optional[CheckpointManager] = None
        self.cache_coord: Optional[CacheCoordinator] = None
    
    def setup_optimizations(self) -> None:
        """Phase 1-3 最適化をセットアップ"""
        if self.use_checkpoint:
            checkpoint_dir = Path("models/v456/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_mgr = CheckpointManager(
                save_dir=str(checkpoint_dir),
                compress="zstd",
                keep_last=5,
            )
            logger.info(f"✓ CheckpointManager initialized (zstd)")
        
        if self.use_cache:
            self.cache_coord = CacheCoordinator(
                max_items=1000,
                ttl_seconds=3600,
            )
            logger.info(f"✓ CacheCoordinator initialized (LRU+TTL)")
    
    def load_data(self, data_path: Path) -> Optional[pd.DataFrame]:
        """データを読み込む"""
        logger.info(f"📥 Loading data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✓ Loaded {len(df)} bars")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def create_environment(self, df: pd.DataFrame) -> Optional[FastIntradayEnvV456]:
        """環境を作成（型安全ファクトリー使用）"""
        logger.info("Creating training environment...")
        
        def factory_create() -> Optional[FastIntradayEnvV456]:
            factory = EnvironmentFactory(df)
            return factory.create_training_env()
        
        env = safe_operation(
            factory_create,
            default_result=None,
            operation_name="environment_creation"
        )
        
        return env
    
    def train(self, env: FastIntradayEnvV456) -> Optional[SAC]:
        """訓練を実行"""
        logger.info("\n" + "=" * 70)
        logger.info(f"Training Start: {self.timesteps:,} timesteps")
        logger.info("=" * 70 + "\n")
        
        # SAC モデル作成
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            gamma=0.99,
            tau=0.005,
            batch_size=self.batch_size,
            buffer_size=1_000_000,
            verbose=1,
        )
        logger.info(f"✓ SAC model created")
        
        # 訓練実行
        def run_training() -> SAC:
            callback = V456TrainingCallback(
                checkpoint_mgr=self.checkpoint_mgr,
                cache_coord=self.cache_coord,
            )
            
            model.learn(
                total_timesteps=self.timesteps,
                callback=callback,
                progress_bar=True,
            )
            
            return model
        
        trained_model = safe_operation(
            run_training,
            default_result=None,
            operation_name="training"
        )
        
        return trained_model
    
    def save_model(self, model: SAC) -> Optional[Path]:
        """モデルを保存"""
        final_dir = Path("models/v456/final")
        final_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        final_path = final_dir / f"v456_trained_{timestamp}"
        
        try:
            model.save(str(final_path))
            logger.info(f"✓ Model saved: {final_path}")
            return final_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None


def main() -> int:
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="v456 訓練スクリプト (リファクタリング版)"
    )
    parser.add_argument("--timesteps", type=int, default=50000, help="訓練ステップ数")
    parser.add_argument("--batch-size", type=int, default=256, help="バッチサイズ")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="学習率")
    parser.add_argument(
        "--data",
        type=str,
        default="test_synthetic_dataset.csv",
        help="データファイルパス"
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("v456 Training Pipeline (Refactored)")
    logger.info("=" * 70)
    
    # パイプライン作成
    pipeline = V456TrainingPipeline(
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_checkpoint=True,
        use_cache=True,
    )
    
    # 最適化セットアップ
    pipeline.setup_optimizations()
    
    # データロード
    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        # フォールバック
        fallback_paths = [
            Path("test_synthetic_dataset.csv"),
            Path("data/datasets/test_synthetic_dataset.csv"),
        ]
        for fallback in fallback_paths:
            if fallback.exists():
                data_path = fallback
                logger.info(f"Using fallback: {data_path}")
                break
    
    df = pipeline.load_data(data_path)
    if df is None:
        logger.error("Failed to load data")
        return 1
    
    # 環境作成
    env = pipeline.create_environment(df)
    if env is None:
        logger.error("Failed to create environment")
        return 1
    
    # 訓練実行
    model = pipeline.train(env)
    if model is None:
        logger.error("Training failed")
        return 1
    
    # モデル保存
    model_path = pipeline.save_model(model)
    if model_path is None:
        logger.error("Failed to save model")
        return 1
    
    # 結果サマリー
    logger.info("\n" + "=" * 70)
    logger.info("✅ Training Completed Successfully")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Batch Size: {args.batch_size}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
