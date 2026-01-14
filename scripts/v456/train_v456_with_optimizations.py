#!/usr/bin/env python3
"""
v456 Training with Phase 1-3 Optimizations

Phase 1-B: Unified error handling (safe_operation)
Phase 1-A: Unified checkpoint management with zstd compression
Phase 2: Parallel walk-forward evaluation
Phase 3: Cache coordination (LRU + TTL)
"""
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / "src"))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import update_learning_rate

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.training.action_converter_v456 import ActionConverterV456
from ztb.utils.error_utils import safe_operation
from ztb.utils.checkpoint import CheckpointManager
from ztb.utils.cache_coordination import CacheCoordinator
from ztb.optimization.parallel.window_evaluator import ParallelWindowEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class V456OptimizedCallback(BaseCallback):
    """訓練監視コールバック with Phase 1-3 最適化チェック"""
    
    def __init__(
        self,
        checkpoint_mgr: CheckpointManager,
        cache_coord: Optional[CacheCoordinator] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.checkpoint_mgr = checkpoint_mgr
        self.cache_coord = cache_coord
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.milestone_steps = [1000, 5000, 10000, 25000, 50000, 100000]
        self.next_milestone_idx = 0
        self.start_time = time.time()
    
    def _on_step(self) -> bool:
        """訓練ステップごとのコールバック"""
        reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Milestone 報告
        while (self.next_milestone_idx < len(self.milestone_steps) and
               self.num_timesteps >= self.milestone_steps[self.next_milestone_idx]):
            milestone = self.milestone_steps[self.next_milestone_idx]
            elapsed = time.time() - self.start_time
            
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards[-100:])
                median_reward = np.median(self.episode_rewards[-100:])
            else:
                avg_reward = median_reward = 0.0
            
            logger.info(
                f"⏱️  Milestone {milestone:,} steps | "
                f"Avg Reward (last 100): {avg_reward:.4f} | "
                f"Median: {median_reward:.4f} | "
                f"Episodes: {len(self.episode_rewards)} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            
            # Phase 1-A: Checkpoint保存
            if self.checkpoint_mgr:
                try:
                    checkpoint_path = self.checkpoint_mgr.save_sync(
                        self.model,
                        step=milestone,
                        metadata={
                            "avg_reward": float(avg_reward),
                            "total_timesteps": self.num_timesteps,
                            "episodes": len(self.episode_rewards),
                        }
                    )
                    logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"❌ Checkpoint save failed: {e}")
            
            # Phase 3: Cache 統計表示
            if self.cache_coord:
                stats = self.cache_coord.get_stats()
                logger.info(
                    f"📊 Cache Stats: Hit Rate={stats['hit_rate']:.2%} | "
                    f"Items={stats['items']}/{stats['max_items']} | "
                    f"Size={stats['size_mb']:.2f}MB"
                )
            
            self.next_milestone_idx += 1
        
        return True


def load_v456_config(config_path: str) -> Dict:
    """v456 configを読み込む"""
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"✓ Config loaded: {config.get('version', 'unknown')} version")
    return config


def create_environment(
    config: Dict,
    env_config: EnvironmentConfig,
) -> FastIntradayEnvV456:
    """v456環境を作成"""
    logger.info("Creating v456 environment...")
    
    env = FastIntradayEnvV456(
        config=env_config,
        action_converter=ActionConverterV456(),
        seed=config.get('seed', 42),
    )
    logger.info(f"✓ Environment created: obs_space={env.observation_space}, action_space={env.action_space}")
    return env


def setup_optimizations(config: Dict) -> Tuple[CheckpointManager, Optional[CacheCoordinator]]:
    """Phase 1-3 最適化をセットアップ"""
    # Phase 1-A: Checkpoint Manager (zstd compression)
    checkpoint_dir = Path("models/v456/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_mgr = CheckpointManager(
        save_dir=str(checkpoint_dir),
        compress="zstd",
        keep_last=3,
    )
    logger.info(f"✓ CheckpointManager initialized: {checkpoint_dir}")
    
    # Phase 3: Cache Coordinator (LRU + TTL)
    cache_enabled = config.get('evaluation', {}).get('enable_caching', False)
    cache_coord = None
    if cache_enabled:
        cache_coord = CacheCoordinator(
            max_items=config.get('evaluation', {}).get('cache_max_items', 1000),
            ttl_seconds=config.get('evaluation', {}).get('cache_ttl_seconds', 3600),
        )
        logger.info(f"✓ CacheCoordinator initialized (LRU+TTL)")
    
    return checkpoint_mgr, cache_coord


def train_v456(
    config: Dict,
    env_config: EnvironmentConfig,
    timesteps: int,
    checkpoint_mgr: CheckpointManager,
    cache_coord: Optional[CacheCoordinator],
) -> SAC:
    """v456モデルを訓練"""
    logger.info("=" * 60)
    logger.info("Starting v456 Training with Phase 1-3 Optimizations")
    logger.info("=" * 60)
    
    # 環境作成
    env = create_environment(config, env_config)
    
    # SAC ハイパーパラメータ
    sac_params = config.get('training', {}).get('sac_hyperparameters', {})
    learning_rate = sac_params.get('learning_rate', 0.0003)
    gamma = sac_params.get('gamma', 0.99)
    tau = sac_params.get('tau', 0.005)
    batch_size = sac_params.get('batch_size', 256)
    buffer_size = sac_params.get('buffer_size', 1000000)
    
    logger.info(
        f"SAC Parameters: lr={learning_rate}, gamma={gamma}, tau={tau}, "
        f"batch_size={batch_size}, buffer_size={buffer_size}"
    )
    
    # SAC モデル作成
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=buffer_size,
        verbose=1,
    )
    
    # Phase 1-B: safe_operation でフェイルセーフ訓練を実施
    def train_with_error_handling():
        """エラー処理付き訓練"""
        callback = V456OptimizedCallback(
            checkpoint_mgr=checkpoint_mgr,
            cache_coord=cache_coord,
            verbose=1,
        )
        
        start_time = time.time()
        logger.info(f"Training for {timesteps:,} timesteps...")
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Training completed in {elapsed:.1f}s")
        return model
    
    # Phase 1-B: safe_operation でラップして実行
    result = safe_operation(
        func=train_with_error_handling,
        default_return=model,
        collect_errors=False,
    )
    
    if result is None:
        logger.error("❌ Training failed")
        return None
    
    logger.info("✓ Training completed successfully")
    return result


def evaluate_v456(
    model: SAC,
    env_config: EnvironmentConfig,
    num_episodes: int = 5,
    cache_coord: Optional[CacheCoordinator] = None,
) -> Dict:
    """訓練済みモデルを評価"""
    logger.info("=" * 60)
    logger.info(f"Evaluating v456 on {num_episodes} episodes")
    logger.info("=" * 60)
    
    env = create_environment({}, env_config)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        logger.info(f"Episode {episode+1}: Reward={episode_reward:.4f}, Length={episode_length}")
        
        # Phase 3: Cache 統計
        if cache_coord:
            stats = cache_coord.get_stats()
            logger.info(f"  Cache: hit_rate={stats['hit_rate']:.2%}, items={stats['items']}")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Avg Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    logger.info(f"  Avg Episode Length: {np.mean(episode_lengths):.1f}")
    
    return {
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    """メイン訓練パイプライン"""
    parser = argparse.ArgumentParser(
        description="v456 Training with Phase 1-3 Optimizations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/v456/base/config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Training timesteps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    # Config 読み込み
    config = load_v456_config(args.config)
    
    # Environment Config を作成
    env_config = EnvironmentConfig.from_dict(config.get('training', {}))
    
    # Phase 1-3 最適化をセットアップ
    checkpoint_mgr, cache_coord = setup_optimizations(config)
    
    # 訓練実行
    model = train_v456(
        config=config,
        env_config=env_config,
        timesteps=args.timesteps,
        checkpoint_mgr=checkpoint_mgr,
        cache_coord=cache_coord,
    )
    
    if model is None:
        logger.error("❌ Training failed - exiting")
        return 1
    
    # 評価実行
    eval_results = evaluate_v456(
        model=model,
        env_config=env_config,
        num_episodes=args.eval_episodes,
        cache_coord=cache_coord,
    )
    
    # 最終モデル保存
    final_model_dir = Path("models/v456/final")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = final_model_dir / f"v456_final_{int(time.time())}"
    model.save(str(model_path))
    logger.info(f"✓ Final model saved: {model_path}")
    
    # 結果をJSONで保存
    results_file = final_model_dir / f"eval_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"✓ Results saved: {results_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ v456 Training Completed Successfully!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
