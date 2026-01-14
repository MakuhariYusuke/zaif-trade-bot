#!/usr/bin/env python3
"""
v456 実運用訓練スクリプト

Phase 1-3 最適化を統合し、実データで訓練実施
- Phase 1-B: 統一エラーハンドリング
- Phase 1-A: Checkpoint 管理（zstd 圧縮）
- Phase 2: 並列ウィンドウ評価
- Phase 3: キャッシュ統合（LRU+TTL）
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
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
sys.path.insert(0, str(workspace_root / "scripts" / "v456"))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.action_converter_v456 import ActionConverterV456
from ztb.utils.error_utils import safe_operation
from ztb.utils.checkpoint import CheckpointManager
from ztb.utils.cache_coordination import CacheCoordinator

# Local feature calculators
from feature_calculator_v456 import MTFFeatureCalculator, RegimeFeatureCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OptimizedTrainingCallback(BaseCallback):
    """訓練監視コールバック with Phase 1-3 統合"""
    
    def __init__(
        self,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        cache_coord: Optional[CacheCoordinator] = None,
    ):
        super().__init__()
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
                f"Avg Reward: {avg_reward:.4f} | "
                f"Episodes: {len(self.episode_rewards)} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            
            # Phase 1-A: Checkpoint 保存
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
                    logger.info(f"  ✓ Checkpoint saved")
                except Exception as e:
                    logger.error(f"  ❌ Checkpoint failed: {e}")
            
            # Phase 3: Cache 統計
            if self.cache_coord:
                stats = self.cache_coord.get_stats()
                logger.info(
                    f"  📊 Cache: hit_rate={stats['hit_rate']:.1%}, "
                    f"items={stats['items']}/{stats['max_items']}"
                )
            
            self.next_milestone_idx += 1
        
        return True


def load_and_split_data(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """データをロードして分割"""
    logger.info(f"📥 Loading data from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # タイムスタンプ処理
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    logger.info(f"✓ Loaded {len(df)} bars")
    
    # 70/15/15 分割
    n = len(df)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]
    
    logger.info(f"  Train: {len(train_df)} bars")
    logger.info(f"  Val: {len(val_df)} bars")
    logger.info(f"  Test: {len(test_df)} bars\n")
    
    return {"train": train_df, "val": val_df, "test": test_df}


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を計算"""
    df_copy = df.copy()
    
    try:
        # MTF 特徴量
        mtf_calc = MTFFeatureCalculator()
        mtf_features = mtf_calc.calculate_all_mtf_features(df_copy)
        for col in mtf_features.columns:
            if col not in df_copy.columns:
                df_copy[col] = mtf_features[col]
        
        # Regime 特徴量
        regime_features = RegimeFeatureCalculator.calculate_regime_features(df_copy)
        for col in regime_features.columns:
            if col not in df_copy.columns:
                df_copy[col] = regime_features[col]
        
        logger.info(f"✓ Calculated features: {df_copy.shape[1]} columns")
    except Exception as e:
        logger.warning(f"Feature calculation failed: {e}, using base data")
    
    return df_copy


def create_env(df: pd.DataFrame, env_config: EnvironmentConfig) -> FastIntradayEnvV456:
    """訓練環境を作成"""
    try:
        env = FastIntradayEnvV456(
            df=df,
            base_feature_columns=[col for col in df.columns if col.startswith("base_")],
            mtf_feature_columns=[col for col in df.columns if col.startswith("mtf_")],
            regime_feature_columns=[col for col in df.columns if col.startswith("regime_")],
            initial_balance=env_config.initial_balance if hasattr(env_config, 'initial_balance') else 1_000_000,
            max_position=env_config.max_position_size if hasattr(env_config, 'max_position_size') else 1.0,
        )
        logger.info(f"✓ Environment created: obs_shape={env.observation_space.shape}")
        return env
    except Exception as e:
        logger.error(f"Environment creation failed: {e}")
        raise


def main():
    """メイン訓練パイプライン"""
    parser = argparse.ArgumentParser(description="v456 訓練実行")
    parser.add_argument("--timesteps", type=int, default=50000, help="訓練ステップ数")
    parser.add_argument("--data", type=str, default="data/datasets/btc_jpy_real_dataset.csv", help="データファイル")
    parser.add_argument("--config", type=str, default="config/v456/base/config.yaml", help="Config ファイル")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="学習率")
    parser.add_argument("--batch-size", type=int, default=256, help="バッチサイズ")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("v456 訓練開始 (Phase 1-3 最適化統合)")
    logger.info("=" * 70)
    
    # データロード
    data_path = Path(args.data)
    try:
        data_dict = load_and_split_data(data_path)
    except FileNotFoundError:
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Using alternative data source...")
        # Fallback: use any available CSV
        alt_paths = list(Path("data/datasets").glob("*.csv"))
        if alt_paths:
            data_path = alt_paths[0]
            logger.info(f"Using: {data_path}")
            data_dict = load_and_split_data(data_path)
        else:
            logger.error("No data files found")
            return 1
    
    # Config ロード
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Config loaded: {config.get('version', 'unknown')}")
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = {}
    
    # 環境設定
    env_config = EnvironmentConfig()
    
    # Phase 1-3 最適化セットアップ
    checkpoint_dir = Path("models/v456/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_mgr = CheckpointManager(
        save_dir=str(checkpoint_dir),
        compress="zstd",
        keep_last=5,
    )
    logger.info(f"✓ CheckpointManager initialized (zstd)")
    
    cache_coord = CacheCoordinator(max_items=1000, ttl_seconds=3600)
    logger.info(f"✓ CacheCoordinator initialized (LRU+TTL)")
    
    # 訓練データで特徴量を計算
    train_df = calculate_features(data_dict["train"])
    
    # 環境作成
    def create_training_env():
        """訓練環境を作成"""
        return create_env(train_df, env_config)
    
    env = safe_operation(
        create_training_env,
        default_result=None,
        operation_name="env_creation"
    )
    
    if env is None:
        logger.error("❌ Failed to create environment")
        return 1
    
    # SAC モデル作成
    logger.info("\n" + "=" * 70)
    logger.info("SAC モデル初期化")
    logger.info("=" * 70)
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        gamma=0.99,
        tau=0.005,
        batch_size=args.batch_size,
        buffer_size=1_000_000,
        verbose=1,
    )
    logger.info(f"✓ SAC model created")
    
    # Phase 1-B: safe_operation でフェイルセーフ訓練実行
    logger.info("\n" + "=" * 70)
    logger.info(f"訓練開始: {args.timesteps:,} timesteps")
    logger.info("=" * 70 + "\n")
    
    def train_with_safety():
        """安全な訓練実行"""
        callback = OptimizedTrainingCallback(
            checkpoint_mgr=checkpoint_mgr,
            cache_coord=cache_coord,
        )
        
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )
        
        return model
    
    trained_model = safe_operation(
        train_with_safety,
        default_result=None,
        operation_name="training"
    )
    
    if trained_model is None:
        logger.error("❌ Training failed")
        return 1
    
    # 最終モデル保存
    final_dir = Path("models/v456/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"v456_trained_{int(time.time())}"
    trained_model.save(str(final_path))
    logger.info(f"\n✓ Final model saved: {final_path}")
    
    # 結果をレポート
    logger.info("\n" + "=" * 70)
    logger.info("✅ 訓練完了")
    logger.info("=" * 70)
    logger.info(f"Model: {final_path}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Batch Size: {args.batch_size}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
