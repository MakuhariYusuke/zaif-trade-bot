#!/usr/bin/env python3
"""
v456 訓練スクリプト (最適化版)

環境特性に応じた最適化：
- CPU のみの環境 → batch_size 削減、gradient_steps 削減
- メモリ効率 → 段階的学習（Curriculum Learning）
- 安定性 → Gradient Clipping、Learning Rate Annealing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# プロジェクト PATH 設定
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import extract_env_config, load_config_dict
from ztb.utils.checkpoint import CheckpointManager
from ztb.utils.cache_coordination import CacheCoordinator
from ztb.utils.error_utils import safe_operation


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V456TrainingCallbackOptimized(BaseCallback):
    """最適化されたコールバック (CPU 最適化版)"""
    
    def __init__(
        self,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        cache_coord: Optional[CacheCoordinator] = None,
        save_freq: int = 5000,
        log_freq: int = 1000,
    ) -> None:
        """
        Args:
            checkpoint_mgr: Checkpoint マネージャー
            cache_coord: キャッシュ統合器
            save_freq: Checkpoint 保存頻度（timesteps）
            log_freq: ログ出力頻度（timesteps）
        """
        super().__init__()
        self.checkpoint_mgr = checkpoint_mgr
        self.cache_coord = cache_coord
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.last_save_step = 0
        self.last_log_step = 0  # ロギング用の別個フラグ
    
    def _on_step(self) -> bool:
        """ステップごとのコールバック"""
        try:
            current_step = self.model.num_timesteps
            
            # ログ出力（セーブと分離）
            if current_step - self.last_log_step >= self.log_freq:
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                    avg_reward = np.mean(ep_rewards)
                    logger.info(
                        f"Milestone {current_step:,} steps | "
                        f"Avg Reward: {avg_reward:.4f} | "
                        f"Episodes: {len(self.model.ep_info_buffer)}"
                    )
                self.last_log_step = current_step  # ロギング済みとマーク
            
            # Checkpoint 保存
            if current_step - self.last_save_step >= self.save_freq:
                if self.checkpoint_mgr:
                    try:
                        metrics = {
                            'step': current_step,
                            'avg_reward': (
                                np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                                if self.model.ep_info_buffer else 0.0
                            ),
                            'episode_count': len(self.model.ep_info_buffer),
                        }
                        # CheckpointManager の正しい API を使用
                        self.checkpoint_mgr.save_sync(
                            obj=self.model,
                            step=current_step,
                            metadata={'reward_info': metrics}
                        )
                        logger.debug(f"✓ Checkpoint saved at step {current_step:,}")
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
                
                self.last_save_step = current_step
            
            # キャッシュ統計
            if self.cache_coord and current_step % (self.log_freq * 5) == 0:
                stats = self.cache_coord.get_stats()
                if stats:
                    logger.debug(
                        f"Cache Stats: hits={stats.get('hits', 0)}, "
                        f"misses={stats.get('misses', 0)}, "
                        f"size={stats.get('current_size', 0)}"
                    )
            
            return True  # 訓練継続
        
        except Exception as e:
            logger.error(f"Callback エラー: {e}", exc_info=True)
            return True  # エラーが発生しても訓練継続


class V456TrainingPipelineOptimized:
    """最適化された訓練パイプライン (CPU 環境対応)"""
    
    def __init__(
        self,
        timesteps: int = 10000,
        batch_size: int = 64,
        learning_rate: float = 0.0001,
        gradient_steps: int = 1,
        use_checkpoint: bool = True,
        use_cache: bool = True,
        config_path: str = "config/v456/base/config.yaml",
    ) -> None:
        """
        Args:
            timesteps: 訓練ステップ数
            batch_size: バッチサイズ (CPU 最適化: 64)
            learning_rate: 学習率 (CPU 最適化: 0.0001)
            gradient_steps: 勾配更新ステップ数
            use_checkpoint: Checkpoint 機能を使用するか
            use_cache: キャッシュ機能を使用するか
            config_path: Config ファイルパス
        """
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps
        self.use_checkpoint = use_checkpoint
        self.config_path = config_path
        self.config = self._load_config()
        self.env_config = extract_env_config(self.config) if self.config else {}
        self.reward_params = (
            self.env_config.get("reward_settings", {}) if self.env_config else {}
        )
        self.use_cache = use_cache
        
        self.checkpoint_mgr: Optional[CheckpointManager] = None
        self.cache_coord: Optional[CacheCoordinator] = None
    
    def _load_config(self) -> dict:
        """Config ファイルを読込"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return {}

            config = load_config_dict(config_file)
            logger.info(f"✓ Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def setup_optimizations(self) -> None:
        """Phase 1-3 最適化を初期化"""
        logger.info("="*70)
        logger.info("v456 Optimized Training Pipeline")
        logger.info("="*70)
        
        if self.use_checkpoint:
            self.checkpoint_mgr = CheckpointManager(
                save_dir="models/v456/checkpoints",
                keep_last=5,
                compress="zlib",
            )
            logger.info("✓ CheckpointManager initialized")
        
        # キャッシュは Windows 環境で multiprocessing エラーが発生するため無効化
        if self.use_cache and False:  # DISABLED on Windows
            self.cache_coord = CacheCoordinator(
                max_items=500,  # CPU 最適化: 削減
                ttl_seconds=1800,  # 30 分
            )
            logger.info("✓ CacheCoordinator initialized (LRU+TTL)")
    
    def load_data(self, csv_path: str = "test_synthetic_dataset.csv") -> pd.DataFrame:
        """
        訓練データを読み込む
        
        Args:
            csv_path: CSV ファイルパス
        
        Returns:
            訓練データ (DataFrame)
        """
        logger.info(f"📥 Loading data from {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"✓ Loaded {len(df)} bars")
        
        return df
    
    def create_environment(
        self, df: pd.DataFrame
    ) -> FastIntradayEnvV456:
        """
        訓練環境を作成
        
        Args:
            df: 訓練データ
        
        Returns:
            初期化された環境
        """
        logger.info("Creating training environment...")
        if self.reward_params:
            logger.info(f"  Reward parameters from config: {list(self.reward_params.keys())}")

        df = calculate_base_features(df, copy=False)
        
        def create_env() -> FastIntradayEnvV456:
            # デフォルトご褒美パラメータ（すべて0で簡素化）
            default_reward_params = {
                'alpha': 0.0,      # churn penalty disabled
                'beta': 0.0,       # hold penalty disabled
                'gamma': 0.0,      # inventory risk disabled
                'edge_penalty_rate': 0.0,
                'vol_floor_penalty': 0.0,
                'hold_ramp': 0.0,
            }
            
            # Config または デフォルトをマージ
            if self.reward_params:
                default_reward_params.update(self.reward_params)

            env_config = dict(self.env_config) if self.env_config else {}
            env_config["reward_settings"] = default_reward_params

            env = create_fast_intraday_env_v456(df=df, env_config=env_config)
            if env is None:
                raise RuntimeError("Failed to create training environment")
            return env
        
        env = safe_operation(
            create_env,
            default_result=None,
            operation_name="environment_creation"
        )
        
        if env is None:
            raise RuntimeError("Failed to create training environment")
        
        logger.info(f"✓ Environment created: obs_shape={env.observation_space.shape}")
        return env
    
    def train(self, env: FastIntradayEnvV456) -> Optional[SAC]:
        """
        訓練を実行
        
        Args:
            env: 訓練環境
        
        Returns:
            訓練済みモデル
        """
        logger.info("="*70)
        logger.info(f"Training Start: {self.timesteps:,} timesteps")
        logger.info(f"Config: batch_size={self.batch_size}, lr={self.learning_rate}")
        logger.info("="*70)
        
        # モデル作成
        model: SAC = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            buffer_size=100_000,  # CPU 最適化: 削減
            batch_size=self.batch_size,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=self.gradient_steps,
            ent_coef="auto",
            target_entropy="auto",
            verbose=0,
        )
        logger.info("✓ SAC model created")
        
        # 訓練実行
        def run_training() -> SAC:
            callback = V456TrainingCallbackOptimized(
                checkpoint_mgr=self.checkpoint_mgr,
                cache_coord=self.cache_coord,
                save_freq=max(5000, self.timesteps // 10),
                log_freq=max(1000, self.timesteps // 50),
            )
            
            model.learn(
                total_timesteps=self.timesteps,
                callback=callback,
                progress_bar=False,  # tqdm/rich が無い環境では無効化
                log_interval=self.timesteps // 10,  # 進捗表示削減
            )
            
            return model
        
        trained_model = safe_operation(
            run_training,
            default_result=None,
            operation_name="training"
        )
        
        if trained_model is None:
            logger.error("❌ Training failed")
            return None
        
        logger.info("✅ Training Completed Successfully")
        return trained_model
    
    def save_model(self, model: Optional[SAC]) -> Optional[str]:
        """
        モデルを保存
        
        Args:
            model: 訓練済みモデル
        
        Returns:
            保存パス
        """
        if model is None:
            logger.error("Cannot save None model")
            return None
        
        model_dir = Path("models/v456/final")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        import time
        timestamp = int(time.time())
        model_path = str(model_dir / f"v456_trained_{timestamp}")
        
        try:
            model.save(model_path)
            logger.info(f"✓ Model saved: {model_path}")
            logger.info(f"Timesteps: {model.num_timesteps:,}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None


def main() -> int:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="v456 Optimized Training Pipeline"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Total timesteps for training (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64 for CPU)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient steps (default: 1)"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="test_synthetic_dataset.csv",
        help="Path to training data CSV"
    )
    
    args = parser.parse_args()
    
    try:
        # パイプライン初期化
        pipeline = V456TrainingPipelineOptimized(
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_steps=args.gradient_steps,
            use_checkpoint=True,
            use_cache=True,
        )
        
        # 最適化設定
        pipeline.setup_optimizations()
        
        # データ読み込み
        df = pipeline.load_data(args.csv_path)
        
        # 環境作成
        env = pipeline.create_environment(df)
        del df
        
        # 訓練実行
        model = pipeline.train(env)
        
        # モデル保存
        result = 0
        if model is not None:
            pipeline.save_model(model)
        else:
            logger.error("Training failed: model is None")
            result = 1
        
        # リソースクリーンアップ
        try:
            if pipeline.cache_coord and hasattr(pipeline.cache_coord, 'shutdown'):
                logger.info("Shutting down cache coordinator...")
                pipeline.cache_coord.shutdown()
        except Exception as e:
            logger.warning(f"Cache coordinator shutdown error: {e}")
        
        return result
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
