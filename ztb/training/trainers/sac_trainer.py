"""
SAC (Soft Actor-Critic) Algorithm Trainer.

SACアルゴリズム専用のトレーナー。
AlgorithmFactoryから生成されたSACAlgorithmを使用して訓練を実行する。
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from ztb.training.algorithms import AlgorithmFactory
from ztb.training.core.config_manager import ConfigManager
from ztb.trading.environment.environment import HeavyTradingEnv  # 🔧 Fixed import
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACAlgorithmTrainer:
    """
    SAC (Soft Actor-Critic) アルゴリズムのトレーナー。
    
    AlgorithmFactoryから生成されたSACAlgorithmを使用して、
    環境の作成、モデルの初期化、訓練、保存を実行する。
    """
    
    def __init__(self, config_manager: ConfigManager, progress_bar_enabled: bool = False):
        """
        SACAlgorithmTrainerを初期化。
        
        Args:
            config_manager: ConfigManager instance
            progress_bar_enabled: Whether progress bar is enabled
        """
        self.config_manager = config_manager
        self.progress_bar_enabled = progress_bar_enabled
        self.logger = get_logger(__name__)
    
    def train(self, unified_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        SAC訓練を実行。
        
        Args:
            unified_config: 統合設定辞書
            
        Returns:
            訓練結果（モデルパス、ログパス等）
        """
        self.logger.info("🚀 Starting SAC training")
        
        # 1. SAC設定を取得
        sac_config = unified_config.get("sac_hyperparameters", {})
        if not sac_config:
            self.logger.warning("No sac_hyperparameters found in config, using defaults")
            from ztb.training.algorithms.sac import SACAlgorithm
            sac_config = SACAlgorithm.get_default_config()
        
        # 2. 環境を作成
        env_config = unified_config.get("environment", {})
        
        # SAC requires continuous action space
        env_config["use_continuous_actions"] = True
        env_config["enable_action_masking"] = False  # SAC doesn't support action masking
        
        self.logger.info("🔧 Configured environment for SAC: continuous actions enabled")
        
        # ConfigManagerからデータを取得
        from ztb.utils.data_utils import load_csv_data_optimized
        data_path = unified_config.get("data_path", "btc_jpy_real_dataset.csv")
        
        df = load_csv_data_optimized(data_path)
        
        # HeavyTradingEnvを直接作成（env_configをconfigオブジェクトに変換）
        from ztb.trading.environment.utils.config import EnvironmentConfig
        
        # Convert dict to EnvironmentConfig
        config_obj = EnvironmentConfig.from_dict(env_config)
        env = HeavyTradingEnv(df=df, config=config_obj)
        vec_env = DummyVecEnv([lambda: env])
        self.logger.info(f"✅ Environment created with {len(df)} timesteps (continuous action space)")
        self.logger.info(f"   Action space: {env.action_space}")
        
        # 3. SACAlgorithmを作成
        sac_algo = AlgorithmFactory.create("sac")
        self.logger.info(f"✅ Algorithm created: {sac_algo}")
        
        # 4. モデルを作成
        model_name = unified_config.get("model_name", "sac_model")
        session_id = unified_config.get("session_id", "sac_session")
        log_dir = Path("checkpoints") / session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        model = sac_algo.create_model(
            env=vec_env,
            config=sac_config,
            tensorboard_log=str(log_dir)
        )
        self.logger.info(f"✅ Model created: {model}")
        
        # 5. コールバックを作成
        callbacks = []
        
        # チェックポイントコールバック
        checkpoint_interval = unified_config.get("checkpoint_interval", 10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=str(log_dir / "checkpoints"),
            name_prefix=model_name
        )
        callbacks.append(checkpoint_callback)
        
        callback_list = CallbackList(callbacks)
        
        # 6. 訓練実行
        total_timesteps = unified_config.get("total_timesteps", 100000)
        self.logger.info(f"🏃 Training for {total_timesteps} timesteps...")
        
        trained_model = sac_algo.train(
            model=model,
            total_timesteps=total_timesteps,
            callback=callback_list
        )
        
        # 7. モデル保存
        model_path = log_dir / f"{model_name}_final.zip"
        sac_algo.save(trained_model, str(model_path))
        self.logger.info(f"💾 Model saved to: {model_path}")
        
        # 8. 結果を返す
        result = {
            "model_path": str(model_path),
            "log_path": str(log_dir),
            "total_timesteps": total_timesteps,
            "algorithm": "sac",
            "success": True
        }
        
        self.logger.info("🎉 SAC training completed successfully!")
        return result
