"""
SAC（Soft Actor-Critic）アルゴリズム実装。

SACは、エントロピー正則化を用いたoff-policyアクター・クリティック手法。
金融取引において、PPOよりも探索と活用のバランスが優れている可能性がある。

主な特徴：
- Off-policy: 過去の経験（Replay Buffer）を効率的に再利用
- エントロピー正則化: 自動的な探索・活用バランス調整
- alpha自動調整: target_entropyに基づく適応的なエントロピー係数
- ソフトアップデート: targetネットワークの安定的な更新

References:
    - Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
    - Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications"
"""

import logging
from typing import Any, Dict, Optional, Callable
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from ztb.training.algorithms.base_algorithm import BaseRLAlgorithm

logger = logging.getLogger(__name__)


# SACデフォルト設定（Stable-Baselines3準拠 + 金融取引向け調整）
DEFAULT_SAC_CONFIG = {
    # 学習率
    "learning_rate": 3e-4,
    
    # Replay Buffer
    "buffer_size": 50000,  # 金融データの季節性を考慮したサイズ
    
    # 訓練パラメータ
    "learning_starts": 1000,  # バッファが十分溜まってから学習開始
    "batch_size": 256,
    "tau": 0.005,  # ソフトアップデート係数（小さいほど安定）
    "gamma": 0.99,  # 割引率
    "train_freq": 1,  # 毎ステップ訓練
    "gradient_steps": 1,  # 各訓練で1回勾配更新
    
    # エントロピー正則化
    "ent_coef": "auto",  # 自動調整（推奨）
    "target_entropy": "auto",  # 自動設定（-dim(A)）
    
    # ネットワーク構造
    "policy_kwargs": None,  # デフォルトのMLP
    
    # その他
    "verbose": 1,
    "tensorboard_log": None,
    "device": "auto",
}


class SACAlgorithm(BaseRLAlgorithm):
    """
    SAC（Soft Actor-Critic）アルゴリズムのラッパークラス。
    
    BaseRLAlgorithmインターフェースに準拠し、
    AlgorithmFactoryから使用可能。
    
    金融取引への適用において、PPOと比較して以下の利点がある：
    - 過去の経験を効率的に再利用（sample efficiency）
    - エントロピー係数の自動調整による適応的な探索
    - 連続行動・離散行動両対応
    
    Example:
        >>> from ztb.training.algorithms import AlgorithmFactory
        >>> AlgorithmFactory.register("sac", SACAlgorithm)
        >>> sac = AlgorithmFactory.create("sac")
        >>> model = sac.create_model(env, config)
        >>> sac.train(model, total_timesteps=100000)
    """
    
    def __init__(self):
        """SACAlgorithmを初期化。"""
        self._model: Optional[BaseAlgorithm] = None
        logger.info("SACAlgorithm initialized")
    
    @property
    def algorithm_name(self) -> str:
        """
        アルゴリズム名を取得。
        
        Returns:
            "sac"
        """
        return "sac"
    
    def __repr__(self) -> str:
        """文字列表現。"""
        return f"SACAlgorithm(model={'initialized' if self._model else 'not_initialized'})"
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        SACのデフォルト設定を取得。
        
        Returns:
            デフォルト設定の辞書
            
        Example:
            >>> config = SACAlgorithm.get_default_config()
            >>> config["learning_rate"]
            0.0003
        """
        return DEFAULT_SAC_CONFIG.copy()
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        SAC設定の妥当性を検証。
        
        Args:
            config: 検証する設定辞書
            
        Returns:
            設定が妥当ならTrue
            
        Raises:
            ValueError: 必須パラメータが不足している場合
            
        Example:
            >>> config = {"learning_rate": 3e-4, "buffer_size": 50000}
            >>> SACAlgorithm.validate_config(config)
            True
        """
        # 必須パラメータ
        required_params = ["learning_rate", "buffer_size", "batch_size"]
        
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required SAC parameter: {param}")
        
        # 値の範囲チェック
        if config["learning_rate"] <= 0:
            raise ValueError(f"learning_rate must be positive, got {config['learning_rate']}")
        
        if config["buffer_size"] <= 0:
            raise ValueError(f"buffer_size must be positive, got {config['buffer_size']}")
        
        if config["batch_size"] <= 0:
            raise ValueError(f"batch_size must be positive, got {config['batch_size']}")
        
        # buffer_size >= batch_size
        if config["buffer_size"] < config["batch_size"]:
            raise ValueError(
                f"buffer_size ({config['buffer_size']}) must be >= batch_size ({config['batch_size']})"
            )
        
        logger.debug(f"SAC config validation passed: {config}")
        return True
    
    def create_model(
        self,
        env: VecEnv,
        config: Dict[str, Any],
        tensorboard_log: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        SACモデルを作成。
        
        Args:
            env: 訓練環境（VecEnv）
            config: SAC設定（ハイパーパラメータ等）
            tensorboard_log: TensorBoardログディレクトリ（Optional）
            
        Returns:
            作成されたSACモデル
            
        Raises:
            ValueError: 設定が不正な場合
            
        Example:
            >>> sac = SACAlgorithm()
            >>> config = sac.get_default_config()
            >>> model = sac.create_model(vec_env, config, "./logs")
        """
        # 設定検証
        self.validate_config(config)
        
        logger.info(f"Creating SAC model with config: {config}")
        
        # Stable-Baselines3 SAC用パラメータ抽出
        sac_params = {
            "policy": config.get("policy", "MlpPolicy"),
            "env": env,
            "learning_rate": config["learning_rate"],
            "buffer_size": config["buffer_size"],
            "learning_starts": config.get("learning_starts", DEFAULT_SAC_CONFIG["learning_starts"]),
            "batch_size": config["batch_size"],
            "tau": config.get("tau", DEFAULT_SAC_CONFIG["tau"]),
            "gamma": config.get("gamma", DEFAULT_SAC_CONFIG["gamma"]),
            "train_freq": config.get("train_freq", DEFAULT_SAC_CONFIG["train_freq"]),
            "gradient_steps": config.get("gradient_steps", DEFAULT_SAC_CONFIG["gradient_steps"]),
            "ent_coef": config.get("ent_coef", DEFAULT_SAC_CONFIG["ent_coef"]),
            "target_entropy": config.get("target_entropy", DEFAULT_SAC_CONFIG["target_entropy"]),
            "verbose": config.get("verbose", DEFAULT_SAC_CONFIG["verbose"]),
            "tensorboard_log": tensorboard_log,
            "device": config.get("device", DEFAULT_SAC_CONFIG["device"]),
        }
        
        # policy_kwargsがあれば追加
        if "policy_kwargs" in config and config["policy_kwargs"] is not None:
            sac_params["policy_kwargs"] = config["policy_kwargs"]
        
        # SACモデル作成
        self._model = SAC(**sac_params)
        
        logger.info(
            f"SAC model created: lr={config['learning_rate']}, "
            f"buffer_size={config['buffer_size']}, batch_size={config['batch_size']}, "
            f"ent_coef={config.get('ent_coef', 'auto')}"
        )
        
        return self._model
    
    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> BaseAlgorithm:
        """
        SACモデルを訓練。
        
        Args:
            model: 訓練するSACモデル
            total_timesteps: 訓練ステップ数
            callback: 訓練中のコールバック（Optional）
            **kwargs: その他のlearn()引数
            
        Returns:
            訓練済みモデル
            
        Raises:
            ValueError: モデルが初期化されていない場合
            
        Example:
            >>> sac = SACAlgorithm()
            >>> model = sac.create_model(env, config)
            >>> trained = sac.train(model, total_timesteps=100000)
        """
        if model is None:
            raise ValueError("Model must be initialized before training. Call create_model() first.")
        
        logger.info(f"Starting SAC training for {total_timesteps} timesteps")
        
        # 訓練実行
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )
        
        logger.info(f"SAC training completed: {total_timesteps} timesteps")
        
        return model
    
    def save(self, model: BaseAlgorithm, save_path: str) -> None:
        """
        モデルを保存。
        
        Args:
            model: 保存するモデル
            save_path: 保存先パス
            
        Example:
            >>> sac.save(model, "models/sac_v395a.zip")
        """
        model.save(save_path)
        logger.info(f"SAC model saved to {save_path}")
    
    @staticmethod
    def load(load_path: str, env: Optional[VecEnv] = None) -> BaseAlgorithm:
        """
        モデルを読み込み。
        
        Args:
            load_path: モデルファイルパス
            env: 環境（Optional、推論時に必要）
            
        Returns:
            読み込まれたモデル
            
        Example:
            >>> model = SACAlgorithm.load("models/sac_v395a.zip", env=vec_env)
        """
        model = SAC.load(load_path, env=env)
        logger.info(f"SAC model loaded from {load_path}")
        return model
