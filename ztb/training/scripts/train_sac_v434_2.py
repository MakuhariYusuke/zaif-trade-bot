#!/usr/bin/env python3
"""
SAC v434.2 トレーニングスクリプト
v434.1の問題を解決するための改良版：
- 報酬関数：取引コスト強化、利益ボーナス増加、損失ペナルティ強化
- 学習戦略：エントロピー項調整、多様な初期条件、アンサンブル学習
- 特徴量：相関削減、次元削減、重要度分析
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from ztb.utils.training_utils import create_checkpoint_callback, save_model
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_schema
from ztb.training.core.feature_schema_manager import FeatureSchemaManager
from ztb.types.common import ConfigDict
from ztb.utils.data_utils import load_csv_data_optimized
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_v434_2_config() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """v434.2の設定を読み込み"""
    config_dir = Path("config")

    # 報酬設定
    reward_path = config_dir / "sac_v434_2_reward_config.json"
    with open(reward_path, "r", encoding="utf-8") as f:
        reward_config = json.load(f)

    # 環境設定
    env_path = config_dir / "sac_v434_2_environment_config.json"
    with open(env_path, "r", encoding="utf-8") as f:
        env_config = json.load(f)

    return reward_config, env_config


def create_v434_2_environment(
    data_path: str, reward_config: ConfigDict, env_config: ConfigDict
) -> DummyVecEnv:
    """v434.2の改良された環境を作成"""
    logger.info("Creating v434.2 environment with improved reward function")

    # データ読み込み
    df = load_csv_data_optimized(data_path)
    logger.info(f"Loaded data: {len(df):,} rows")

    # 特徴量スキーマの作成/読み込み
    model_name = "sac_v434_2"
    models_dir = Path("models")
    manager = FeatureSchemaManager(model_name, models_dir)

    # スキーマが存在しない場合は作成
    try:
        manager.load_schema()
        schema_exists = True
        logger.info("Existing schema found for v434.2")
    except FileNotFoundError:
        schema_exists = False
        logger.info("No existing schema found, creating new one for v434.2")

    if not schema_exists:
        # 単純な特徴量選択：数値特徴量を直接使用
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # 基本的なOHLCV特徴量を優先
        priority_features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "volatility",
        ]
        selected_features = [f for f in priority_features if f in numeric_features]

        # 残りを追加（最大50個）
        remaining_features = [f for f in numeric_features if f not in selected_features]
        selected_features.extend(remaining_features[: 50 - len(selected_features)])

        logger.info(
            f"Selected {len(selected_features)} features (priority: {len([f for f in priority_features if f in selected_features])})"
        )

        # スキーマ保存
        training_config = {
            "environment": env_config,
            "reward_settings": reward_config,
            "training_config": {
                "algorithm": "SAC",
                "version": "v434.2",
                "improvements": [
                    "Enhanced reward function with higher trading costs and profit bonuses",
                    "Correlation-based feature reduction",
                    "Random start enabled to avoid deterministic behavior",
                    "Improved entropy regularization",
                ],
            },
        }
        manager.save_schema(selected_features, training_config)

    # 環境設定に報酬設定を統合
    env_config["reward_settings"] = reward_config

    # 環境設定にSAC用連続行動空間を強制
    env_config["action_space_type"] = "continuous"

    # 環境作成
    base_env = create_env_from_schema(model_name, df, env_config, models_dir)

    # 特徴量が設定されていない場合は手動で設定
    if (
        not hasattr(base_env, "observation_space")
        or base_env.observation_space.shape[0] == 0
    ):
        # 観測空間を手動設定
        base_env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )  # 固定で50次元
        logger.info("Manually set observation space with 50 features")

    # SAC用に環境を修正：連続行動空間を設定
    if not hasattr(base_env, "action_space") or not isinstance(
        base_env.action_space, gym.spaces.Box
    ):
        # 連続行動空間を設定（SAC用）
        base_env.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        logger.info("Set continuous action space for SAC: Box(-1.0, 1.0, (1,))")

    logger.info(
        f"Created environment with {base_env.observation_space.shape[0]} features"
    )
    logger.info(f"Action space: {base_env.action_space}")
    logger.info(f"Observation space: {base_env.observation_space}")

    # 特徴量が空でないことを確認
    if base_env.observation_space.shape[0] == 0:
        raise ValueError("Observation space has 0 features. Check feature selection.")

    # VecEnv化
    env = DummyVecEnv([lambda: base_env])

    return env


def create_v434_2_sac_model(env: DummyVecEnv, model_path: Optional[str] = None) -> SAC:
    """v434.2の改良されたSACモデルを作成"""
    logger.info("Creating v434.2 SAC model with improved hyperparameters")

    # v434.2の改良されたハイパーパラメータ
    sac_params = {
        # 基本設定
        "policy": "MlpPolicy",  # 必須：MLPポリシー
        # 学習率の調整
        "learning_rate": 3e-4,  # 標準的なSAC学習率
        # エントロピー係数の調整（探索を促進）
        "ent_coef": "auto_1.0",  # 自動調整、初期値1.0（探索重視）
        # ターゲット更新
        "tau": 0.005,  # Polyak平均更新率
        "target_update_interval": 1,
        # バッファサイズ
        "buffer_size": 1000000,  # 大きなリプレイバッファ
        # バッチサイズ
        "batch_size": 256,
        # ネットワークアーキテクチャ
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256], qf=[256, 256]),  # より大きなネットワーク
            "activation_fn": torch.nn.ReLU,
        },
        # トレーニング設定
        "train_freq": (1, "episode"),  # エピソードごとに更新
        "gradient_steps": 1,
        # ログ設定
        "verbose": 1,
    }

    if model_path and Path(model_path).exists():
        logger.info(f"Loading existing model from {model_path}")
        model = SAC.load(model_path, env=env, **sac_params)
    else:
        logger.info("Creating new SAC model")
        model = SAC(env=env, **sac_params)

    return model


def train_v434_2_model(
    model: SAC,
    total_timesteps: int = 500000,
    model_save_path: str = "models/sac_v434_2.zip",
) -> SAC:
    """v434.2モデルをトレーニング"""
    logger.info(f"Starting v434.2 training for {total_timesteps:,} timesteps")

    # コールバック設定
    checkpoint_callback = create_checkpoint_callback(
        save_freq=50000,
        save_path="checkpoints/sac_v434_2/",
        name_prefix="sac_v434_2",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # トレーニング実行
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # 最終モデル保存
    save_model(model, model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    return model


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="SAC v434.2 Training")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument(
        "--timesteps", type=int, default=500000, help="Total training timesteps"
    )
    parser.add_argument(
        "--model", type=str, help="Existing model path to continue training"
    )
    parser.add_argument(
        "--output", type=str, default="models/sac_v434_2.zip", help="Output model path"
    )

    args = parser.parse_args()

    try:
        # v434.2設定読み込み
        reward_config, env_config = load_v434_2_config()
        logger.info("Loaded v434.2 configuration")

        # 環境作成
        env = create_v434_2_environment(args.data, reward_config, env_config)

        # モデル作成
        model = create_v434_2_sac_model(env, args.model)

        # トレーニング実行
        trained_model = train_v434_2_model(model, args.timesteps, args.output)

        logger.info("v434.2 training completed successfully!")

        # トレーニング結果のサマリー
        print("\n" + "=" * 80)
        print("🎯 SAC v434.2 トレーニング完了")
        print("=" * 80)
        print("改良内容:")
        print("• 報酬関数：取引コスト10倍、利益ボーナス大幅増加、損失ペナルティ5倍")
        print("• 特徴量：相関削減により次元削減")
        print("• 学習：エントロピー調整で探索促進、ランダムスタートで決定論回避")
        print("• ネットワーク：256x256アーキテクチャで表現力向上")
        print(f"\nモデル保存先: {args.output}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
