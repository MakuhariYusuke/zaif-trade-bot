"""
SAC v446 Algorithm Tuning Script

Optimizes PPO learning parameters and neural network architectures using Optuna
(PPO is used instead of SAC due to discrete action space compatibility)
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.utils.logging_utils import setup_logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

setup_logging(level=logging.INFO)

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from backtest.data_generator import generate_synthetic_data
from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


class ActionWrapper(gym.Wrapper):
    """Wrapper to convert PPO actions to int for environment compatibility."""

    def step(self, action):
        # Convert numpy array to int
        if isinstance(action, np.ndarray):
            action = int(action.item())
        return self.env.step(action)


def create_training_env(data_df, config, params=None):
    """訓練用環境を作成"""
    logger = logging.getLogger(__name__)

    # V4FeatureExtractorで特徴量を拡張
    logger.info("🔧 Applying V4FeatureExtractor with short-term enhanced features...")
    feature_extractor = V4FeatureExtractor(config={})

    # 特徴量抽出
    enhanced_df = feature_extractor.generate_features(
        data_df, feature_set="full", model_type="sac"
    )
    logger.info(f"✅ Enhanced features: {enhanced_df.shape[1]} columns")

    # 環境設定
    reward_scaling = params.get("reward_scaling", 1.0) if params else 1.0
    max_position_size = params.get("max_position", 0.1) if params else 0.1

    env_config = EnvironmentConfig(
        reward_scaling=reward_scaling,
        transaction_cost=0.001,
        commission=0.0,
        slippage=0.0,
        max_steps=1000,
        max_position_size=max_position_size,
        risk_free_rate=0.0,
        timeframe="1m",
        feature_set="full",
        feature_names=None,  # Auto-detect from data
        exchange="coincheck",
        stop_loss_threshold=0.05,
        max_consecutive_trades=5,
        min_holding_period=3,
        reward_position_soft_cap=0.8,
        reward_position_penalty_scale=0.5,
        reward_position_penalty_exponent=4.0,
        reward_inventory_window=128,
        reward_inventory_penalty_scale=0.1,
        reward_trade_frequency_penalty=0.2,
        reward_trade_frequency_halflife=8.0,
        reward_trade_cooldown_steps=2,
        reward_trade_cooldown_penalty=0.2,
        reward_max_consecutive_trades=5,
        reward_consecutive_trade_penalty=0.1,
        reward_volatility_window=32,
        reward_volatility_penalty_scale=0.05,
        reward_sharpe_bonus_scale=0.02,
        reward_clip_value=10000.0,
        enable_forced_diversity=False,
        action_bonuses={
            "buy_action_bonus": 0.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
        },
        base_action_penalty=0.015,
        behavior_optimization={},
        initial_portfolio_value=200000.0,
        reward_profit_bonus_multipliers=[1.0, 1.0, 0.8],
        reward_settings=None,
        memory_logging_enabled=False,
        memory_log_interval_steps=None,
        max_action_history=512,
        use_standardized_observations=True,
        use_continuous_actions=False,  # Discrete actions for PPO
        action_space_type=None,
        target_feature_count=None,
        enable_action_masking=False,
        continuous_to_discrete_threshold=0.3333,
        continuous_to_discrete_threshold_neg=-0.3333,
        signal_guidance_enabled=False,
        signal_guidance_mode="partial",
        signal_bonus_weight=0.1,
        signal_penalty_weight=0.05,
        signal_weight=1.0,
        guidance_decay=0.95,
        market_regime=None,
        advanced_market_regime=None,
        dynamic_reward_shaping=None,
        adaptive_feature_selection=None,
        allow_reverse=True,
        enforce_reverse_cooldown=False,
        random_start=False,
    )

    # 環境作成
    env = HeavyTradingEnv(
        data_df=enhanced_df,
        config=env_config,
    )

    # Action wrapper to handle PPO discrete actions
    env = ActionWrapper(env)

    return env, enhanced_df


def create_sac_model(env, ppo_params, net_arch_params):
    """PPOモデルを作成（SACの代わりにPPOを使用）"""
    from stable_baselines3 import PPO

    logger = logging.getLogger(__name__)

    # ネットワークアーキテクチャ設定
    network_type = net_arch_params.get("network_type", "mlp")

    if network_type == "mlp":
        # MLPアーキテクチャ
        net_arch = net_arch_params.get("net_arch", [256, 256])
        activation_fn = net_arch_params.get("activation_fn", nn.ReLU)
        policy_kwargs = {
            "net_arch": {"pi": net_arch, "vf": net_arch},
            "activation_fn": activation_fn,
        }
    else:
        # デフォルトMLP
        net_arch = net_arch_params.get("net_arch", [256, 256])
        policy_kwargs = {
            "net_arch": {"pi": net_arch, "vf": net_arch},
            "activation_fn": nn.ReLU,
        }

    # PPOモデル作成（SACの代わりにPPOを使用）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_params["learning_rate"],
        n_steps=2048,  # PPO特有パラメータ
        batch_size=ppo_params["batch_size"],
        n_epochs=10,
        gamma=ppo_params["gamma"],
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clip_range=ppo_params.get("clip_range", 0.2),
        ent_coef=ppo_params["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cpu",  # CPU環境
    )

    return model


def train_and_evaluate_ppo(params, n_timesteps=5000):
    """PPOモデルを訓練し、評価する"""
    logger = logging.getLogger(__name__)

    try:
        # データ生成
        logger.info("📊 Generating synthetic training data...")
        data_config = {
            "n_periods": 1000,  # 訓練用データサイズ
            "start_price": 50000.0,
            "volatility": 500,
        }
        data_df = generate_synthetic_data(**data_config)

        # 環境作成
        from ztb.trading.environment.utils.config import EnvironmentConfig

        env_config = EnvironmentConfig(
            reward_scaling=1.0,
            transaction_cost=0.001,
            commission=0.0,
            slippage=0.0,
            max_steps=1000,
            max_position_size=0.1,
            risk_free_rate=0.0,
            timeframe="1m",
            feature_set="full",
            exchange="coincheck",
            stop_loss_threshold=0.05,
            max_consecutive_trades=5,
            min_holding_period=3,
            reward_position_soft_cap=0.8,
            reward_position_penalty_scale=0.5,
            reward_position_penalty_exponent=4.0,
            reward_inventory_window=128,
            reward_inventory_penalty_scale=0.1,
            reward_trade_frequency_penalty=0.2,
            reward_trade_frequency_halflife=8.0,
            reward_trade_cooldown_steps=2,
            reward_trade_cooldown_penalty=0.2,
            reward_max_consecutive_trades=5,
            reward_consecutive_trade_penalty=0.1,
            reward_volatility_window=32,
            reward_volatility_penalty_scale=0.05,
            reward_sharpe_bonus_scale=0.02,
            reward_clip_value=10000.0,
            enable_forced_diversity=False,
            action_bonuses={
                "buy_action_bonus": 0.0,
                "sell_action_bonus": 0.0,
                "hold_action_bonus": 0.0,
            },
            base_action_penalty=0.015,
            behavior_optimization={},
            initial_portfolio_value=200000.0,
            reward_profit_bonus_multipliers=[1.0, 1.0, 0.8],
            memory_logging_enabled=False,
            max_action_history=512,
            use_standardized_observations=True,
            use_continuous_actions=False,
            enable_action_masking=False,
            continuous_to_discrete_threshold=0.3333,
            continuous_to_discrete_threshold_neg=-0.3333,
            signal_guidance_enabled=False,
            signal_guidance_mode="partial",
            signal_bonus_weight=0.1,
            signal_penalty_weight=0.05,
            signal_weight=1.0,
            guidance_decay=0.95,
            allow_reverse=True,
            enforce_reverse_cooldown=False,
            random_start=False,
        )

        # 環境を直接作成
        env = HeavyTradingEnv(df=data_df, config=env_config)

        # PPOパラメータ
        ppo_params = {
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "gamma": params["gamma"],
            "ent_coef": params["ent_coef"],
            "clip_range": params["clip_range"],
            "gae_lambda": params["gae_lambda"],
        }

        # ネットワークアーキテクチャパラメータ
        net_arch_params = {
            "network_type": params["network_type"],
            "net_arch": params["net_arch"],
            "activation_fn": params["activation_fn"],
        }

        # モデル作成
        logger.info("🏗️ Creating PPO model...")
        model = create_sac_model(env, ppo_params, net_arch_params)

        # 訓練
        logger.info(f"🎯 Training PPO model for {n_timesteps} timesteps...")
        model.learn(total_timesteps=n_timesteps, progress_bar=False)

        # 評価（シンプルなバックテスト）
        logger.info("📈 Evaluating trained model...")
        obs, _ = env.reset()
        total_reward = 0
        episode_rewards = []

        for _ in range(500):  # 500ステップ評価
            action, _ = model.predict(obs, deterministic=True)
            # PPO returns numpy array for discrete actions, convert to int
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                episode_rewards.append(total_reward)
                obs, _ = env.reset()
                total_reward = 0

        # 平均リターンを計算
        if episode_rewards:
            avg_return = np.mean(episode_rewards)
        else:
            avg_return = total_reward

        logger.info(".4f")

        # 環境クリーンアップ
        env.close()

        return avg_return

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return -1000.0  # 失敗時は低いスコア


def objective(trial):
    """Optunaの目的関数"""
    logger = logging.getLogger(__name__)

    # PPO学習パラメータの最適化
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.01, 0.1, 1.0])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)

    # ネットワークアーキテクチャパラメータの最適化
    network_type = trial.suggest_categorical("network_type", ["mlp"])

    # MLPアーキテクチャ
    if network_type == "mlp":
        # 隠れ層構造の選択
        net_arch_options = [
            [128, 128],
            [256, 256],
            [128, 256, 128],
            [256, 128, 64],
            [512, 256],
        ]
        net_arch_idx = trial.suggest_int("net_arch_idx", 0, len(net_arch_options) - 1)
        net_arch = net_arch_options[net_arch_idx]

        # 活性化関数
        activation_options = [nn.ReLU, nn.Tanh, nn.ELU]
        activation_idx = trial.suggest_int(
            "activation_idx", 0, len(activation_options) - 1
        )
        activation_fn = activation_options[activation_idx]
    else:
        # デフォルト
        net_arch = [256, 256]
        activation_fn = nn.ReLU

    # パラメータ辞書の作成
    params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "network_type": network_type,
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }

    logger.info(f"🔍 Trial {trial.number}: Testing parameters: {params}")

    # 訓練と評価
    score = train_and_evaluate_ppo(
        params, n_timesteps=5000
    )  # PPO用にステップ数を増やす

    logger.info(".4f")

    return score


def main():
    """メイン実行関数"""
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting SAC v446 Algorithm Tuning")

    # Optuna Studyの作成
    study = optuna.create_study(
        direction="maximize",  # リターンを最大化
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="sac_v446_algorithm_tuning",
    )

    # 最適化実行
    n_trials = 20  # 試行回数
    logger.info(f"🎯 Running optimization with {n_trials} trials...")

    study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1時間タイムアウト

    # 結果表示
    logger.info("📋 Optimization completed!")
    logger.info(f"🏆 Best trial: {study.best_trial.number}")
    logger.info(".4f")
    logger.info("🔧 Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "timestamp": timestamp,
        "study_name": study.study_name,
        "n_trials": n_trials,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"trial": i, "value": trial.value, "params": trial.params}
            for i, trial in enumerate(study.trials)
        ],
    }

    results_file = results_dir / f"sac_v446_algorithm_tuning_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    main()
