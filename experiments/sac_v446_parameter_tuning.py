#!/usr/bin/env python3
"""
SAC v446 Parameter Tuning Script

Optimizes parameters for short-term enhanced features using Optuna
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.utils.config_loader import safe_json_load
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


def create_short_term_enhanced_env(data_df, config, params=None):
    """短期間拡張特徴量を使用した環境を作成"""
    logger = logging.getLogger(__name__)

    # V4FeatureExtractorで特徴量を拡張
    logger.info("🔧 Applying V4FeatureExtractor with short-term enhanced features...")
    feature_extractor = V4FeatureExtractor(config=config)

    # 特徴量抽出
    enhanced_df = feature_extractor.generate_features(
        data_df, feature_set="full", model_type="sac"
    )

    logger.info(f"✅ Enhanced features: {len(enhanced_df.columns)} columns")
    logger.info(
        f"📊 New features added: {len(enhanced_df.columns) - len(data_df.columns)}"
    )

    # 特徴量名を取得
    feature_names = feature_extractor.get_available_features(model_type="sac")
    logger.info(f"🎯 Total features: {len(feature_names)}")

    # パラメータから設定を取得（デフォルト値を使用）
    reward_scaling = params.get("reward_scaling", 1.0) if params else 1.0
    max_position_size = params.get("max_position", 0.1) if params else 0.1

    # 環境設定
    env_config = EnvironmentConfig(
        transaction_cost=0.001,  # 0.1% 手数料
        max_position_size=max_position_size,  # 最大ポジションサイズ
        feature_names=list(enhanced_df.columns),  # データフレームの実際の特徴量を使用
        reward_scaling=reward_scaling,
        max_steps=len(enhanced_df),
    )

    # HeavyTradingEnvの作成
    env = HeavyTradingEnv(
        df=enhanced_df,
        config=env_config,
        initial_balance=1000000,  # 100万円スタート
    )

    return env, enhanced_df


def run_backtest_with_params(params, model_path=None, config_path=None, n_episodes=1):
    """パラメータ付きでバックテストを実行"""
    logger = logging.getLogger(__name__)

    # 設定ファイルの読み込み
    if config_path is None:
        config_path = Path(__file__).parent / "config.py"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using default config")
        config = {}
    else:
        config = safe_json_load(config_path)

    # パラメータからデータを生成
    n_periods = params.get("data_periods", 1000)
    volatility = params.get("data_volatility", 500)
    trend_strength = params.get("data_trend_strength", 0.001)

    logger.info("📊 Generating synthetic market data...")
    data_df = generate_synthetic_data(
        n_periods=n_periods, start_price=50000.0, volatility=volatility
    )

    logger.info(f"✅ Generated {len(data_df)} periods of market data")
    logger.info(
        f"📈 Price range: ${data_df['close'].min():.2f} - ${data_df['close'].max():.2f}"
    )

    # 短期間拡張特徴量環境の作成
    initial_balance = 1000000  # 100万円スタート
    env, enhanced_df = create_short_term_enhanced_env(data_df, config, params)

    # ランダムエージェントを使用（パラメータチューニング時は学習済みモデルを使わない）
    logger.info("🎲 Using random agent for parameter evaluation")
    model = None

    # バックテスト実行
    results = []
    total_rewards = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_portfolio_values = [
            info.get("portfolio_value", info.get("balance", initial_balance))
        ]

        done = False
        step = 0

        while not done:
            # ランダム行動
            action = env.action_space.sample()

            # 環境ステップ
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_portfolio_values.append(
                info.get(
                    "portfolio_value", info.get("balance", episode_portfolio_values[-1])
                )
            )
            step += 1

        # エピソード結果の保存
        final_balance = info.get("portfolio_value", initial_balance)
        total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100

        results.append(
            {
                "episode": episode + 1,
                "total_reward": episode_reward,
                "initial_balance": initial_balance,
                "final_balance": final_balance,
                "total_return_pct": total_return_pct,
                "steps": step,
                "portfolio_values": episode_portfolio_values,
            }
        )

        total_rewards.append(episode_reward)

    # 平均リターンを目的関数として返す
    avg_return = np.mean([r["total_return_pct"] for r in results])

    return {
        "avg_return": avg_return,
        "avg_reward": np.mean(total_rewards),
        "results": results,
    }


def objective(trial):
    """Optunaの目的関数"""
    params = {
        "data_periods": trial.suggest_int("data_periods", 800, 1500),
        "data_volatility": trial.suggest_float("data_volatility", 200, 800),
        "data_trend_strength": trial.suggest_float(
            "data_trend_strength", 0.0005, 0.002
        ),
        "reward_scaling": trial.suggest_float("reward_scaling", 0.5, 2.0),
        "max_position": trial.suggest_float("max_position", 0.05, 0.2),
    }

    results = run_backtest_with_params(params, n_episodes=1)
    objective_value = results["avg_return"]

    logger = logging.getLogger(__name__)
    logger.info(f"Trial {trial.number}: Return={objective_value:.2f}%")

    return objective_value


def main():
    """メイン関数 - パラメータチューニング実行"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting SAC v446 Parameter Tuning")
    logger.info("=" * 60)

    # Optunaスタディの作成
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="sac_v446_parameter_tuning",
    )

    # 最適化実行
    n_trials = 10  # 試行回数
    logger.info(f"🎯 Running {n_trials} trials for parameter optimization...")

    study.optimize(objective, n_trials=n_trials)

    # 結果の表示
    logger.info("\n📊 Parameter Tuning Results:")
    logger.info("=" * 40)
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Value (Return %): {study.best_value:.2f}%")
    logger.info("Best Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # 結果の保存
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "all_trials": [
            {"trial": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
        "timestamp": datetime.now().isoformat(),
        "study_name": "sac_v446_parameter_tuning",
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"sac_v446_tuning_{timestamp}.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info("\n✅ SAC v446 Parameter Tuning Completed!")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC v446 Parameter Tuning")
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of Optuna trials"
    )

    args = parser.parse_args()

    # パラメータチューニング実行
    results = main()

    print("\n🎯 SAC v446 Parameter Tuning Completed!")
    print("=" * 50)
    print(f"Best Return: {results['best_value']:.2f}%")
    print(f"Best Parameters: {results['best_params']}")
