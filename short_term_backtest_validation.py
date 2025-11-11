#!/usr/bin/env python3
"""
Short-term Enhanced Features Backtest
短期間拡張特徴量を使用したバックテスト検証
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import SAC

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import setup_logging
from ztb.utils.config_loader import safe_json_load

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

setup_logging(level=logging.INFO)

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from backtest.data_generator import generate_synthetic_data


def create_short_term_enhanced_env(data_df, config):
    """短期間拡張特徴量を使用した環境を作成"""
    logger = logging.getLogger(__name__)

    # V4FeatureExtractorで特徴量を拡張
    logger.info("🔧 Applying V4FeatureExtractor with short-term enhanced features...")
    feature_extractor = V4FeatureExtractor(config=config)

    # 特徴量抽出
    enhanced_df = feature_extractor.generate_features(data_df, feature_set="full", model_type="sac")

    logger.info(f"✅ Enhanced features: {len(enhanced_df.columns)} columns")
    logger.info(f"📊 New features added: {len(enhanced_df.columns) - len(data_df.columns)}")

    # 特徴量名を取得
    feature_names = feature_extractor.get_available_features(model_type="sac")
    logger.info(f"🎯 Total features: {len(feature_names)}")

    # 環境設定
    env_config = EnvironmentConfig(
        transaction_cost=0.001,    # 0.1% 手数料
        max_position_size=0.1,    # 最大ポジションサイズ 10%
        feature_names=list(enhanced_df.columns),  # データフレームの実際の特徴量を使用
        reward_scaling=1.0,
        max_steps=len(enhanced_df),
    )

    # HeavyTradingEnvの作成
    env = HeavyTradingEnv(
        df=enhanced_df,
        config=env_config,
        initial_balance=1000000,  # 100万円スタート
    )

    return env, enhanced_df


def run_short_term_backtest(model_path=None, config_path=None, n_episodes=1):
    """短期間拡張特徴量を使用したバックテストを実行"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Short-term Enhanced Features Backtest")
    logger.info("=" * 60)

    # 設定ファイルの読み込み
    if config_path is None:
        config_path = Path(__file__).parent / "config.py"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using default config")
        config = {}
    else:
        config = safe_json_load(config_path)
        logger.info(f"✅ Loaded config from {config_path}")

    # テストデータの生成
    logger.info("📊 Generating synthetic market data...")
    data_df = generate_synthetic_data(
        n_periods=1000,  # 1000期間のデータ
        start_price=50000.0,
        volatility=500
    )

    logger.info(f"✅ Generated {len(data_df)} periods of market data")
    logger.info(f"📈 Price range: ${data_df['close'].min():.2f} - ${data_df['close'].max():.2f}")

    # 短期間拡張特徴量環境の作成
    initial_balance = 1000000  # 100万円スタート
    env, enhanced_df = create_short_term_enhanced_env(data_df, config)

    # モデルの読み込みまたはランダムエージェントの使用
    if model_path and Path(model_path).exists():
        logger.info(f"🤖 Loading model from {model_path}")
        model = SAC.load(model_path)
    else:
        logger.info("🎲 Using random agent (no trained model provided)")
        model = None

    # バックテスト実行
    results = []
    total_rewards = []
    total_portfolio_values = []

    for episode in range(n_episodes):
        logger.info(f"🏃 Episode {episode + 1}/{n_episodes}")

        obs, info = env.reset()
        episode_reward = 0
        episode_portfolio_values = [info.get('portfolio_value', info.get('balance', initial_balance))]

        done = False
        step = 0

        while not done:
            # 行動決定
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # ランダム行動
                action = env.action_space.sample()

            # 環境ステップ
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_portfolio_values.append(info.get('portfolio_value', info.get('balance', episode_portfolio_values[-1])))
            step += 1

            if step % 100 == 0:
                logger.debug(f"Step {step}: Balance=${info.get('portfolio_value', info.get('balance', episode_portfolio_values[-1])):.2f}, Reward={reward:.4f}")

        # エピソード結果の保存
        results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'initial_balance': initial_balance,
            'final_balance': info.get('portfolio_value', initial_balance),
            'total_return_pct': ((info.get('portfolio_value', initial_balance) - initial_balance) / initial_balance) * 100,
            'steps': step,
            'portfolio_values': episode_portfolio_values
        })

        total_rewards.append(episode_reward)
        total_portfolio_values.extend(episode_portfolio_values)

        logger.info(f"📊 Episode {episode + 1} Results:")
        logger.info(f"   Total Reward: {episode_reward:.2f}")
        logger.info(".2f")
        logger.info(".2f")

    # 全体結果の集計
    avg_reward = np.mean(total_rewards)
    avg_return = np.mean([r['total_return_pct'] for r in results])
    max_return = np.max([r['total_return_pct'] for r in results])
    min_return = np.min([r['total_return_pct'] for r in results])

    # 詳細なメトリクス計算
    final_portfolio_values = [r['portfolio_values'][-1] for r in results]
    portfolio_returns = []

    for r in results:
        initial = r['initial_balance']
        final = r['final_balance']
        if initial > 0:
            portfolio_returns.append((final - initial) / initial)

    # 統計分析
    if portfolio_returns:
        logger.info("\n📈 Portfolio Performance Summary:")
        logger.info("=" * 40)
        logger.info(f"Average Return: {np.mean(portfolio_returns)*100:.2f}%")
        logger.info(f"Return Std Dev: {np.std(portfolio_returns)*100:.2f}%")
        logger.info(f"Max Return: {np.max(portfolio_returns)*100:.2f}%")
        logger.info(f"Min Return: {np.min(portfolio_returns)*100:.2f}%")
        logger.info(f"Sharpe Ratio: {np.mean(portfolio_returns)/np.std(portfolio_returns):.4f}")

        # 勝率計算
        positive_returns = sum(1 for r in portfolio_returns if r > 0)
        win_rate = positive_returns / len(portfolio_returns) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")

    # 特徴量の効果分析
    logger.info("\n🔍 Feature Analysis:")
    logger.info("=" * 30)

    # 新特徴量の統計
    short_term_features = ['realized_volatility', 'tick_volume_ratio', 'order_flow_imbalance']
    for feature in short_term_features:
        if feature in enhanced_df.columns:
            values = enhanced_df[feature].dropna()
            if len(values) > 0:
                logger.info(f"{feature}:")
                logger.info(f"  Mean: {values.mean():.6f}")
                logger.info(f"  Std: {values.std():.6f}")
                logger.info(f"  Min: {values.min():.6f}")
                logger.info(f"  Max: {values.max():.6f}")

    # 結果の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent / f"short_term_backtest_results_{timestamp}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n💾 Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Short-term Enhanced Features Backtest")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")

    args = parser.parse_args()

    # バックテスト実行
    results = run_short_term_backtest(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.episodes
    )

    print("\n🎯 Short-term Enhanced Features Backtest Completed!")
    print("=" * 50)