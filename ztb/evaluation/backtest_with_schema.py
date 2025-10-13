#!/usr/bin/env python3
"""
Schema-aware Backtest Script

モデルのスキーマ情報を自動検出してバックテストを実行します。
v381（110特徴量）とv384（68特徴量）の両方に対応。
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from datetime import datetime
from typing import Any

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.training.policies.policy_utils import predict_with_masks
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_backtest_with_schema(
    model_path: str,
    data_path: str,
    episodes: int = 10
) -> dict[str, Any]:
    """
    スキーマを考慮したバックテスト

    Args:
        model_path: モデルファイルパス
        data_path: データファイルパス
        episodes: エピソード数

    Returns:
        バックテスト結果
    """
    logger.info("="*80)
    logger.info("Schema-aware Backtest")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")

    # データ読み込み
    df = load_csv_data_optimized(data_path)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # スキーマベースで環境作成（自動的に特徴量数を調整）
    # Note: create_env_from_model_path内でデフォルトで
    # enable_correlation_reduction=Falseが設定される
    env = create_env_from_model_path(model_path, df)
    logger.info(f"Environment created with {env.observation_space.shape[0] if hasattr(env, 'observation_space') and env.observation_space else 0} features")
    
    # 環境の特徴量リストを確認
    if hasattr(env, 'feature_names') and env.feature_names is not None:
        logger.info(f"Environment features ({len(env.feature_names)}): {env.feature_names[:5]}...")
    
    # データの利用可能な特徴量を確認
    available_features = set(df.columns)
    required_features = set(env.feature_names) if hasattr(env, 'feature_names') and env.feature_names is not None else set()
    missing_features = required_features - available_features
    if missing_features:
        logger.warning(f"Missing features in data: {missing_features}")
    
    required_count = len(required_features) if required_features else (env.observation_space.shape[0] if hasattr(env, 'observation_space') and env.observation_space else 0)
    logger.info(f"Data has {len(df.columns)} columns, environment needs {required_count} features")

    # モデル読み込み
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # バックテスト実行
    all_rewards = []
    all_pnls = []
    all_returns = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    
    # Get initial portfolio value from environment
    initial_portfolio_value = getattr(env, 'initial_portfolio_value', 100000.0)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_pnl = 0.0
        steps = 0
        final_portfolio = initial_portfolio_value

        while not (done or truncated) and steps < 1000:
            action, _ = predict_with_masks(model, obs, env, deterministic=True)
            action = action.item()

            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            ep_pnl += info.get('pnl', 0.0)
            
            # Track final portfolio value
            if 'portfolio_value' in info:
                final_portfolio = info['portfolio_value']

            # Count actions
            if action == 0:
                action_counts["HOLD"] += 1
            elif action == 1:
                action_counts["BUY"] += 1
            else:
                action_counts["SELL"] += 1

            steps += 1
        
        # Calculate return percentage
        return_pct = ((final_portfolio - initial_portfolio_value) / initial_portfolio_value) * 100

        all_rewards.append(ep_reward)
        all_pnls.append(ep_pnl)
        all_returns.append(return_pct)

        logger.info(
            f"Episode {ep+1:2d}: Reward={ep_reward:7.2f}, "
            f"Return={return_pct:6.2f}%, PnL={ep_pnl:10,.0f} JPY, Steps={steps:4d}"
        )

    # 結果サマリー
    total_actions = sum(action_counts.values())

    results = {
        "model_path": model_path,
        "data_path": data_path,
        "episodes": episodes,
        "avg_reward": float(np.mean(all_rewards)),
        "avg_return_pct": float(np.mean(all_returns)),
        "best_return_pct": float(np.max(all_returns)),
        "worst_return_pct": float(np.min(all_returns)),
        "avg_pnl": float(np.mean(all_pnls)),
        "total_pnl": float(np.sum(all_pnls)),
        "action_distribution": {
            k: {"count": v, "pct": (v/total_actions*100 if total_actions > 0 else 0)}
            for k, v in action_counts.items()
        },
    }

    logger.info("\n" + "="*80)
    logger.info("Backtest Results")
    logger.info("="*80)
    logger.info(f"Average Reward: {results['avg_reward']:.2f}")
    logger.info(f"Average Return: {results['avg_return_pct']:.2f}%")
    logger.info(f"Best Return: {results['best_return_pct']:.2f}%")
    logger.info(f"Worst Return: {results['worst_return_pct']:.2f}%")
    logger.info(f"Average PnL: {results['avg_pnl']:,.0f} JPY")
    logger.info(f"Total PnL: {results['total_pnl']:,.0f} JPY")
    logger.info(f"Action Distribution:")
    action_dist = results.get('action_distribution', {})
    if isinstance(action_dist, dict):
        for action, stats in action_dist.items():
            if 'count' in stats and 'pct' in stats:
                logger.info(f"  {action}: {stats['count']:,} ({stats['pct']:.1f}%)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema-aware Backtest")
    parser.add_argument("--model", required=True, help="Model path (.zip)")
    parser.add_argument("--data", default="ml-dataset-enhanced.csv", help="Data path (.csv)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    try:
        results = run_backtest_with_schema(
            model_path=args.model,
            data_path=args.data,
            episodes=args.episodes
        )

        # 結果保存
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())