#!/usr/bin/env python3
"""
SACモデル用簡易バックテストスクリプト
スキーマ不要バージョン
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ztb.metrics.metrics import calculate_distribution_stats
from ztb.trading.environment.constants import DEFAULT_TRANSACTION_COST
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.data_utils import load_csv_data_optimized


def run_simple_sac_backtest(model_path: str, data_path: str, episodes: int = 10):
    """
    SACモデルの簡易バックテスト
    """
    model_path = Path(model_path)
    print(f"\n{'='*80}")
    print(f"Simple SAC Backtest: {model_path.stem}")
    print(f"{'='*80}\n")

    # データ読み込み
    df = load_csv_data_optimized(data_path)
    print(f"Data: {len(df):,} rows")

    # 環境作成（HeavyTradingEnv直接使用）
    env_config = EnvironmentConfig(
        initial_portfolio_value=200000.0,
        transaction_cost=DEFAULT_TRANSACTION_COST,
        max_position_size=1.0,
        use_continuous_actions=True,
        reward_scaling=100.0,
        exchange="zaif",
    )

    env = HeavyTradingEnv(df=df, config=env_config)
    print(f"Environment: {env.observation_space.shape[0]} features")

    # SACモデル読み込み
    model = SAC.load(str(model_path))
    print("SAC Model loaded\n")

    # バックテスト実行
    episode_rewards = []
    episode_returns = []
    total_trades = 0

    initial_portfolio_value = env.initial_portfolio_value

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_trades = 0

        while not done:
            # SAC予測（決定的）
            action, _ = model.predict(obs, deterministic=True)

            # ステップ
            step_result = env.step(action)
            obs, reward, done, truncated, info = step_result

            ep_reward += reward

            # トレード回数カウント（連続アクションなので閾値で判定）
            if abs(action[0]) > 0.1:  # 閾値以上のアクション
                ep_trades += 1

        # エピソード統計
        # ポジション強制クローズ
        if env.position != 0:
            try:
                final_close_pnl = env.position_manager.close_position(env.current_step)
                env._sync_from_position_manager()
                print(f"  ⚠️  Forced position close: PnL = {final_close_pnl:+.2f} 円")
            except Exception as e:
                print(f"  ⚠️  Position close error: {e}")

        # 最終ポートフォリオ値を取得
        final_value = env.initial_portfolio_value + env.realized_pnl
        return_pct = (
            (final_value - initial_portfolio_value) / initial_portfolio_value
        ) * 100

        episode_rewards.append(ep_reward)
        episode_returns.append(return_pct)
        total_trades += ep_trades

        print(
            f"Episode {ep+1:2d}: Reward={ep_reward:7.2f}, Return={return_pct:6.2f}%, Trades={ep_trades:3d}, Final={final_value:,.2f}円"
        )

    # サマリー
    print(f"\n{'='*80}")
    print("BACKTEST RESULTS")
    print(f"{'='*80}")

    reward_stats = calculate_distribution_stats(episode_rewards)
    return_stats = calculate_distribution_stats(episode_returns)

    print(f"Average Reward:  {reward_stats['mean']:7.2f} ± {reward_stats['std']:6.2f}")
    print(
        f"Average Return:  {return_stats['mean']:6.2f}% ± {return_stats['std']:5.2f}%"
    )
    print(f"Best Return:     {np.max(episode_returns):6.2f}%")
    print(f"Worst Return:    {np.min(episode_returns):6.2f}%")
    print(f"Total Trades:    {total_trades}")
    print(f"Trades/Episode:  {total_trades/episodes:.1f}")
    print(f"{'='*80}\n")

    return {
        "model_name": model_path.stem,
        "avg_reward": reward_stats["mean"],
        "std_reward": reward_stats["std"],
        "avg_return": return_stats["mean"],
        "std_return": return_stats["std"],
        "best_return": np.max(episode_returns),
        "worst_return": np.min(episode_returns),
        "total_trades": total_trades,
        "trades_per_episode": total_trades / episodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, required=True, help="Data path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    args = parser.parse_args()

    run_simple_sac_backtest(args.model, args.data, args.episodes)
