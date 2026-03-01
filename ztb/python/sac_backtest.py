#!/usr/bin/env python3
"""
SACモデル用バックテストスクリプト
v434.1学習戦略検証用
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.metrics.metrics import calculate_distribution_stats
from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.io.data_loader import DataLoader


def run_sac_backtest(model_path: str, data_path: str, episodes: int = 10):
    """
    SACモデルのバックテスト実行
    """
    model_path = Path(model_path)
    print(f"\n{'='*80}")
    print(f"SAC Backtest: {model_path.stem}")
    print(f"{'='*80}\n")

    # データ読み込み
    df = DataLoader.load_csv_optimized(data_path)
    print(f"Data: {len(df):,} rows")

    # 環境作成（スキーマベース）
    base_env = create_env_from_model_path(model_path, df)
    print(f"Environment: {base_env.observation_space.shape[0]} features")

    # VecEnv化（SAC用）
    env = DummyVecEnv([lambda: base_env])

    # SACモデル読み込み
    model = SAC.load(str(model_path), env=env)
    print("SAC Model loaded\n")

    # バックテスト実行
    episode_rewards = []
    episode_returns = []
    total_trades = 0

    # 環境から初期ポートフォリオ値を取得
    initial_portfolio_value = base_env.initial_portfolio_value

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_trades = 0

        while not done:
            # SAC予測（決定的）
            action, _ = model.predict(obs, deterministic=True)

            # ステップ
            obs, reward, done, _ = env.step(action)

            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            # トレード回数カウント（連続アクションなので閾値で判定）
            if abs(action[0]) > 0.1:  # 閾値以上のアクション
                ep_trades += 1

            if done[0] if isinstance(done, np.ndarray) else done:
                break

        # エピソード統計
        # ポジション強制クローズ
        if hasattr(base_env, "position") and base_env.position != 0:
            try:
                final_close_pnl = base_env.position_manager.close_position(
                    base_env.current_step
                )
                base_env._sync_from_position_manager()
                print(f"  ⚠️  Forced position close: PnL = {final_close_pnl:+.2f} 円")
            except Exception:
                pass

        # 最終ポートフォリオ値を取得
        final_value = base_env.initial_portfolio_value + getattr(
            base_env, "realized_pnl", 0
        )
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
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
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

    run_sac_backtest(args.model, args.data, args.episodes)
