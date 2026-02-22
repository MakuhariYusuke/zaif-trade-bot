"""
Stochastic推論でのバックテスト (deterministic=False)

v393でdeterministic=Trueだと全てHOLDになるため、
deterministic=Falseでの収益性を評価する
"""

import argparse
from pathlib import Path
from typing import Any, List, cast

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.io.data_loader import DataLoader


def run_stochastic_backtest(
    model_path: str,
    data_path: str,
    episodes: int = 10,
) -> None:
    """Stochastic推論でバックテストを実行"""
    model_path_obj = Path(model_path)
    print(f"\n{'='*80}")
    print(f"Stochastic Backtest (deterministic=False): {model_path_obj.stem}")
    print(f"{'='*80}\n")

    # データ読み込み
    df = DataLoader.load_csv_optimized(data_path)
    print(f"Data: {len(df):,} rows")

    # 環境作成（スキーマベース）
    base_env = create_env_from_model_path(model_path, df)
    obs_shape = base_env.observation_space.shape
    if obs_shape is not None:
        print(f"Environment: {obs_shape[0]} features")
    else:
        print("Environment: unknown features")

    # VecEnv化
    env = DummyVecEnv([lambda: base_env])

    # モデル読み込み
    model = MaskablePPO.load(model_path, env=env)
    print("Model loaded\n")

    # バックテスト実行
    episode_rewards: List[float] = []
    episode_returns: List[float] = []
    total_trades = 0
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    initial_portfolio_value = base_env.initial_portfolio_value

    for ep in range(episodes):
        obs = env.reset()
        episode_reward = 0.0
        trades = 0
        step_count = 0

        while True:
            # 🎯 deterministic=False でStochastic推論
            # VecEnvの場合、obsはタプルなので最初の要素を使用
            obs_array = obs[0] if isinstance(obs, tuple) else obs
            action, _ = model.predict(obs_array, deterministic=False)
            action_counts[int(action[0])] += 1

            obs, reward, done, info = env.step(action)
            episode_reward += float(reward[0])
            step_count += 1

            # doneはnumpy配列なのでboolに変換
            if bool(cast(Any, done)[0]):
                info_list = cast(Any, info)  # VecEnvの戻り値はリスト
                final_value = info_list[0].get(
                    "portfolio_value", initial_portfolio_value
                )
                trades = int(info_list[0].get("total_trades", 0))
                total_trades += trades

                returns = (
                    (final_value - initial_portfolio_value) / initial_portfolio_value
                ) * 100
                episode_rewards.append(episode_reward)
                episode_returns.append(returns)

                print(
                    f"Episode {ep+1:2d}: "
                    f"Reward={episode_reward:7.2f}, "
                    f"Return={returns:6.2f}%, "
                    f"Trades={trades:3d}, "
                    f"Final={final_value:,.2f}円, "
                    f"Steps={step_count}"
                )
                break

    # 統計表示
    episode_rewards_array = np.array(episode_rewards)
    episode_returns_array = np.array(episode_returns)

    print(f"\n{'='*80}")
    print("RESULTS (Stochastic Inference)")
    print(f"{'='*80}")
    print(
        f"Average Reward:   {episode_rewards_array.mean():7.2f} ± {episode_rewards_array.std():6.2f}"
    )
    print(
        f"Average Return:   {episode_returns_array.mean():6.2f}% ± {episode_returns_array.std():5.2f}%"
    )
    print(f"Best Return:      {episode_returns_array.max():6.2f}%")
    print(f"Worst Return:     {episode_returns_array.min():6.2f}%")
    print(f"Total Trades:   {total_trades:4d}")
    print(f"Trades/Episode: {total_trades/episodes:5.1f}")
    print()

    # Action分布
    total_actions = sum(action_counts.values())
    print("Action Distribution (Stochastic):")
    print(
        f"  HOLD: {action_counts[0]:4d}/{total_actions} ({action_counts[0]/total_actions*100:5.1f}%)"
    )
    print(
        f"  BUY:  {action_counts[1]:4d}/{total_actions} ({action_counts[1]/total_actions*100:5.1f}%)"
    )
    print(
        f"  SELL: {action_counts[2]:4d}/{total_actions} ({action_counts[2]/total_actions*100:5.1f}%)"
    )
    print(f"{'='*80}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stochastic Backtest")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")

    args = parser.parse_args()
    run_stochastic_backtest(args.model, args.data, args.episodes)


if __name__ == "__main__":
    main()
