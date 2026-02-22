#!/usr/bin/env python3
"""
SAC v434.2 Pendulum Control Task Evaluation
v434.2モデルは制御タスク用なので、Pendulum環境で評価
"""

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.io.json_io import write_json

def run_v434_2_pendulum_evaluation(model_path: str, episodes: int = 10):
    """
    v434.2モデルをPendulum制御タスクで評価
    """
    print(f"\n{'='*80}")
    print("SAC v434.2 Pendulum Control Task Evaluation")
    print(f"{'='*80}\n")

    # Pendulum環境作成
    env = gym.make("Pendulum-v1")
    vec_env = DummyVecEnv([lambda: env])
    print("Environment: Pendulum-v1 (continuous control task)")

    # SACモデル読み込み
    model = SAC.load(str(model_path), env=vec_env)
    print("SAC Model loaded\n")

    # 評価実行
    episode_rewards = []
    episode_lengths = []

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            # SAC予測（決定的）
            action, _ = model.predict(obs, deterministic=True)

            # ステップ実行
            obs, reward, done, info = vec_env.step(action)

            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            ep_length += 1

            # 最大ステップ数制限（Pendulumのタイムリミット）
            if ep_length >= 200:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        print(f"Episode {ep+1:2d}: Reward={ep_reward:7.2f}, Length={ep_length:3d}")

    # 結果サマリー
    print(f"\n{'='*80}")
    print("V434.2 PENDULUM EVALUATION RESULTS")
    print(f"{'='*80}")
    print(
        f"Average Reward:  {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):6.2f}"
    )
    print(
        f"Average Length:  {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):6.1f}"
    )
    print(f"Best Reward:     {np.max(episode_rewards):7.2f}")
    print(f"Worst Reward:    {np.min(episode_rewards):7.2f}")
    print(f"Total Episodes:  {episodes}")
    print(f"{'='*80}\n")

    # v434.2の制御性能分析
    print("CONTROL PERFORMANCE ANALYSIS:")
    print("• Pendulum task: Balance inverted pendulum using continuous torque control")
    print("• Reward function: Negative of (angle + angular velocity + torque penalty)")
    print("• Goal: Keep pendulum upright with minimal control effort")
    print("• v434.2 improvements: Enhanced SAC algorithm for continuous control")
    print()

    # 結果保存
    result = {
        "model_name": Path(model_path).stem,
        "version": "v434.2_pendulum_eval",
        "timestamp": datetime.now().isoformat(),
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "best_reward": float(np.max(episode_rewards)),
        "worst_reward": float(np.min(episode_rewards)),
        "total_episodes": int(episodes),
        "task_type": "pendulum_control",
        "evaluation_type": "continuous_control_task",
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC v434.2 Pendulum Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")

    args = parser.parse_args()

    try:
        result = run_v434_2_pendulum_evaluation(args.model, args.episodes)

        # 結果保存
        output_file = (
            f"pendulum_eval_v434_2_{Path(args.model).stem}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        write_json(output_file, result, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
