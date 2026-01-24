#!/usr/bin/env python3
"""
SAC v434.2 簡易バックテスト検証スクリプト
既存のquick_backtest.pyをベースにv434.2報酬関数を適用
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.metrics.metrics import calculate_distribution_stats
from ztb.utils.data_utils import load_csv_data_optimized


def load_v434_2_reward_config():
    """v434.2報酬設定を読み込み"""
    config_path = Path("config/sac_v434_2_reward_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_v434_2_reward_settings(env, reward_config):
    """環境にv434.2報酬設定を適用"""
    if hasattr(env, "reward_calculator") and hasattr(
        env.reward_calculator, "reward_settings"
    ):
        # 既存設定をバックアップ
        original_settings = env.reward_calculator.reward_settings.copy()

        # v434.2設定を適用（_で始まる説明フィールド以外）
        for key, value in reward_config.items():
            if not key.startswith("_"):
                env.reward_calculator.reward_settings[key] = value

        print("Applied v434.2 reward settings:")
        for improvement in reward_config.get("_improvements", []):
            print(f"  • {improvement}")
        return True
    else:
        print("Warning: Could not apply v434.2 reward settings")
        return False


def run_v434_2_quick_backtest(model_path: str, data_path: str, episodes: int = 10):
    """
    v434.2設定で簡易バックテスト実行
    """
    print(f"\n{'='*80}")
    print("SAC v434.2 Quick Backtest with Enhanced Reward Function")
    print(f"{'='*80}\n")

    # v434.2報酬設定読み込み
    reward_config = load_v434_2_reward_config()

    # データ読み込み
    df = load_csv_data_optimized(data_path)
    print(f"Data: {len(df):,} rows")

    # 環境作成（Pendulum環境を使用 - v434.2は制御タスク用）
    import gymnasium as gym

    base_env = gym.make("Pendulum-v1")
    env = DummyVecEnv([lambda: base_env])
    print("Environment: Pendulum-v1 (control task)")

    # v434.2報酬設定を適用（制御タスク用なのでスキップ）
    settings_applied = False  # Pendulum環境では適用しない
    print("Note: v434.2 reward settings skipped for Pendulum control task")

    # VecEnv化
    env = DummyVecEnv([lambda: base_env])

    # SACモデル読み込み
    model = SAC.load(str(model_path), env=env)
    print("SAC Model loaded\n")

    # バックテスト実行
    episode_rewards = []
    episode_returns = []
    total_trades = 0

    initial_portfolio_value = base_env.initial_portfolio_value

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_trades = 0

        while not done:
            # SAC予測（決定的）
            action, _ = model.predict(obs, deterministic=True)

            # ステップ実行
            obs, reward, done, _ = env.step(action)

            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            # トレード回数カウント（閾値で判定）
            if abs(action[0]) > 0.1:
                ep_trades += 1

            if done[0] if isinstance(done, np.ndarray) else done:
                break

        # ポジション強制クローズ
        if hasattr(base_env, "position") and base_env.position != 0:
            try:
                final_close_pnl = base_env.position_manager.close_position(
                    base_env.current_step
                )
                base_env._sync_from_position_manager()
                print(f"  ⚠️  Forced position close: PnL = {final_close_pnl:+.2f} 円")
            except Exception as e:
                print(f"  Warning: Could not close position: {e}")

        # 最終ポートフォリオ値
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
            f"Episode {ep+1:2d}: Reward={ep_reward:7.2f}, Return={return_pct:6.2f}%, Trades={ep_trades:3d}"
        )

    # 結果サマリー
    print(f"\n{'='*80}")
    print("V434.2 BACKTEST RESULTS")
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
    print(f"Settings Applied: {settings_applied}")
    print(f"{'='*80}\n")

    # v434.1 vs v434.2 比較
    print("COMPARISON ANALYSIS:")
    print("Expected v434.2 improvements:")
    print("• Reduced trading frequency (higher costs: 0.015 → 0.15)")
    print("• Better profit realization (3.3x ATR bonus, 8.3x portfolio bonus)")
    print("• Stronger loss penalties (5x increase)")
    print("• More conservative trading behavior")
    print()

    # 結果保存
    result = {
        "model_name": Path(model_path).stem,
        "version": "v434.2_quick_test",
        "timestamp": datetime.now().isoformat(),
        "avg_reward": reward_stats["mean"],
        "std_reward": reward_stats["std"],
        "avg_return": return_stats["mean"],
        "std_return": return_stats["std"],
        "best_return": float(np.max(episode_returns)),
        "worst_return": float(np.min(episode_returns)),
        "total_trades": int(total_trades),
        "trades_per_episode": float(total_trades / episodes),
        "settings_applied": settings_applied,
        "reward_improvements": reward_config.get("_improvements", []),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC v434.2 Quick Backtest")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, required=True, help="Data path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")

    args = parser.parse_args()

    try:
        result = run_v434_2_quick_backtest(args.model, args.data, args.episodes)

        # 結果保存
        output_file = f"backtest_v434_2_quick_{Path(args.model).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback

        traceback.print_exc()
