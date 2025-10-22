#!/usr/bin/env python3
"""
SAC v434.2 バックテスト検証スクリプト
v434.1の問題を解決するための改良された報酬関数と環境設定を検証
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.utils.data_utils import load_csv_data_optimized


def load_v434_2_configs():
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


def run_v434_2_backtest(model_path: str, data_path: str, episodes: int = 10):
    """
    v434.2設定でバックテスト実行
    """
    print(f"\n{'='*80}")
    print("SAC v434.2 Backtest with Enhanced Reward Function")
    print(f"{'='*80}\n")

    # v434.2設定読み込み
    reward_config, env_config = load_v434_2_configs()
    print("Loaded v434.2 configurations:")
    print(f"  Reward improvements: {len(reward_config['_improvements'])} items")
    print(f"  Environment improvements: {len(env_config['_improvements'])} items")
    print()

    # データ読み込み
    df = load_csv_data_optimized(data_path)
    print(f"Data: {len(df):,} rows")

    # 環境作成（v434.2設定適用）
    base_env = create_env_from_model_path(model_path, df)

    # v434.2報酬設定を環境に適用
    if hasattr(base_env, "reward_calculator") and hasattr(
        base_env.reward_calculator, "reward_settings"
    ):
        # 既存の設定をバックアップ
        original_settings = base_env.reward_calculator.reward_settings.copy()

        # v434.2設定を適用
        base_env.reward_calculator.reward_settings.update(reward_config)
        print("Applied v434.2 reward settings to environment")

        # 設定内容を表示
        print("\nKey v434.2 Reward Improvements:")
        for improvement in reward_config["_improvements"]:
            print(f"  • {improvement}")
    else:
        print("Warning: Could not apply v434.2 reward settings")

    # v434.2環境設定を適用
    for key, value in env_config.items():
        if key.startswith("_"):
            continue  # 説明フィールドはスキップ
        if hasattr(base_env.config, key):
            setattr(base_env.config, key, value)
            print(f"Applied env config: {key} = {value}")

    print(f"\nEnvironment: {base_env.observation_space.shape[0]} features")

    # VecEnv化（SAC用）
    env = DummyVecEnv([lambda: base_env])

    # SACモデル読み込み
    model = SAC.load(str(model_path), env=env)
    print("SAC Model loaded with v434.2 settings\n")

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
            except:
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
    print("V434.2 BACKTEST RESULTS")
    print(f"{'='*80}")
    print(
        f"Average Reward:  {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):6.2f}"
    )
    print(
        f"Average Return:  {np.mean(episode_returns):6.2f}% ± {np.std(episode_returns):5.2f}%"
    )
    print(f"Best Return:     {np.max(episode_returns):6.2f}%")
    print(f"Worst Return:    {np.min(episode_returns):6.2f}%")
    print(f"Total Trades:    {total_trades}")
    print(f"Trades/Episode:  {total_trades/episodes:.1f}")
    print(f"{'='*80}\n")

    # v434.1 vs v434.2 の比較
    print("COMPARISON WITH V434.1:")
    print("v434.1 issues:")
    print("  • 0% returns despite 4621 trades/episode")
    print("  • Completely deterministic behavior")
    print("  • Excessive trading (92.4% action rate)")
    print("  • Low trading costs (0.015 penalty)")
    print("  • Weak loss penalty (-0.2)")
    print()
    print("v434.2 improvements:")
    print("  • Higher trading costs (0.15 penalty)")
    print("  • Stronger profit bonuses (3.3x ATR, 8.3x portfolio)")
    print("  • Enhanced loss penalty (5x stronger)")
    print("  • Action frequency penalties")
    print("  • Trade interval bonuses")
    print()

    return {
        "model_name": Path(model_path).stem,
        "version": "v434.2",
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "best_return": np.max(episode_returns),
        "worst_return": np.min(episode_returns),
        "total_trades": total_trades,
        "trades_per_episode": total_trades / episodes,
        "reward_improvements": len(reward_config["_improvements"]),
        "env_improvements": len(env_config["_improvements"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC v434.2 Backtest Validation")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, required=True, help="Data path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")

    args = parser.parse_args()

    result = run_v434_2_backtest(args.model, args.data, args.episodes)

    # 結果をJSONに保存
    output_file = f"backtest_results_sac_v434_2_{Path(args.model).stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
