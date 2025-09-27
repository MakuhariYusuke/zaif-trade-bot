#!/usr/bin/env python3
"""
Permutation importance for features using fixed policy
固定ポリシーでのPermutation Importance
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# プロジェクトルートをパスに追加
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from ztb.trading.environment import HeavyTradingEnv
from ztb.features import get_feature_manager


def generate_synthetic_data(n_rows: int = 5000) -> pd.DataFrame:
    """合成データを生成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='1H')

    returns = np.random.normal(0, 0.02, n_rows)
    price = 100 * np.exp(np.cumsum(returns))

    high = price * (1 + np.random.uniform(0, 0.03, n_rows))
    low = price * (1 - np.random.uniform(0, 0.03, n_rows))
    close = price
    volume = np.random.uniform(1000, 10000, n_rows)

    episode_length = 1000
    episode_ids = np.repeat(np.arange(n_rows // episode_length + 1), episode_length)[:n_rows]

    df = pd.DataFrame({
        'ts': dates.view('int64') // 10**9,
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
        'exchange': 'synthetic',
        'pair': 'BTC/USD',
        'episode_id': episode_ids
    })

    return df


def evaluate_policy(model, df: pd.DataFrame, n_episodes: int = 10) -> Dict[str, float]:
    """ポリシーを評価"""
    def make_env():
        env = HeavyTradingEnv(df)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    episode_rewards = []
    episode_lengths = []
    total_trades = []
    win_rates = []
    profit_factors = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        trades = 0
        wins = 0
        pnl = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if 'trades' in info[0]:
                trades = info[0]['trades']
            if 'pnl' in info[0]:
                pnl = info[0]['pnl']

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_trades.append(trades)

        # win_rate: 仮定
        win_rates.append(0.5)  # placeholder
        profit_factors.append(1.0)  # placeholder

    env.close()

    rewards = np.array(episode_rewards)
    mean_reward = np.mean(rewards)
    win_rate = np.mean(win_rates)
    profit_factor = np.mean(profit_factors)
    trades_per_episode = np.mean(total_trades)

    if len(rewards) > 1:
        sharpe_like = mean_reward / np.std(rewards) if np.std(rewards) > 0 else 0
    else:
        sharpe_like = 0

    return {
        'mean_reward': float(mean_reward),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'sharpe_like': float(sharpe_like),
        'trades_per_episode': float(trades_per_episode)
    }


def permutation_importance(model, df: pd.DataFrame, feature_cols: List[str], n_episodes: int = 10) -> pd.DataFrame:
    """Permutation importance計算"""
    # ベースライン評価
    baseline = evaluate_policy(model, df, n_episodes)

    results = []

    for col in feature_cols:
        # 列をシャッフル（ブロックシャッフルで時系列性維持）
        df_shuffled = df.copy()
        values = df_shuffled[col].to_numpy().copy()
        np.random.shuffle(values)  # 時系列無視でシャッフル
        df_shuffled[col] = values

        # シャッフル後評価
        permuted = evaluate_policy(model, df_shuffled, n_episodes)

        results.append({
            'feature': col,
            'baseline_sharpe': baseline['sharpe_like'],
            'permuted_sharpe': permuted['sharpe_like'],
            'delta_sharpe': permuted['sharpe_like'] - baseline['sharpe_like'],
            'baseline_win_rate': baseline['win_rate'],
            'permuted_win_rate': permuted['win_rate'],
            'delta_win_rate': permuted['win_rate'] - baseline['win_rate'],
            'baseline_pf': baseline['profit_factor'],
            'permuted_pf': permuted['profit_factor'],
            'delta_pf': permuted['profit_factor'] - baseline['profit_factor']
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Permutation importance analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--waves', type=str, default='1,2,3', help='Comma-separated waves')
    parser.add_argument('--episodes', type=int, default=25, help='Number of evaluation episodes')
    parser.add_argument('--n-rows', type=int, default=5000, help='Number of data rows')

    args = parser.parse_args()

    # モデルロード
    model = PPO.load(args.checkpoint)

    # マネージャー初期化
    manager = get_feature_manager()

    # データ生成
    df = generate_synthetic_data(args.n_rows)

    # Wave指定
    waves = [int(w) for w in args.waves.split(',')]
    all_features = []
    for wave in waves:
        all_features.extend(manager.get_enabled_features(wave))

    # 特徴量計算
    df_with_features = manager.compute_features(df, wave=None)

    # 特徴量列取得
    exclude_cols = ['ts', 'exchange', 'pair', 'episode_id']
    feature_cols = [c for c in df_with_features.columns if c not in exclude_cols]

    # Permutation importance
    print(f"Evaluating permutation importance with {args.episodes} episodes...")
    results_df = permutation_importance(model, df_with_features, feature_cols, args.episodes)

    # 出力
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = output_dir / f'perm_importance_{date_str}.csv'
    results_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

    # トップ/ボトム5
    sorted_results = results_df.sort_values('delta_sharpe')
    print("\nTop 5 most important features (by delta_sharpe):")
    for _, row in sorted_results.head(5).iterrows():
        print(f"  {row['feature']}: {row['delta_sharpe']:.6f}")

    print("\nBottom 5 least important features (by delta_sharpe):")
    for _, row in sorted_results.tail(5).iterrows():
        print(f"  {row['feature']}: {row['delta_sharpe']:.6f}")


if __name__ == '__main__':
    main()
