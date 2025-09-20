# PPO Training Script for Heavy Trading Environment
# 重特徴量取引環境でのPPO学習

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))
from envs.heavy_trading_env import HeavyTradingEnv


class TensorBoardCallback(BaseCallback):
    """TensorBoard用のコールバック"""

    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 環境の統計情報をTensorBoardに記録
            if hasattr(self.model.env, 'envs'):
                env = self.model.env.envs[0]
            else:
                env = self.model.env

            if hasattr(env, 'get_statistics'):
                stats = env.get_statistics()
                for key, value in stats.items():
                    self.logger.record(f'env/{key}', value)

        return True


class PPOTrainer:
    """PPOトレーニングマネージャー"""

    def __init__(self, data_path: str, config: Optional[dict] = None):
        self.data_path = Path(data_path)
        self.config = config or self._get_default_config()

        # ログディレクトリの設定
        self.log_dir = Path(self.config['log_dir'])
        self.log_dir.mkdir(exist_ok=True)

        # モデルの保存ディレクトリ
        self.model_dir = Path(self.config['model_dir'])
        self.model_dir.mkdir(exist_ok=True)

        # データの読み込み
        self.df = self._load_data()

        # 環境の作成
        self.env = self._create_env()
        self.model_dir.mkdir(exist_ok=True)

    def _get_default_config(self) -> dict:
        """デフォルト設定を取得"""
        return {
            'total_timesteps': 200000,
            'eval_freq': 5000,
            'n_eval_episodes': 5,
            'batch_size': 64,
            'n_steps': 2048,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'log_dir': './logs/',
            'model_dir': './models/',
            'tensorboard_log': './tensorboard/',
            'verbose': 1,
            'seed': 42,
        }

    def _load_data(self) -> pd.DataFrame:
        """データの読み込み（ワイルドカード対応）"""
        data_path = Path(self.data_path)

        # ワイルドカードが含まれる場合
        if '*' in str(data_path):
            # globパターンでファイルを検索
            import glob
            file_paths = glob.glob(str(data_path))

            if not file_paths:
                raise FileNotFoundError(f"No files found matching pattern: {data_path}")

            print(f"Found {len(file_paths)} files matching pattern: {data_path}")

            # すべてのファイルを読み込んで結合
            dfs = []
            for file_path in file_paths:
                file_path = Path(file_path)
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    print(f"Skipping unsupported file: {file_path}")
                    continue

                dfs.append(df)
                print(f"Loaded {file_path.name}: {len(df)} rows")

            if not dfs:
                raise ValueError("No valid data files found")

            # データを結合
            df = pd.concat(dfs, ignore_index=True)

            # タイムスタンプでソート
            if 'ts' in df.columns:
                df = df.sort_values('ts').reset_index(drop=True)

        else:
            # 単一ファイルの場合
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            if data_path.suffix == '.parquet':
                df = pd.read_parquet(data_path)
            elif data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        print(f"Total loaded data: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        return df

    def _create_env(self):
        """環境の作成"""
        env_config = {
            'reward_scaling': 1.0,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        env = HeavyTradingEnv(self.df, env_config)
        env = Monitor(env, str(self.log_dir / 'monitor.csv'))

        return env

    def train(self) -> PPO:
        """PPOモデルのトレーニング"""
        print("Starting PPO training...")

        # モデルの作成
        model = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
        )

        # コールバックの設定
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=str(self.model_dir / 'best_model'),
            log_path=str(self.log_dir / 'eval'),
            eval_freq=self.config['eval_freq'],
            n_eval_episodes=self.config['n_eval_episodes'],
            deterministic=True,
            render=False,
        )

        tensorboard_callback = TensorBoardCallback(eval_freq=self.config['eval_freq'])

        # トレーニングの実行
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=[eval_callback, tensorboard_callback],
            progress_bar=True,
        )

        # 最終モデルの保存
        model.save(str(self.model_dir / 'final_model'))
        print(f"Model saved to {self.model_dir / 'final_model'}")

        return model

    def evaluate(self, model_path: Optional[str] = None, n_episodes: int = 10) -> dict:
        """モデルの評価"""
        if model_path is None:
            model_path = str(self.model_dir / 'best_model')

        # モデルの読み込み
        model = PPO.load(model_path)

        # 評価環境の作成
        eval_env = DummyVecEnv([lambda: self._create_env()])

        # 評価の実行
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        # 統計の計算
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'total_episodes': n_episodes,
        }

        # 結果の保存
        results_path = self.log_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Evaluation results saved to {results_path}")
        return stats

    def visualize_training(self) -> None:
        """トレーニング結果の可視化"""
        # モニターログの読み込み
        monitor_file = self.log_dir / 'monitor.csv'
        if monitor_file.exists():
            monitor_df = pd.read_csv(monitor_file, skiprows=1)  # ヘッダーをスキップ

            # プロットの作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # リワードの推移
            axes[0, 0].plot(monitor_df['r'], alpha=0.7)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)

            # エピソード長の推移
            axes[0, 1].plot(monitor_df['l'], alpha=0.7)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True)

            # リワードのヒストグラム
            axes[1, 0].hist(monitor_df['r'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Reward Distribution')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)

            # 累積リワード
            axes[1, 1].plot(np.cumsum(monitor_df['r']), alpha=0.7)
            axes[1, 1].set_title('Cumulative Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Cumulative Reward')
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"Training visualization saved to {self.log_dir / 'training_visualization.png'}")


def optimize_hyperparameters(data_path: str, n_trials: int = 50) -> dict:
    """Optunaによるハイパーパラメータ最適化"""

    def objective(trial):
        # ハイパーパラメータのサンプリング
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
            'total_timesteps': 50000,  # 最適化時は短めに
            'eval_freq': 10000,
            'n_eval_episodes': 3,
            'log_dir': './logs/optuna/',
            'model_dir': './models/optuna/',
            'tensorboard_log': './tensorboard/optuna/',
            'verbose': 0,
            'seed': 42,
        }

        # トレーナーの作成とトレーニング
        trainer = PPOTrainer(data_path, config)
        model = trainer.train()

        # 評価
        eval_stats = trainer.evaluate(n_episodes=5)

        return eval_stats['mean_reward']

    # Optunaスタディの作成
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(),
        pruner=MedianPruner()
    )

    # 最適化の実行
    study.optimize(objective, n_trials=n_trials)

    # 最適なハイパーパラメータの表示
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"Best reward: {study.best_value}")

    return study.best_params


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='PPO Training for Heavy Trading Environment')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (parquet or csv)')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'optimize', 'visualize'],
                       default='train', help='Operation mode')
    parser.add_argument('--model-path', type=str, help='Path to model for evaluation')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of evaluation episodes')

    args = parser.parse_args()

    # トレーナーの作成
    trainer = PPOTrainer(args.data)

    if args.mode == 'train':
        model = trainer.train()
        trainer.evaluate(n_episodes=args.n_episodes)

    elif args.mode == 'evaluate':
        stats = trainer.evaluate(args.model_path, args.n_episodes)
        print("Evaluation Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    elif args.mode == 'optimize':
        best_params = optimize_hyperparameters(args.data, args.n_trials)
        # 最適パラメータでの最終トレーニング
        trainer.config.update(best_params)
        trainer.config['total_timesteps'] = 200000  # 本番トレーニング
        model = trainer.train()

    elif args.mode == 'visualize':
        trainer.visualize_training()


if __name__ == '__main__':
    main()