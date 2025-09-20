# Hyperparameter Optimization for Trading RL using Optuna
# Optunaを使用した取引RLのハイパーパラメータ最適化

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, Any, Optional

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))
from envs.heavy_trading_env import HeavyTradingEnv


class OptunaCallback(BaseCallback):
    """Optuna最適化用のコールバック"""

    def __init__(self, trial: optuna.Trial, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.cumulative_reward = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        # 定期的に中間結果を報告
        if self.n_calls % self.eval_freq == 0:
            # 現在の報酬を取得
            if hasattr(self.model.env, 'envs'):
                env = self.model.env.envs[0]
            else:
                env = self.model.env

            if hasattr(env, 'reward_history') and env.reward_history:
                current_reward = sum(env.reward_history[-100:])  # 直近100ステップの報酬
                self.trial.report(current_reward, self.n_calls)

            # プルーニングチェック
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True


class HyperparameterOptimizer:
    """ハイパーパラメータ最適化クラス"""

    def __init__(self, data_path: str, config: dict = None):
        self.data_path = Path(data_path)
        self.config = config or self._get_default_config()

        # データの読み込み
        self.df = self._load_data()

        # 最適化結果保存ディレクトリ
        self.opt_dir = Path(self.config['opt_dir'])
        self.opt_dir.mkdir(exist_ok=True)

    def _get_default_config(self) -> dict:
        """デフォルト設定を取得"""
        return {
            'opt_dir': './optimization/',
            'n_trials': 100,
            'timeout': 3600,  # 1時間
            'n_jobs': 1,
            'eval_freq': 2000,
            'n_eval_episodes': 3,
            'optimization_metric': 'mean_reward',  # 'mean_reward', 'sharpe_ratio', 'total_pnl'
            'study_name': 'ppo_trading_optimization',
            'storage': None,  # SQLiteファイルパス（並列最適化用）
        }

    def _load_data(self) -> pd.DataFrame:
        """データの読み込み"""
        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        print(f"Loaded optimization data: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _create_env(self) -> HeavyTradingEnv:
        """環境の作成"""
        env_config = {
            'reward_scaling': 1.0,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        env = HeavyTradingEnv(self.df, env_config)
        return env

    def _sample_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """PPOハイパーパラメータのサンプリング"""
        return {
            # 学習率
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),

            # ネットワークアーキテクチャ
            'net_arch': trial.suggest_categorical('net_arch', [
                [64, 64], [128, 128], [256, 256], [64, 128, 64], [128, 256, 128]
            ]),

            # 活性化関数
            'activation_fn': trial.suggest_categorical('activation_fn', ['tanh', 'relu']),

            # PPO特有パラメータ
            'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'clip_range_vf': trial.suggest_float('clip_range_vf', 0.1, 0.3),

            # 正則化パラメータ
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),

            # 環境パラメータ
            'reward_scaling': trial.suggest_float('reward_scaling', 0.1, 2.0),
            'transaction_cost': trial.suggest_float('transaction_cost', 0.0001, 0.01),
        }

    def _create_ppo_model(self, params: Dict[str, Any], env) -> PPO:
        """PPOモデルの作成"""
        # 活性化関数の変換
        activation_map = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU}
        activation_fn = activation_map[params['activation_fn']]

        # 環境設定の更新
        env_config = {
            'reward_scaling': params['reward_scaling'],
            'transaction_cost': params['transaction_cost'],
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        # 新しい環境の作成
        env = HeavyTradingEnv(self.df, env_config)

        # モデルの作成
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            clip_range=params['clip_range'],
            clip_range_vf=params['clip_range_vf'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            max_grad_norm=params['max_grad_norm'],
            policy_kwargs={
                'net_arch': params['net_arch'],
                'activation_fn': activation_fn,
            },
            verbose=0,
            seed=42,
        )

        return model, env

    def _evaluate_trial(self, model: PPO, env, n_episodes: int = 5) -> Dict[str, float]:
        """トライアルの評価"""
        episode_rewards = []
        episode_pnls = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_pnl = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_pnl += info.get('pnl', 0)

            episode_rewards.append(episode_reward)
            episode_pnls.append(episode_pnl)

        # 統計の計算
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_pnl': np.mean(episode_pnls),
            'std_pnl': np.std(episode_pnls),
            'sharpe_ratio': np.mean(episode_pnls) / (np.std(episode_pnls) + 1e-6),
            'total_reward': sum(episode_rewards),
            'total_pnl': sum(episode_pnls),
        }

        return stats

    def objective(self, trial: optuna.Trial) -> float:
        """Optunaの目的関数"""
        # ハイパーパラメータのサンプリング
        params = self._sample_ppo_params(trial)

        # 環境とモデルの作成
        env = self._create_env()
        model, env = self._create_ppo_model(params, env)

        # トレーニング
        total_timesteps = 30000  # 最適化時は短めに

        # コールバックの設定
        optuna_callback = OptunaCallback(trial, eval_freq=self.config['eval_freq'])

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[optuna_callback],
                progress_bar=False,
            )

            # 評価
            eval_stats = self._evaluate_trial(model, env, self.config['n_eval_episodes'])

            # 最適化指標の選択
            if self.config['optimization_metric'] == 'mean_reward':
                objective_value = eval_stats['mean_reward']
            elif self.config['optimization_metric'] == 'sharpe_ratio':
                objective_value = eval_stats['sharpe_ratio']
            elif self.config['optimization_metric'] == 'total_pnl':
                objective_value = eval_stats['total_pnl']
            else:
                objective_value = eval_stats['mean_reward']

            # トライアル結果の保存
            trial.set_user_attr('eval_stats', eval_stats)
            trial.set_user_attr('params', params)

            return objective_value

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf')

    def optimize(self) -> optuna.Study:
        """ハイパーパラメータ最適化の実行"""
        print("Starting hyperparameter optimization...")

        # Optunaスタディの作成
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(multivariate=True, group=True),
            pruner=MedianPruner(n_warmup_steps=5),
            study_name=self.config['study_name'],
            storage=self.config['storage'],
            load_if_exists=True,
        )

        # 最適化の実行
        study.optimize(
            self.objective,
            n_trials=self.config['n_trials'],
            timeout=self.config['timeout'],
            n_jobs=self.config['n_jobs'],
            show_progress_bar=True,
        )

        # 結果の保存
        self._save_optimization_results(study)

        return study

    def _save_optimization_results(self, study: optuna.Study) -> None:
        """最適化結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 最適パラメータの保存
        best_params_file = self.opt_dir / f'best_params_{timestamp}.json'
        with open(best_params_file, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        # 最適化履歴の保存
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'state': trial.state.name,
            }
            trials_data.append(trial_data)

        history_file = self.opt_dir / f'optimization_history_{timestamp}.json'
        with open(history_file, 'w') as f:
            json.dump(trials_data, f, indent=2)

        print(f"Optimization results saved:")
        print(f"  Best params: {best_params_file}")
        print(f"  History: {history_file}")

    def create_optimization_plots(self, study: optuna.Study) -> None:
        """最適化結果の可視化"""
        try:
            # 最適化履歴
            fig = plot_optimization_history(study)
            fig.write_image(str(self.opt_dir / 'optimization_history.png'))

            # パラメータ重要度
            fig = plot_param_importances(study)
            fig.write_image(str(self.opt_dir / 'param_importances.png'))

            # 並列座標プロット
            fig = plot_parallel_coordinate(study)
            fig.write_image(str(self.opt_dir / 'parallel_coordinate.png'))

            print(f"Optimization plots saved to {self.opt_dir}")

        except Exception as e:
            print(f"Failed to create plots: {e}")

    def get_best_model(self, study: optuna.Study) -> PPO:
        """最適パラメータでモデルを作成"""
        best_params = study.best_params

        # 環境とモデルの作成
        env = self._create_env()
        model, env = self._create_ppo_model(best_params, env)

        return model

    def retrain_best_model(self, study: optuna.Study, total_timesteps: int = 200000) -> PPO:
        """最適パラメータで本番トレーニング"""
        print("Retraining best model with full timesteps...")

        model = self.get_best_model(study)

        # 本番トレーニング
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )

        # モデルの保存
        model_path = self.opt_dir / 'best_model_final'
        model.save(str(model_path))

        print(f"Best model saved to {model_path}")
        return model

    def analyze_optimization_results(self, study: optuna.Study) -> Dict[str, Any]:
        """最適化結果の分析"""
        # トライアルの統計
        completed_trials = [t for t in study.trials if t.state == optuna.TrialState.COMPLETE]

        if not completed_trials:
            return {}

        values = [t.value for t in completed_trials]

        analysis = {
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'optimization_metric': self.config['optimization_metric'],
        }

        # パラメータ分布の分析
        param_distributions = {}
        for param_name in study.best_params.keys():
            param_values = [t.params[param_name] for t in completed_trials if param_name in t.params]
            if param_values:
                param_distributions[param_name] = {
                    'mean': np.mean(param_values),
                    'std': np.std(param_values),
                    'min': np.min(param_values),
                    'max': np.max(param_values),
                }

        analysis['param_distributions'] = param_distributions

        # 分析結果の保存
        analysis_file = self.opt_dir / 'optimization_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        return analysis


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Trading RL')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=3600, help='Optimization timeout in seconds')
    parser.add_argument('--metric', type=str, choices=['mean_reward', 'sharpe_ratio', 'total_pnl'],
                       default='mean_reward', help='Optimization metric')
    parser.add_argument('--retrain', action='store_true', help='Retrain best model with full timesteps')
    parser.add_argument('--full-timesteps', type=int, default=200000, help='Full training timesteps')

    args = parser.parse_args()

    # 設定の更新
    config = {
        'n_trials': args.n_trials,
        'timeout': args.timeout,
        'optimization_metric': args.metric,
        'opt_dir': './optimization/',
    }

    # 最適化の実行
    optimizer = HyperparameterOptimizer(args.data, config)
    study = optimizer.optimize()

    # 結果の表示
    print("\nOptimization Results:")
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")

    # 可視化の作成
    optimizer.create_optimization_plots(study)

    # 分析の実行
    analysis = optimizer.analyze_optimization_results(study)
    print(f"\nOptimization Analysis:")
    print(f"Completed trials: {analysis.get('completed_trials', 0)}")
    print(f"Mean value: {analysis.get('mean_value', 0):.4f}")
    print(f"Std value: {analysis.get('std_value', 0):.4f}")

    # 本番トレーニング（オプション）
    if args.retrain:
        model = optimizer.retrain_best_model(study, args.full_timesteps)
        print(f"Final model trained with {args.full_timesteps} timesteps")


if __name__ == '__main__':
    main()