# PPO Training Script for Heavy Trading Environment
# 重特徴量取引環境でのPPO学習

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import psutil
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
import logging
import gzip
from logging.handlers import BufferingHandler
import concurrent.futures
import threading

# ローカルモジュールのインポート
# sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # プロジェクトルートを追加
from .environment import HeavyTradingEnv
from ..utils.perf.cpu_tune import apply_cpu_tuning

# 非同期チェックポイント保存用
_save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_save_lock = threading.Lock()
_keep_last = 5  # 保持世代数

def save_checkpoint_async(model, path_base: str, notifier=None):
    """チェックポイントを非同期で保存（原子的・世代管理付き）"""
    def _job():
        try:
            # tmpに保存
            tmp_path = f"{path_base}.tmp"
            final_path = f"{path_base}.zip"
            
            with _save_lock:
                model.save(tmp_path)
                # 原子的rename
                os.replace(tmp_path, final_path)
            
            logging.info(f"[CKPT] saved: {final_path}")
            
            # 世代管理: 同じprefixの古いファイルを削除
            dir_path = Path(final_path).parent
            prefix = Path(final_path).stem.split('_')[0] if '_' in Path(final_path).stem else Path(final_path).stem
            
            checkpoints = sorted(dir_path.glob(f"{prefix}_*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_ckpt in checkpoints[_keep_last:]:
                old_ckpt.unlink()
                logging.info(f"[CKPT] removed old: {old_ckpt}")
                
        except Exception as e:
            logging.exception(f"[CKPT] save failed: {e}")
            if notifier:
                notifier.send_error_notification("Checkpoint Save Error", f"Failed to save {path_base}: {str(e)}")
    
    _save_executor.submit(_job)

class TensorBoardCallback(BaseCallback):
    """TensorBoard用のコールバック"""

    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        """
        コンストラクタ

        Args:
            eval_freq (int): 評価頻度（ステップ数）
            verbose (int): 詳細ログのレベル
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        """
        各ステップで呼び出されるメソッド。環境の統計情報をTensorBoardに記録します。

        Returns:
            bool: トレーニングを継続する場合はTrue
        """
        if self.n_calls % self.eval_freq == 0 and self.model.env:
            # 環境の統計情報をTensorBoardに記録
            # VecEnvから get_statistics メソッドを直接取得
            stats_list = self.model.env.get_attr('get_statistics')
            if stats_list and callable(stats_list[0]):
                stats = stats_list[0]()
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        self.logger.record(f'env/{key}', value)
        return True


class CheckpointCallback(BaseCallback):
    """チェックポイント保存用のコールバック"""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "checkpoint", verbose: int = 0, notifier=None, session_id=None):
        """
        コンストラクタ

        Args:
            save_freq (int): チェックポイントを保存する頻度（ステップ数）
            save_path (str): チェックポイントを保存するパス
            name_prefix (str): チェックポイントファイル名のプレフィックス
            verbose (int): 詳細ログのレベル
            notifier: 通知用のオプショナルなNotifierオブジェクト
            session_id: トレーニングセッションのID
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.notifier = notifier
        self.session_id = session_id

    def _on_step(self) -> bool:
        """
        各ステップで呼び出されるメソッド。定期的にモデルのチェックポイントを保存します。

        Returns:
            bool: トレーニングを継続する場合はTrue
        """
        try:
            if self.n_calls % self.save_freq == 0:
                checkpoint_path = self.save_path / f"{self.name_prefix}_{self.n_calls}"
                # 非同期保存
                save_checkpoint_async(self.model, str(checkpoint_path), self.notifier)
                
                total_timesteps = getattr(self.model, "_total_timesteps", 1000000)
                progress_percent = (self.n_calls / total_timesteps) * 100
                progress_msg = f"Step {self.n_calls:,} / {self.n_calls:,} ({progress_percent:.1f}%)"

                # INFOログ
                logging.info(progress_msg)

                # Discord通知（10%ごと）
                if int(progress_percent) % 10 == 0 and self.notifier:
                    self.notifier.send_custom_notification(
                        f"📊 Training Progress ({self.session_id})",
                        progress_msg,
                        color=0x00ff00
                    )

                if self.verbose > 0:
                    print(progress_msg)

        except Exception as e:
            logging.error(f"Error in checkpoint callback: {e}")
            if self.notifier:
                self.notifier.send_error_notification("Checkpoint Error", str(e))

        return True


class SafetyCallback(BaseCallback):
    """安全策コールバック - トレード数が0のまま学習を停止"""

    def __init__(self, max_zero_trades=10000, verbose=0):
        """
        コンストラクタ

        Args:
            max_zero_trades (int): トレード数が0のまま許容する最大ステップ数
            verbose (int): 詳細ログレベル
        """
        super().__init__(verbose)
        self.max_zero_trades = max_zero_trades
        self.zero_trade_count = 0

    def _on_step(self) -> bool:
        """
        各ステップでトレード数をチェックし、0のまま続くと学習を停止

        Returns:
            bool: トレーニングを継続する場合はTrue
        """
        try:
            # 環境の統計情報を取得
            if self.model.env:
                stats_list = self.model.env.get_attr('get_statistics')
                if stats_list and callable(stats_list[0]):
                    stats = stats_list[0]()
                    if isinstance(stats, dict):
                        total_trades = stats.get('total_trades', 0)

                        if total_trades == 0:
                            self.zero_trade_count += 1
                            if self.zero_trade_count >= self.max_zero_trades:
                                logging.warning(f"No trades for {self.max_zero_trades} steps, stopping training")
                                return False  # 学習停止
                        else:
                            self.zero_trade_count = 0

        except Exception as e:
            logging.error(f"Error in safety callback: {e}")

        return True


class PPOTrainer:
    """PPOトレーニングマネージャー"""

    def __init__(self, data_path: str, config: Optional[dict] = None, checkpoint_interval: int = 10000, checkpoint_dir: str = 'models/checkpoints'):
        """
        コンストラクタ

        Args:
            data_path (str): トレーニングデータのパス
            config (Optional[dict]): トレーニング設定
            checkpoint_interval (int): チェックポイント保存の間隔（ステップ数）
            checkpoint_dir (str): チェックポイント保存ディレクトリ
        """
        # CPU最適化を最初に適用
        apply_cpu_tuning()
        
        self.data_path = Path(data_path)
        
        # CPU最適化設定
        self._setup_cpu_optimization()
        
        self.config = config or self._get_default_config()
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)

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

    def _setup_cpu_optimization(self) -> None:
        """CPU最適化設定"""
        from ..utils.perf.cpu_tune import auto_config_threads
        
        # 環境変数から設定取得
        num_processes = int(os.environ.get("PARALLEL_PROCESSES", "1"))
        pin_cores_str = os.environ.get("CPU_AFFINITY")
        pin_to_cores = [int(x) for x in pin_cores_str.split(",")] if pin_cores_str else None
        
        # 自動設定決定
        cpu_config = auto_config_threads(num_processes, pin_to_cores)
        
        # 環境変数設定
        for key, value in cpu_config.items():
            if key.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_')):
                os.environ[key] = str(value)
        
        # PyTorch設定
        torch.set_num_threads(cpu_config['torch_threads'])
        torch.backends.mkldnn.enabled = True
        
        # ログ出力
        logging.info(f"CPU: phys={cpu_config['physical_cores']}, log={cpu_config['logical_cores']}, "
                     f"procs={cpu_config['num_processes']}, pin={cpu_config['pin_to_cores']}, "
                     f"torch={cpu_config['torch_threads']}, OMP={cpu_config['OMP_NUM_THREADS']}, "
                     f"MKL={cpu_config['MKL_NUM_THREADS']}, OPENBLAS={cpu_config['OPENBLAS_NUM_THREADS']}")

    def _get_default_config(self) -> dict:
        """
        デフォルトのPPOトレーニング設定を返します。

        Returns:
            dict: デフォルト設定の辞書
        """
        return {
            'total_timesteps': 200000,  # 本番用と同じ値に統一
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
        """
        指定されたパスからトレーニングデータを読み込みます。
        ワイルドカードを含むパスにも対応し、複数のファイルを結合できます。

        Returns:
            pd.DataFrame: 読み込まれたデータフレーム

        Raises:
            FileNotFoundError: データファイルが見つからない場合
            ValueError: サポートされていないファイル形式または有効なデータファイルがない場合
        """
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
        """
        トレーニング用のHeavyTradingEnv環境を作成し、Monitorでラップします。
        設定ファイルから取引手数料を読み込みます。

        Returns:
            Monitor: モニターでラップされた環境オブジェクト
        """
        # rl_config.jsonから手数料設定を読み込み
        # rl_config.jsonから手数料設定を読み込み
        config_path = os.environ.get("RL_CONFIG_PATH")
        if config_path is None:
            # プロジェクトルートの絶対パスをデフォルトに
            config_path = str(Path(__file__).parent.parent.parent.parent / "rl_config.json")
        config_path = Path(config_path)
        transaction_cost = 0.001  # デフォルト
        if config_path.exists():
            with open(config_path, 'r') as f:
                rl_config = json.load(f)
            fee_config = rl_config.get('fee_model', {})
            transaction_cost = fee_config.get('default_fee_rate', 0.001)
        env_config = {
            'reward_scaling': 1.0,
            'transaction_cost': transaction_cost,
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        env = HeavyTradingEnv(self.df, env_config)
        env = Monitor(env, str(self.log_dir / 'monitor.csv'))

        return env

    def train(self, notifier=None, session_id=None) -> PPO:
        """
        設定に基づいてPPOモデルをトレーニングします。

        Args:
            notifier: 通知用のオプショナルなNotifierオブジェクト
            session_id: トレーニングセッションのID

        Returns:
            PPO: トレーニング済みのPPOモデル

        Raises:
            Exception: トレーニング中にエラーが発生した場合
        """
        # I/O最適化: ログバッファリングを設定
        buffer_handler = BufferingHandler(1000)  # 1000メッセージごとにフラッシュ
        buffer_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        buffer_handler.setFormatter(formatter)
        logging.getLogger().addHandler(buffer_handler)
        
        logging.info("Starting PPO training...")
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

        # チェックポイントコールバックの設定
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_interval,
            save_path=str(self.checkpoint_dir),
            name_prefix=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            verbose=1,
            notifier=notifier,
            session_id=session_id
        )

        # 安全策コールバックの設定
        safety_callback = SafetyCallback(max_zero_trades=1000, verbose=1)

        try:
            # トレーニングの実行
            logging.info(f"Training started with total_timesteps: {self.config['total_timesteps']}")
            model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=[eval_callback, tensorboard_callback, checkpoint_callback, safety_callback],
                progress_bar=True,
            )

            # 最終モデルの保存
            model.save(str(self.model_dir / 'final_model'))
            logging.info(f"Model saved to {self.model_dir / 'final_model'}")
            print(f"Model saved to {self.model_dir / 'final_model'}")

            # I/O最適化: バッファをフラッシュ
            buffer_handler.flush()
            
            return model

        except Exception as e:
            logging.exception(f"Training failed: {e}")
            # I/O最適化: エラー時もバッファをフラッシュ
            buffer_handler.flush()
            if notifier:
                notifier.send_error_notification("Training Failed", f"Session {session_id}: {str(e)}")
            raise

    def evaluate(self, model_path: Optional[str] = None, n_episodes: int = 10) -> dict:
        """
        指定されたモデルを評価し、統計情報を返します。

        Args:
            model_path (Optional[str]): 評価するモデルのパス。Noneの場合は最良モデルを使用。
            n_episodes (int): 評価エピソード数

        Returns:
            dict: 評価結果の統計情報（平均報酬など）
        """
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
                # obsがタプルの場合、最初の要素（観測データ）を使用
                predict_obs = obs[0] if isinstance(obs, tuple) else obs
                action, _ = model.predict(predict_obs, deterministic=True)
                obs, reward, done_vec, info = eval_env.step(action)
                done = done_vec[0]
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
        """
        monitor.csvログからトレーニング結果を可視化し、画像を保存します。
        """
        # モニターログの読み込み
        monitor_file = self.log_dir / 'monitor.csv'
        if monitor_file.exists():
            # ヘッダー行数を自動判定
            with open(monitor_file, 'r', encoding='utf-8') as f:
                header_lines = 0
                for line in f:
                    if line.startswith('#'):
                        header_lines += 1
                    else:
                        break
            monitor_df = pd.read_csv(monitor_file, skiprows=header_lines)

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
    """
    Optunaによるハイパーパラメータ最適化
    :param data_path: トレーニングデータのパス
    :param n_trials: 試行回数  
    """

    def objective(trial):
        """
        Optunaの目的関数。指定されたハイパーパラメータでモデルを評価します。

        Args:
            trial: OptunaのTrialオブジェクト

        Returns:
            float: 評価結果の平均報酬
        """
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
    """
    メイン関数
    コマンドライン引数でモードを指定して実行
    例: python main.py --data ./data/train_features.parquet --mode train
    """
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
        # まずハイパーパラメータ最適化を実施
        best_params = optimize_hyperparameters(args.data, args.n_trials)
        # total_timesteps を除外してから config を更新
        best_params.pop('total_timesteps', None)
        trainer.config.update(best_params)
        trainer.config['total_timesteps'] = 200000  # 本番トレーニング
        # 本番トレーニングを実施
        model = trainer.train()
        trainer.evaluate(n_episodes=args.n_episodes)
    elif args.mode == 'optimize':
        best_params = optimize_hyperparameters(args.data, args.n_trials)
        # 最適パラメータでの最終トレーニング
        trainer.config.update(best_params)
        trainer.config['total_timesteps'] = 200000  # 本番トレーニング
        model = trainer.train()

    elif args.mode == 'visualize':
        trainer.visualize_training()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)