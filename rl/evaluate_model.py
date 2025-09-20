# Evaluation and Visualization Script for Trading RL Models
# 取引RLモデルの評価と可視化スクリプト

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))
from envs.heavy_trading_env import HeavyTradingEnv


class TradingEvaluator:
    """取引モデルの評価クラス"""

    def __init__(self, model_path: str, data_path: str, config: dict = None):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.config = config or self._get_default_config()

        # データの読み込み
        self.df = self._load_data()

        # モデルの読み込み
        self.model = self._load_model()

        # 環境の作成
        self.env = self._create_env()

        # 結果保存ディレクトリ
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)

        # TensorBoard設定
        self.tensorboard_log_dir = Path(self.config.get('tensorboard_log', './tensorboard/'))
        self.tensorboard_log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir / 'evaluation'))

    def _get_default_config(self) -> dict:
        """デフォルト設定を取得"""
        return {
            'results_dir': './results/',
            'n_eval_episodes': 20,
            'max_steps_per_episode': 10000,
            'render_mode': None,
            'deterministic': True,
            'plot_style': 'seaborn',
        }

    def _load_data(self) -> pd.DataFrame:
        """データの読み込み"""
        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        print(f"Loaded evaluation data: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _load_model(self) -> PPO:
        """モデルの読み込み"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        model = PPO.load(str(self.model_path))
        print(f"Loaded model: {self.model_path}")
        return model

    def _create_env(self) -> HeavyTradingEnv:
        """評価環境の作成"""
        env_config = {
            'reward_scaling': 1.0,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        }

        env = HeavyTradingEnv(self.df, env_config)
        return env

    def evaluate_model(self) -> Dict:
        """モデルの包括的な評価"""
        print("Starting comprehensive model evaluation...")

        # 複数エピソードの評価
        all_rewards = []
        all_positions = []
        all_pnls = []
        all_actions = []
        all_states = []

        for episode in range(self.config['n_eval_episodes']):
            print(f"Evaluating episode {episode + 1}/{self.config['n_eval_episodes']}")

            episode_data = self._evaluate_single_episode()
            all_rewards.append(episode_data['rewards'])
            all_positions.append(episode_data['positions'])
            all_pnls.append(episode_data['pnls'])
            all_actions.append(episode_data['actions'])
            all_states.append(episode_data['states'])

        # 統計の計算
        stats = self._calculate_statistics(all_rewards, all_positions, all_pnls, all_actions)

        # 結果の保存
        self._save_evaluation_results(stats, all_rewards, all_positions, all_pnls, all_actions)

        return stats

    def _evaluate_single_episode(self) -> Dict:
        """単一エピソードの評価"""
        obs, info = self.env.reset()
        done = False

        rewards = []
        positions = []
        pnls = []
        actions = []
        states = []

        step_count = 0
        while not done and step_count < self.config['max_steps_per_episode']:
            # 行動の予測
            action, _ = self.model.predict(obs, deterministic=self.config['deterministic'])

            # 環境ステップ
            next_obs, reward, done, truncated, info = self.env.step(action)

            # データの記録
            rewards.append(reward)
            positions.append(info['position'])
            pnls.append(info.get('pnl', 0))
            actions.append(action)
            states.append(obs.copy())

            obs = next_obs
            step_count += 1

        return {
            'rewards': rewards,
            'positions': positions,
            'pnls': pnls,
            'actions': actions,
            'states': states,
        }

    def _calculate_statistics(self, all_rewards: List, all_positions: List,
                           all_pnls: List, all_actions: List) -> Dict:
        """統計の計算"""
        # リワード統計
        all_episode_rewards = [sum(episode_rewards) for episode_rewards in all_rewards]
        all_episode_pnls = [sum(episode_pnls) for episode_pnls in all_pnls]

        # ポジション統計
        all_position_changes = []
        for positions in all_positions:
            changes = [abs(positions[i] - positions[i-1]) for i in range(1, len(positions))]
            all_position_changes.extend(changes)

        # 行動統計
        all_episode_actions = [action for episode_actions in all_actions for action in episode_actions]

        stats = {
            'reward_stats': {
                'mean_total_reward': np.mean(all_episode_rewards),
                'std_total_reward': np.std(all_episode_rewards),
                'min_total_reward': np.min(all_episode_rewards),
                'max_total_reward': np.max(all_episode_rewards),
                'mean_step_reward': np.mean([r for episode in all_rewards for r in episode]),
                'total_reward_sum': sum(all_episode_rewards),
            },
            'pnl_stats': {
                'mean_total_pnl': np.mean(all_episode_pnls),
                'std_total_pnl': np.std(all_episode_pnls),
                'min_total_pnl': np.min(all_episode_pnls),
                'max_total_pnl': np.max(all_episode_pnls),
                'total_pnl_sum': sum(all_episode_pnls),
                'sharpe_ratio': self._calculate_sharpe_ratio(list(map(float, all_episode_pnls))),
                'sortino_ratio': self._calculate_sortino_ratio(list(map(float, all_episode_pnls))),
                'max_drawdown': self._calculate_max_drawdown(list(map(float, all_episode_pnls))),
                'calmar_ratio': self._calculate_calmar_ratio(list(map(float, all_episode_pnls))),
            },
            'trading_stats': {
                'total_trades': sum(len([a for a in actions if a != 0]) for actions in all_actions),
                'mean_trades_per_episode': np.mean([len([a for a in actions if a != 0]) for actions in all_actions]),
                'position_change_rate': np.mean(all_position_changes) if all_position_changes else 0,
                'hold_ratio': all_episode_actions.count(0) / len(all_episode_actions),
                'buy_ratio': all_episode_actions.count(1) / len(all_episode_actions),
                'sell_ratio': all_episode_actions.count(2) / len(all_episode_actions),
                'min_trades_per_episode': min(len([a for a in actions if a != 0]) for actions in all_actions),
                'max_trades_per_episode': max(len([a for a in actions if a != 0]) for actions in all_actions),
                'hold_ratio_penalty': self._calculate_hold_ratio_penalty(all_actions),
            },
            'episode_stats': {
                'num_episodes': len(all_rewards),
                'mean_episode_length': np.mean([len(r) for r in all_rewards]),
                'total_steps': sum(len(r) for r in all_rewards),
            }
        }

        return stats

    def _save_evaluation_results(self, stats: Dict, all_rewards: List,
                               all_positions: List, all_pnls: List, all_actions: List) -> None:
        """評価結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 統計の保存
        stats_file = self.results_dir / f'evaluation_stats_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        # 生データの保存
        raw_data = {
            'episode_rewards': [sum(r) for r in all_rewards],
            'episode_pnls': [sum(p) for p in all_pnls],
            'all_rewards': all_rewards,
            'all_positions': all_positions,
            'all_pnls': all_pnls,
            'all_actions': all_actions,
        }

        raw_file = self.results_dir / f'evaluation_raw_{timestamp}.json'
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)

        print(f"Evaluation results saved to {self.results_dir}")
        print(f"Stats: {stats_file}")
        print(f"Raw data: {raw_file}")

        # TensorBoardに指標を記録
        self._log_to_tensorboard(stats, timestamp)

    def _log_to_tensorboard(self, stats: Dict, timestamp: str) -> None:
        """TensorBoardに評価指標を記録"""
        try:
            # 報酬統計
            reward_stats = stats['reward_stats']
            self.writer.add_scalar('Evaluation/Mean_Reward', reward_stats['mean_total_reward'], 0)
            self.writer.add_scalar('Evaluation/Mean_Step_Reward', reward_stats['mean_step_reward'], 0)

            # PnL統計
            pnl_stats = stats['pnl_stats']
            self.writer.add_scalar('Evaluation/Mean_PnL', pnl_stats['mean_total_pnl'], 0)
            self.writer.add_scalar('Evaluation/PnL_Std', pnl_stats['std_total_pnl'], 0)
            self.writer.add_scalar('Evaluation/Sharpe_Ratio', pnl_stats['sharpe_ratio'], 0)
            self.writer.add_scalar('Evaluation/Sortino_Ratio', pnl_stats['sortino_ratio'], 0)
            self.writer.add_scalar('Evaluation/Max_Drawdown', pnl_stats['max_drawdown'], 0)
            self.writer.add_scalar('Evaluation/Calmar_Ratio', pnl_stats['calmar_ratio'], 0)

            # 取引統計
            trading_stats = stats['trading_stats']
            self.writer.add_scalar('Evaluation/Total_Trades', trading_stats['total_trades'], 0)
            self.writer.add_scalar('Evaluation/Mean_Trades_Per_Episode', trading_stats['mean_trades_per_episode'], 0)
            self.writer.add_scalar('Evaluation/Hold_Ratio', trading_stats['hold_ratio'], 0)
            self.writer.add_scalar('Evaluation/Buy_Ratio', trading_stats['buy_ratio'], 0)
            self.writer.add_scalar('Evaluation/Sell_Ratio', trading_stats['sell_ratio'], 0)

            # エピソード統計
            episode_stats = stats['episode_stats']
            self.writer.add_scalar('Evaluation/Mean_Episode_Length', episode_stats['mean_episode_length'], 0)
            self.writer.add_scalar('Evaluation/Total_Steps', episode_stats['total_steps'], 0)

            self.writer.flush()
            print(f"Metrics logged to TensorBoard: {self.tensorboard_log_dir}/evaluation")

        except Exception as e:
            print(f"Warning: Failed to log to TensorBoard: {e}")

    def create_visualizations(self) -> None:
        """評価結果の可視化"""
        print("Creating visualizations...")

        # スタイル設定
        plt.style.use(self.config['plot_style'])
        sns.set_palette("husky")

        # 評価結果ファイルの読み込み
        stats_files = list(self.results_dir.glob('evaluation_stats_*.json'))
        if not stats_files:
            print("No evaluation stats found. Run evaluation first.")
            return

        latest_stats_file = max(stats_files, key=lambda x: x.stat().st_mtime)
        with open(latest_stats_file, 'r') as f:
            stats = json.load(f)

        # 可視化の作成
        self._create_reward_analysis_plot(stats)
        self._create_pnl_analysis_plot(stats)
        self._create_trading_behavior_plot(stats)
        self._create_summary_dashboard(stats)

        print(f"Visualizations saved to {self.results_dir}")

    def _create_reward_analysis_plot(self, stats: Dict) -> None:
        """リワード分析プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # リワード分布
        episode_rewards = stats.get('episode_rewards', [])
        if episode_rewards:
            axes[0, 0].hist(episode_rewards, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Episode Reward Distribution')
            axes[0, 0].set_xlabel('Total Reward')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--',
                              label=f'Mean: {np.mean(episode_rewards):.2f}')
            axes[0, 0].legend()

        # リワード統計
        reward_stats = stats['reward_stats']
        labels = list(reward_stats.keys())
        values = list(reward_stats.values())

        axes[0, 1].bar(labels, values, alpha=0.7)
        axes[0, 1].set_title('Reward Statistics')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 累積リワード
        if episode_rewards:
            cumulative = np.cumsum(episode_rewards)
            axes[1, 0].plot(cumulative, alpha=0.7)
            axes[1, 0].set_title('Cumulative Episode Rewards')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Cumulative Reward')
            axes[1, 0].grid(True)

        # リワード vs エピソード長
        episode_lengths = stats.get('episode_lengths', [])
        if episode_lengths and len(episode_lengths) == len(episode_rewards):
            axes[1, 1].scatter(episode_lengths, episode_rewards, alpha=0.6)
            axes[1, 1].set_title('Reward vs Episode Length')
            axes[1, 1].set_xlabel('Episode Length')
            axes[1, 1].set_ylabel('Total Reward')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pnl_analysis_plot(self, stats: Dict) -> None:
        """PnL分析プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PnL分布
        episode_pnls = stats.get('episode_pnls', [])
        if episode_pnls:
            axes[0, 0].hist(episode_pnls, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Episode PnL Distribution')
            axes[0, 0].set_xlabel('Total PnL')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(episode_pnls), color='red', linestyle='--',
                              label=f'Mean: {np.mean(episode_pnls):.4f}')
            axes[0, 0].legend()

        # PnL統計
        pnl_stats = stats['pnl_stats']
        labels = list(pnl_stats.keys())
        values = list(pnl_stats.values())

        axes[0, 1].bar(labels, values, alpha=0.7)
        axes[0, 1].set_title('PnL Statistics')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Sharpe比率の表示
        sharpe = pnl_stats.get('sharpe_ratio', 0)
        axes[1, 0].text(0.5, 0.5, f'Sharpe Ratio: {sharpe:.4f}',
                       transform=axes[1, 0].transAxes, fontsize=16,
                       ha='center', va='center')
        axes[1, 0].set_title('Risk-Adjusted Performance')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')

        # 累積PnL
        if episode_pnls:
            cumulative = np.cumsum(episode_pnls)
            axes[1, 1].plot(cumulative, alpha=0.7)
            axes[1, 1].set_title('Cumulative Episode PnL')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Cumulative PnL')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'pnl_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_trading_behavior_plot(self, stats: Dict) -> None:
        """取引行動分析プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 行動分布
        trading_stats = stats['trading_stats']
        actions = ['Hold', 'Buy', 'Sell']
        ratios = [trading_stats['hold_ratio'], trading_stats['buy_ratio'], trading_stats['sell_ratio']]

        axes[0, 0].pie(ratios, labels=actions, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Action Distribution')

        # 取引統計
        trade_labels = ['Total Trades', 'Mean Trades/Episode', 'Position Change Rate']
        trade_values = [trading_stats['total_trades'], trading_stats['mean_trades_per_episode'],
                       trading_stats['position_change_rate']]

        axes[0, 1].bar(trade_labels, trade_values, alpha=0.7)
        axes[0, 1].set_title('Trading Statistics')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # エピソード統計
        episode_stats = stats['episode_stats']
        episode_labels = list(episode_stats.keys())
        episode_values = list(episode_stats.values())

        axes[1, 0].bar(episode_labels, episode_values, alpha=0.7)
        axes[1, 0].set_title('Episode Statistics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 空のプロット
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Trading Behavior Analysis\nComplete',
                       transform=axes[1, 1].transAxes, fontsize=16,
                       ha='center', va='center')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'trading_behavior.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_dashboard(self, stats: Dict) -> None:
        """サマリーダッシュボード"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 主要指標
        main_metrics = {
            'Mean Reward': stats['reward_stats']['mean_total_reward'],
            'Mean PnL': stats['pnl_stats']['mean_total_pnl'],
            'Sharpe Ratio': stats['pnl_stats']['sharpe_ratio'],
            'Total Trades': stats['trading_stats']['total_trades'],
            'Win Rate': stats['reward_stats'].get('win_rate', 0),
            'Total Episodes': stats['episode_stats']['num_episodes'],
        }

        # メトリクス表示
        for i, (label, value) in enumerate(main_metrics.items()):
            row, col = i // 2, i % 2
            axes[row, col].text(0.5, 0.5, f'{label}\n{value:.4f}',
                               transform=axes[row, col].transAxes, fontsize=14,
                               ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3"))
            axes[row, col].set_title(label, fontsize=12)
            axes[row, col].axis('off')

        # 全体タイトル
        fig.suptitle('Trading RL Model Evaluation Summary', fontsize=16, y=0.98)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_models(self, model_paths: List[str], model_names: List[str] = None) -> None:
        """複数モデルの比較"""
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(len(model_paths))]

        comparison_results = {}

        for model_path, name in zip(model_paths, model_names):
            print(f"Evaluating {name}...")
            self.model_path = Path(model_path)
            self.model = self._load_model()

            stats = self.evaluate_model()
            comparison_results[name] = stats

        # 比較プロットの作成
        self._create_model_comparison_plot(comparison_results)

        # 比較結果の保存
        comparison_file = self.results_dir / 'model_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        print(f"Model comparison saved to {comparison_file}")

    def _create_model_comparison_plot(self, comparison_results: Dict) -> None:
        """モデル比較プロット"""
        models = list(comparison_results.keys())
        metrics = ['mean_total_reward', 'mean_total_pnl', 'sharpe_ratio', 'total_trades']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2

            values = [comparison_results[model]['reward_stats' if 'reward' in metric else
                                               'pnl_stats' if 'pnl' in metric else
                                               'trading_stats'][metric]
                     for model in models]

            axes[row, col].bar(models, values, alpha=0.7)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_ylabel('Value')
            axes[row, col].tick_params(axis='x', rotation=45)

            # 値の表示
            for j, v in enumerate(values):
                axes[row, col].text(j, v + max(values) * 0.01, f'{v:.3f}',
                                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Sharpe比率の計算（安定化処理付き）"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=float)
        mean_return = float(np.mean(returns_array))
        std_return = float(np.std(returns_array))

        # 安定化処理1: 標準偏差が小さすぎる場合の対策
        min_std = 0.01  # 最小標準偏差を1%に設定
        if std_return < min_std:
            std_return = min_std

        # 安定化処理2: Winsorize処理（極端な値を制限）
        # 99パーセンタイルと1パーセンタイルでクリッピング
        if len(returns_array) > 10:  # サンプル数が十分な場合のみ適用
            p1 = np.percentile(returns_array, 1)
            p99 = np.percentile(returns_array, 99)
            returns_array = np.clip(returns_array, p1, p99)
            std_return = float(np.std(returns_array))
            mean_return = float(np.mean(returns_array))

            # 再チェック
            if std_return < min_std:
                std_return = min_std

        # リスクフリーレートを考慮
        excess_return = mean_return - risk_free_rate

        sharpe = excess_return / std_return

        # 安定化処理3: 極端なSharpe比率を制限
        max_sharpe = 10.0  # 最大Sharpe比率を10に制限
        min_sharpe = -10.0  # 最小Sharpe比率を-10に制限

        return float(np.clip(sharpe, min_sharpe, max_sharpe))

    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Sortino比率の計算（下方偏差のみを使用）"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=float)
        mean_return = float(np.mean(returns_array))

        # 下方偏差の計算（負のリターンのみ）
        downside_returns = returns_array[returns_array < risk_free_rate]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > risk_free_rate else 0.0

        downside_std = float(np.std(downside_returns))

        # 安定化処理
        epsilon = 1e-6
        if downside_std < epsilon:
            downside_std = epsilon

        excess_return = mean_return - risk_free_rate

        return float(excess_return / downside_std)

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """最大ドローダウンの計算"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=float)
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown))

        return abs(max_drawdown)  # 正の値として返す

    def _calculate_calmar_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calmar比率の計算（年率リターン / 最大ドローダウン）"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=float)
        total_return = float(np.sum(returns_array))
        max_dd = self._calculate_max_drawdown(returns)

        # 安定化処理
        epsilon = 1e-6
        if max_dd < epsilon:
            return float('inf') if total_return > 0 else 0.0

        # 年率換算（簡易版: 総リターンを年率に換算）
        # 実際の運用では適切な期間での計算が必要
        annualized_return = total_return  # 簡易的に総リターンを使用

        return float(annualized_return / max_dd)


    def _calculate_hold_ratio_penalty(self, all_actions: List[List[int]]) -> float:
        """ホールド率ペナルティの計算"""
        hold_ratios = []
        penalties = []

        for actions in all_actions:
            hold_ratio = actions.count(0) / len(actions)
            hold_ratios.append(hold_ratio)

            # ホールド率が90%を超える場合のペナルティ
            if hold_ratio > 0.9:
                penalty = (hold_ratio - 0.9) * 10  # 超過分に対して10倍のペナルティ
                penalties.append(penalty)
            else:
                penalties.append(0.0)

        # エピソードごとの平均ペナルティを返す
        return float(np.mean(penalties)) if penalties else 0.0


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Trading RL Model Evaluation and Visualization')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'visualize', 'compare'],
                       default='evaluate', help='Operation mode')
    parser.add_argument('--compare-models', nargs='+', help='Paths to models for comparison')
    parser.add_argument('--model-names', nargs='+', help='Names for compared models')
    parser.add_argument('--n-episodes', type=int, default=20, help='Number of evaluation episodes')

    args = parser.parse_args()

    # 設定の更新
    config = {
        'n_eval_episodes': args.n_episodes,
        'results_dir': './results/',
    }

    evaluator = TradingEvaluator(args.model, args.data, config)

    if args.mode == 'evaluate':
        stats = evaluator.evaluate_model()
        print("\nEvaluation Summary:")
        print(f"Mean Reward: {stats['reward_stats']['mean_total_reward']:.4f}")
        print(f"Mean PnL: {stats['pnl_stats']['mean_total_pnl']:.4f}")
        print(f"Sharpe Ratio: {stats['pnl_stats']['sharpe_ratio']:.4f}")
        print(f"Total Trades: {stats['trading_stats']['total_trades']}")

    elif args.mode == 'visualize':
        evaluator.create_visualizations()

    elif args.mode == 'compare':
        if not args.compare_models:
            print("Error: --compare-models required for comparison mode")
            return

        evaluator.compare_models(args.compare_models, args.model_names)


if __name__ == '__main__':
    main()