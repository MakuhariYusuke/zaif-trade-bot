# Evaluation and Visualization Script for Trading RL Models
# 取引RLモデルの評価と可視化スクリプト

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from ztb.utils.data_utils import load_csv_data
from ztb.utils.errors import safe_operation

warnings.filterwarnings("ignore")

# ローカルモジュールのインポート
parent_path = str(Path(__file__).parent.parent)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from ztb.trading.environment import HeavyTradingEnv


class EvaluationResult(TypedDict, total=False):
    """Type definition for evaluation results"""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    data_quality_score: float  # Composite score based on NaN rate, correlation, etc.
    evaluation_timestamp: str
    feature_count: int
    model_config: Dict[str, Any]


class TradingEvaluator:
    """取引モデルの評価クラス"""

    def __init__(
        self, model_path: str, data_path: str, config: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__()
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
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard設定
        self.tensorboard_log_dir = Path(
            self.config.get("tensorboard_log", "./tensorboard/")
        )
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=str(self.tensorboard_log_dir / "evaluation")
        )

    def _get_default_config(self) -> dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "results_dir": "./results/",
            "n_eval_episodes": 20,
            "max_steps_per_episode": 10000,
            "render_mode": None,
            "deterministic": True,
            "plot_style": "seaborn",
        }

    def _load_data(self) -> pd.DataFrame:
        """データの読み込み（キャッシュ最適化付き）"""
        # キャッシュチェック
        cache_path = self.data_path.with_suffix(".pkl")
        if (
            cache_path.exists()
            and cache_path.stat().st_mtime > self.data_path.stat().st_mtime
        ):
            print(f"Loading cached data from {cache_path}")
            df = pd.read_pickle(cache_path)
        else:
            if self.data_path.suffix == ".parquet":
                df = pd.read_parquet(self.data_path)
            elif self.data_path.suffix == ".csv":
                df = load_csv_data(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

            # キャッシュ保存
            df.to_pickle(cache_path)
            print(f"Cached data to {cache_path}")

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
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
        }

        env = HeavyTradingEnv(self.df, env_config)
        return env

    def evaluate_model(self) -> Dict[str, Any]:
        """モデルの包括的な評価"""
        return safe_operation(
            logger=None,  # Use default logger
            operation=self._evaluate_model_impl,
            context="model_evaluation",
            default_result={},  # Empty dict on error
        )

    def _evaluate_model_impl(self) -> Dict[str, Any]:
        """Implementation of model evaluation."""
        print("Starting comprehensive model evaluation...")

        # 複数エピソードの評価
        all_rewards = []
        all_positions = []
        all_pnls = []
        all_actions = []
        all_states = []

        for episode in range(self.config["n_eval_episodes"]):
            print(f"Evaluating episode {episode + 1}/{self.config['n_eval_episodes']}")

            episode_data = self._evaluate_single_episode()
            all_rewards.append(episode_data["rewards"])
            all_positions.append(episode_data["positions"])
            all_pnls.append(episode_data["pnls"])
            all_actions.append(episode_data["actions"])
            all_states.append(episode_data["states"])

        # 統計の計算
        stats = self._calculate_statistics(
            all_rewards, all_positions, all_pnls, all_actions
        )

        # 結果の保存
        self._save_evaluation_results(
            stats, all_rewards, all_positions, all_pnls, all_actions
        )

        return stats

    def compare_models(
        self,
        model_paths: List[str],
        model_names: Optional[List[str]] = None,
        dsr_trials: Optional[int] = None,
        bootstrap_resamples: int = 1000,
        bootstrap_block: Optional[int] = None,
        bootstrap_overlap: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """複数のモデルの比較評価"""
        print("Starting model comparison...")

        if model_names is None:
            model_names = [f"Model_{i + 1}" for i in range(len(model_paths))]

        if len(model_names) != len(model_paths):
            raise ValueError("model_names must have same length as model_paths")

        # Calculate DSR trials
        strategies = len(model_paths)
        windows = self.config["n_eval_episodes"]  # Assume windows = episodes for now
        default_dsr_trials = min(1000, strategies * windows)
        dsr_trials = dsr_trials or default_dsr_trials

        # Calculate bootstrap parameters
        n = len(self.df)  # Number of data points
        bootstrap_block = bootstrap_block or max(16, math.ceil(math.sqrt(n)))
        bootstrap_overlap = bootstrap_overlap if bootstrap_overlap is not None else True

        print(
            f"Bootstrap: resamples={bootstrap_resamples}, block={bootstrap_block}, overlap={bootstrap_overlap}"
        )

        results = {}
        for model_path, model_name in zip(model_paths, model_names):
            print(f"\nEvaluating {model_name}...")
            # Create evaluator for this model
            evaluator = TradingEvaluator(model_path, str(self.data_path), self.config)
            stats = evaluator.evaluate_model()
            results[model_name] = stats

        # Save comparison results
        comparison_data = {
            "dsr_trials": dsr_trials,
            "strategies_compared": strategies,
            "independent_windows": windows,
            "models": results,
            "timestamp": datetime.now().isoformat(),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.results_dir / f"model_comparison_{timestamp}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2, default=str)

        # Save metrics.json
        metrics_file = self.results_dir / "metrics.json"
        metrics_data = {
            "dsr_trials": dsr_trials,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_block": bootstrap_block,
            "bootstrap_overlap": bootstrap_overlap,
            "evaluation_timestamp": datetime.now().isoformat(),
            "strategies_compared": strategies,
            "independent_windows": windows,
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        print(f"Comparison results saved to {comparison_file}")
        print(f"Metrics saved to {metrics_file}")
        return comparison_data

    def _evaluate_single_episode(self) -> Dict[str, Any]:
        """単一エピソードの評価"""
        obs, info = self.env.reset()
        done = False

        rewards = []
        positions = []
        pnls = []
        actions = []
        states = []

        step_count = 0
        while not done and step_count < self.config["max_steps_per_episode"]:
            # 行動の予測
            action_value, _ = self.model.predict(
                obs, deterministic=self.config["deterministic"]
            )
            action = cast(int, action_value.item())

            # 環境ステップ
            next_obs, reward, done, _, info = self.env.step(action)

            # データの記録
            rewards.append(reward)
            positions.append(info["position"])
            pnls.append(info.get("pnl", 0))
            actions.append(action)
            states.append(obs.copy())

            obs = next_obs
            step_count += 1

        return {
            "rewards": rewards,
            "positions": positions,
            "pnls": pnls,
            "actions": actions,
            "states": states,
        }

    def _calculate_statistics(
        self,
        all_rewards: List[List[float]],
        all_positions: List[List[float]],
        all_pnls: List[List[float]],
        all_actions: List[List[int]],
    ) -> Dict[str, Any]:
        """統計の計算"""
        # リワード統計
        all_episode_rewards = [sum(episode_rewards) for episode_rewards in all_rewards]
        all_episode_pnls = [sum(episode_pnls) for episode_pnls in all_pnls]

        # ポジション統計
        all_position_changes = []
        for positions in all_positions:
            changes = [
                abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))
            ]
            all_position_changes.extend(changes)

        # 行動統計
        all_episode_actions = [
            action for episode_actions in all_actions for action in episode_actions
        ]

        episode_lengths = [len(r) for r in all_rewards]

        win_rate = (
            np.mean([1 if r > 0 else 0 for r in all_episode_rewards])
            if all_episode_rewards
            else 0.0
        )
        stats = {
            "reward_stats": {
                "mean_total_reward": np.mean(all_episode_rewards),
                "std_total_reward": np.std(all_episode_rewards),
                "min_total_reward": np.min(all_episode_rewards),
                "max_total_reward": np.max(all_episode_rewards),
                "mean_step_reward": np.mean(
                    [r for episode in all_rewards for r in episode]
                ),
                "total_reward_sum": sum(all_episode_rewards),
                "win_rate": win_rate,
                "sharpe_ratio": (
                    np.mean(all_episode_rewards) / np.std(all_episode_rewards)
                    if np.std(all_episode_rewards) > 0
                    else 0.0
                ),
            },
            "pnl_stats": {
                "mean_total_pnl": np.mean(all_episode_pnls),
                "std_total_pnl": np.std(all_episode_pnls),
                "min_total_pnl": np.min(all_episode_pnls),
                "max_total_pnl": np.max(all_episode_pnls),
                "total_pnl_sum": sum(all_episode_pnls),
                "sharpe_ratio": self._calculate_sharpe_ratio(
                    list(map(float, all_episode_pnls))
                ),
                "sortino_ratio": self._calculate_sortino_ratio(
                    list(map(float, all_episode_pnls))
                ),
                "max_drawdown": self._calculate_max_drawdown(
                    list(map(float, all_episode_pnls))
                ),
                "max_drawdown_duration_days": self._calculate_max_drawdown_duration(
                    list(map(float, all_episode_pnls))
                ),
                "calmar_ratio": self._calculate_calmar_ratio(
                    list(map(float, all_episode_pnls))
                ),
            },
            "trading_stats": {
                "total_trades": sum(
                    len([a for a in actions if a != 0]) for actions in all_actions
                ),
                "mean_trades_per_episode": np.mean(
                    [len([a for a in actions if a != 0]) for actions in all_actions]
                ),
                "position_change_rate": (
                    np.mean(all_position_changes) if all_position_changes else 0
                ),
                "position_change_variance": (
                    np.var(all_position_changes) if all_position_changes else 0
                ),
                "hold_ratio": all_episode_actions.count(0) / len(all_episode_actions),
                "buy_ratio": all_episode_actions.count(1) / len(all_episode_actions),
                "sell_ratio": all_episode_actions.count(2) / len(all_episode_actions),
                "min_trades_per_episode": min(
                    len([a for a in actions if a != 0]) for actions in all_actions
                ),
                "max_trades_per_episode": max(
                    len([a for a in actions if a != 0]) for actions in all_actions
                ),
                "hold_ratio_penalty": self._calculate_hold_ratio_penalty(all_actions),
                "profit_factor": self._calculate_profit_factor(
                    list(map(float, all_episode_pnls))
                ),
                "profit_per_trade": (
                    sum(all_episode_pnls)
                    / sum(
                        len([a for a in actions if a != 0]) for actions in all_actions
                    )
                    if sum(
                        len([a for a in actions if a != 0]) for actions in all_actions
                    )
                    > 0
                    else 0.0
                ),
                "win_rate": (
                    sum(1 for pnl in all_episode_pnls if pnl > 0)
                    / len(all_episode_pnls)
                    if all_episode_pnls
                    else 0.0
                ),
            },
            "episode_stats": {
                "num_episodes": len(all_rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "total_steps": sum(episode_lengths),
            },
            "episode_lengths": episode_lengths,
            "episode_rewards": all_episode_rewards,
            "episode_pnls": all_episode_pnls,
        }

        # Calculate data quality score
        stats["data_quality_score"] = cast(
            Any, self._calculate_data_quality_score(stats)
        )

        return stats

    def _calculate_data_quality_score(self, stats: Dict[str, Any]) -> float:
        """Calculate composite data quality score based on various metrics"""
        score_components = []

        # Component 1: Reward stability (lower std is better)
        reward_std = stats["reward_stats"]["std_total_reward"]
        reward_mean = abs(stats["reward_stats"]["mean_total_reward"])
        if reward_mean > 0:
            reward_stability = min(1.0, reward_std / reward_mean)  # Normalized std/mean
            score_components.append(
                1.0 - reward_stability
            )  # Higher stability = higher score
        else:
            score_components.append(0.5)  # Neutral score if mean is zero or negative

        # Component 2: Win rate (higher is better)
        win_rate = stats["trading_stats"]["win_rate"]
        score_components.append(win_rate)

        # Component 3: Sharpe ratio quality (higher is better, but penalize extreme values)
        sharpe = stats["reward_stats"]["sharpe_ratio"]
        sharpe_score = min(1.0, max(0.0, sharpe / 2.0))  # Normalize to 0-1 range
        score_components.append(sharpe_score)

        # Component 4: Trading activity balance (avoid excessive holding)
        hold_penalty = stats["trading_stats"]["hold_ratio_penalty"]
        score_components.append(1.0 - hold_penalty)  # Lower penalty = higher score

        # Component 5: Profit factor (higher is better)
        profit_factor = stats["trading_stats"]["profit_factor"]
        profit_score = min(1.0, profit_factor / 3.0)  # Normalize profit factor
        score_components.append(profit_score)

        # Component 6: Outlier rate (lower is better)
        outlier_rate = self._calculate_outlier_rate(stats)
        outlier_score = max(0.0, 1.0 - outlier_rate * 10)  # Penalize high outlier rates
        score_components.append(outlier_score)

        # Component 7: Distribution quality (skew and kurtosis)
        distribution_score = self._calculate_distribution_quality(stats)
        score_components.append(distribution_score)

        # Component 8: Stability trend (consistency over time)
        stability_score = self._calculate_stability_trend(stats)
        score_components.append(stability_score)

        # Calculate weighted average
        weights = [
            0.15,
            0.2,
            0.15,
            0.1,
            0.15,
            0.1,
            0.1,
            0.05,
        ]  # Weights for each component
        quality_score = sum(w * s for w, s in zip(weights, score_components))

        return cast(float, round(quality_score, 3))

    def _calculate_outlier_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate outlier rate using IQR method"""
        try:
            rewards = stats.get("episode_rewards", [])
            if len(rewards) < 4:
                return 0.0

            # Calculate IQR
            q1 = np.percentile(rewards, 25)
            q3 = np.percentile(rewards, 75)
            iqr = q3 - q1

            if iqr == 0:
                return 0.0

            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Count outliers
            outliers = sum(1 for r in rewards if r < lower_bound or r > upper_bound)
            outlier_rate = outliers / len(rewards)

            return outlier_rate
        except Exception:
            return 0.0

    def _calculate_distribution_quality(self, stats: Dict[str, Any]) -> float:
        """Calculate distribution quality based on skew and kurtosis"""
        try:
            rewards = stats.get("episode_rewards", [])
            if len(rewards) < 3:
                return 0.5

            # Calculate skew and kurtosis
            skew = np.mean(((rewards - np.mean(rewards)) / np.std(rewards)) ** 3)
            kurtosis = (
                np.mean(((rewards - np.mean(rewards)) / np.std(rewards)) ** 4) - 3
            )

            # Penalize high skew (prefer normal distribution)
            skew_penalty = min(1.0, abs(skew) / 2.0)

            # Penalize extreme kurtosis (prefer moderate peakedness)
            kurtosis_penalty = min(1.0, abs(kurtosis) / 4.0)

            # Higher score for more normal-like distributions
            quality_score = 1.0 - (skew_penalty * 0.6 + kurtosis_penalty * 0.4)

            return max(0.0, quality_score)
        except Exception:
            return 0.5

    def _calculate_stability_trend(self, stats: Dict[str, Any]) -> float:
        """Calculate stability trend (consistency over time)"""
        try:
            rewards = stats.get("episode_rewards", [])
            if len(rewards) < 5:
                return 0.5

            # Calculate rolling standard deviation trend
            window_size = min(5, len(rewards) // 2)
            rolling_std = [
                np.std(rewards[i : i + window_size])
                for i in range(len(rewards) - window_size + 1)
            ]

            if len(rolling_std) < 2:
                return 0.5

            # Check if standard deviation is decreasing (improving stability)
            std_trend = np.polyfit(range(len(rolling_std)), rolling_std, 1)[0]

            # Negative trend (decreasing std) is good
            if std_trend < 0:
                stability_score = min(1.0, abs(std_trend) * 100)  # Scale appropriately
            else:
                stability_score = max(
                    0.0, 1.0 - std_trend * 50
                )  # Penalize increasing std

            return stability_score
        except Exception:
            return 0.5

    def _save_evaluation_results(
        self,
        stats: Dict[str, Any],
        all_rewards: List[List[float]],
        all_positions: List[List[float]],
        all_pnls: List[List[float]],
        all_actions: List[List[int]],
    ) -> None:
        """評価結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 統計の保存
        stats_file = self.results_dir / f"evaluation_stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # 生データの保存
        raw_data = {
            "episode_rewards": [sum(r) for r in all_rewards],
            "episode_pnls": [sum(p) for p in all_pnls],
            "all_rewards": all_rewards,
            "all_positions": all_positions,
            "all_pnls": all_pnls,
            "all_actions": all_actions,
        }

        raw_file = self.results_dir / f"evaluation_raw_{timestamp}.json"
        with open(raw_file, "w") as f:
            json.dump(raw_data, f, indent=2, default=str)

        print(f"Evaluation results saved to {self.results_dir}")
        print(f"Stats: {stats_file}")
        print(f"Raw data: {raw_file}")

        # TensorBoardに指標を記録
        self._log_to_tensorboard(stats)

    def _log_to_tensorboard(
        self, stats: Dict[str, Any], timestamp: Optional[str] = None
    ) -> None:
        """TensorBoardに評価指標を記録"""
        try:
            # 報酬統計
            reward_stats = stats["reward_stats"]
            self.writer.add_scalar(
                "Evaluation/Mean_Reward", reward_stats["mean_total_reward"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Mean_Step_Reward", reward_stats["mean_step_reward"], 0
            )

            # PnL統計
            pnl_stats = stats["pnl_stats"]
            self.writer.add_scalar(
                "Evaluation/Mean_PnL", pnl_stats["mean_total_pnl"], 0
            )
            self.writer.add_scalar("Evaluation/PnL_Std", pnl_stats["std_total_pnl"], 0)
            self.writer.add_scalar(
                "Evaluation/Sharpe_Ratio", pnl_stats["sharpe_ratio"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Sortino_Ratio", pnl_stats["sortino_ratio"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Max_Drawdown", pnl_stats["max_drawdown"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Calmar_Ratio", pnl_stats["calmar_ratio"], 0
            )

            # 取引統計
            trading_stats = stats["trading_stats"]
            self.writer.add_scalar(
                "Evaluation/Total_Trades", trading_stats["total_trades"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Mean_Trades_Per_Episode",
                trading_stats["mean_trades_per_episode"],
                0,
            )
            self.writer.add_scalar(
                "Evaluation/Position_Change_Rate",
                trading_stats["position_change_rate"],
                0,
            )
            self.writer.add_scalar(
                "Evaluation/Position_Change_Variance",
                trading_stats["position_change_variance"],
                0,
            )
            self.writer.add_scalar(
                "Evaluation/Hold_Ratio", trading_stats["hold_ratio"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Buy_Ratio", trading_stats["buy_ratio"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Sell_Ratio", trading_stats["sell_ratio"], 0
            )
            self.writer.add_scalar(
                "Evaluation/Profit_Per_Trade", trading_stats["profit_per_trade"], 0
            )

            # エピソード統計
            episode_stats = stats["episode_stats"]
            self.writer.add_scalar(
                "Evaluation/Mean_Episode_Length",
                episode_stats["mean_episode_length"],
                0,
            )
            self.writer.add_scalar(
                "Evaluation/Total_Steps", episode_stats["total_steps"], 0
            )

            self.writer.flush()
            print(
                f"Metrics logged to TensorBoard: {self.tensorboard_log_dir}/evaluation"
            )

        except Exception as e:
            print(f"Warning: Failed to log to TensorBoard: {e}")

    def create_visualizations(self) -> None:
        """評価結果の可視化"""
        print("Creating visualizations...")
        # スタイル設定
        import matplotlib

        mpl_version = matplotlib.__version__
        plot_style = self.config["plot_style"]
        if plot_style == "seaborn":
            # matplotlib 3.6以降は 'seaborn-v0_8' を推奨
            major, minor = map(int, mpl_version.split(".")[:2])
            if major > 3 or (major == 3 and minor >= 6):
                plot_style = "seaborn-v0_8"
        plt.style.use(plot_style)
        sns.set_palette("Set2")
        sns.set_palette("Set2")

        # 評価結果ファイルの読み込み
        stats_files = list(self.results_dir.glob("evaluation_stats_*.json"))
        if not stats_files:
            print("No evaluation stats found. Run evaluation first.")
            return

        latest_stats_file = max(stats_files, key=lambda x: x.stat().st_mtime)
        with open(latest_stats_file, "r") as f:
            stats = json.load(f)

        # 可視化の作成
        self._create_reward_analysis_plot(stats)
        self._create_pnl_analysis_plot(stats)
        self._create_trading_behavior_plot(stats)
        self._create_summary_dashboard(stats)

        print(f"Visualizations saved to {self.results_dir}")

    def _create_reward_analysis_plot(self, stats: Dict[str, Any]) -> None:
        """リワード分析プロット"""
        _, axes = plt.subplots(2, 2, figsize=(15, 12))

        # リワード分布
        episode_rewards = stats.get("episode_rewards", [])
        if episode_rewards:
            axes[0][0].hist(episode_rewards, bins=20, alpha=0.7, edgecolor="black")
            axes[0][0].set_title("Episode Reward Distribution")
            axes[0][0].set_xlabel("Total Reward")
            axes[0][0].set_ylabel("Frequency")
            axes[0][0].axvline(
                np.mean(episode_rewards),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(episode_rewards):.2f}",
            )
            axes[0][0].legend()

        # リワード統計
        reward_stats = stats["reward_stats"]
        labels = list(reward_stats.keys())
        values = list(reward_stats.values())

        axes[0][1].bar(labels, values, alpha=0.7)
        axes[0][1].set_title("Reward Statistics")
        axes[0][1].set_ylabel("Value")
        axes[0][1].tick_params(axis="x", rotation=45)

        # 累積リワード
        if episode_rewards:
            cumulative = np.cumsum(episode_rewards)
            axes[1][0].plot(cumulative, alpha=0.7)
            axes[1][0].set_title("Cumulative Episode Rewards")
            axes[1][0].set_xlabel("Episode")
            axes[1][0].set_ylabel("Cumulative Reward")
            axes[1][0].grid(True)

        # リワード vs エピソード長
        episode_lengths = stats.get("episode_lengths", [])
        if episode_lengths and len(episode_lengths) == len(episode_rewards):
            axes[1][1].scatter(episode_lengths, episode_rewards, alpha=0.6)
            axes[1][1].set_title("Reward vs Episode Length")
            axes[1][1].set_xlabel("Episode Length")
            axes[1][1].set_ylabel("Total Reward")
            axes[1][1].grid(True)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "reward_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_pnl_analysis_plot(self, stats: Dict[str, Any]) -> None:
        """PnL分析プロット"""
        _, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PnL分布
        episode_pnls = stats.get("episode_pnls", [])
        if episode_pnls:
            axes[0][0].hist(episode_pnls, bins=20, alpha=0.7, edgecolor="black")
            axes[0][0].set_title("Episode PnL Distribution")
            axes[0][0].set_xlabel("Total PnL")
            axes[0][0].set_ylabel("Frequency")
            axes[0][0].axvline(
                np.mean(episode_pnls),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(episode_pnls):.4f}",
            )
            axes[0][0].legend()

        # PnL統計
        pnl_stats = stats["pnl_stats"]
        labels = list(pnl_stats.keys())
        values = list(pnl_stats.values())

        axes[0][1].bar(labels, values, alpha=0.7)
        axes[0][1].set_title("PnL Statistics")
        axes[0][1].set_ylabel("Value")
        axes[0][1].tick_params(axis="x", rotation=45)

        # Sharpe比率の表示
        sharpe = pnl_stats.get("sharpe_ratio", 0)
        axes[1][0].text(
            0.5,
            0.5,
            f"Sharpe Ratio: {sharpe:.4f}",
            transform=axes[1][0].transAxes,
            fontsize=16,
            ha="center",
            va="center",
        )
        axes[1][0].set_title("Risk-Adjusted Performance")
        axes[1][0].set_xlim(0, 1)
        axes[1][0].set_ylim(0, 1)
        axes[1][0].axis("off")

        # 累積PnL
        if episode_pnls:
            cumulative = np.cumsum(episode_pnls)
            axes[1][1].plot(cumulative, alpha=0.7)
            axes[1][1].set_title("Cumulative Episode PnL")
            axes[1][1].set_xlabel("Episode")
            axes[1][1].set_ylabel("Cumulative PnL")
            axes[1][1].grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / "pnl_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_trading_behavior_plot(self, stats: Dict[str, Any]) -> None:
        """取引行動分析プロット"""
        _, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 行動分布
        trading_stats = stats["trading_stats"]
        actions = ["Hold", "Buy", "Sell"]
        ratios = [
            trading_stats["hold_ratio"],
            trading_stats["buy_ratio"],
            trading_stats["sell_ratio"],
        ]

        axes[0][0].pie(ratios, labels=actions, autopct="%1.1f%%", startangle=90)
        axes[0][0].set_title("Action Distribution")

        # 取引統計
        trade_labels = ["Total Trades", "Mean Trades/Episode", "Position Change Rate"]
        trade_values = [
            trading_stats["total_trades"],
            trading_stats["mean_trades_per_episode"],
            trading_stats["position_change_rate"],
        ]

        axes[0][1].bar(trade_labels, trade_values, alpha=0.7)
        axes[0][1].set_title("Trading Statistics")
        axes[0][1].set_ylabel("Value")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # エピソード統計
        episode_stats = stats["episode_stats"]
        episode_labels = list(episode_stats.keys())
        episode_values = list(episode_stats.values())

        axes[1, 0].bar(episode_labels, episode_values, alpha=0.7)
        axes[1, 0].set_title("Episode Statistics")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 空のプロット
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Trading Behavior Analysis\nComplete",
            transform=axes[1, 1].transAxes,
            fontsize=16,
            ha="center",
            va="center",
        )

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "trading_behavior.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_summary_dashboard(self, stats: Dict[str, Any]) -> None:
        """サマリーダッシュボード"""
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))

        # 主要指標
        main_metrics = {
            "Mean Reward": stats["reward_stats"]["mean_total_reward"],
            "Mean PnL": stats["pnl_stats"]["mean_total_pnl"],
            "Sharpe Ratio": stats["pnl_stats"]["sharpe_ratio"],
            "Total Trades": stats["trading_stats"]["total_trades"],
            "Win Rate": stats["reward_stats"].get("win_rate", 0),
            "Profit Factor": stats["trading_stats"].get("profit_factor", 0),
            "Total Episodes": stats["episode_stats"]["num_episodes"],
        }

        # メトリクス表示
        total_trades = stats["trading_stats"]["total_trades"]
        for i, (label, value) in enumerate(main_metrics.items()):
            row, col = i // 2, i % 2

            # 取引数100件未満の場合の括弧付き表示
            if label in ["Win Rate", "Profit Factor"] and total_trades < 100:
                display_value = f"{value:.4f} (参考)"
            else:
                display_value = f"{value:.4f}"

            axes[row, col].text(
                0.5,
                0.5,
                f"{label}\n{display_value}",
                transform=axes[row, col].transAxes,
                fontsize=14,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3"),
            )
            axes[row, col].set_title(label, fontsize=12)
            axes[row, col].axis("off")

        # 全体タイトル
        fig.suptitle("Trading RL Model Evaluation Summary", fontsize=16, y=0.98)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "evaluation_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _calculate_sharpe_ratio(
        self, returns: List[float], risk_free_rate: float = 0.0
    ) -> float:
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

    def _calculate_sortino_ratio(
        self, returns: List[float], risk_free_rate: float = 0.0
    ) -> float:
        """Sortino比率の計算（下方偏差のみを使用）"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=float)
        mean_return = float(np.mean(returns_array))

        # 下方偏差の計算（負のリターンのみ）
        downside_returns = returns_array[returns_array < risk_free_rate]
        if len(downside_returns) == 0:
            return float("inf") if mean_return > risk_free_rate else 0.0

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

    def _calculate_calmar_ratio(
        self, returns: List[float], risk_free_rate: float = 0.0
    ) -> float:
        """Calmar比率の計算（年率リターン / 最大ドローダウン）"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=float)
        total_return = float(np.sum(returns_array))
        max_dd = self._calculate_max_drawdown(returns)

        # 安定化処理
        epsilon = 1e-6
        if max_dd < epsilon:
            return float("inf") if total_return > 0 else 0.0

        # 年率換算（簡易版: 総リターンを年率に換算）
        # 実際の運用では適切な期間での計算が必要
        annualized_return = total_return  # 簡易的に総リターンを使用

        return float(annualized_return / max_dd)

    def _calculate_profit_factor(self, all_episode_pnls: List[float]) -> float:
        """Profit Factorの計算（総利益 / 総損失）"""
        if not all_episode_pnls:
            return 0.0

        pnls_array = np.array(all_episode_pnls, dtype=float)
        total_profit = np.sum(pnls_array[pnls_array > 0])
        total_loss = abs(np.sum(pnls_array[pnls_array < 0]))

        if total_loss == 0:
            return float("inf") if total_profit > 0 else 0.0

        return float(total_profit / total_loss)

    def _calculate_hold_ratio_penalty(self, all_actions: List[List[int]]) -> float:
        """ホールド率ペナルティの計算"""
        hold_ratios = []
        penalties = []

        for actions in all_actions:
            if not actions:
                continue  # 空リストはスキップ
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

    def _calculate_max_drawdown_duration(self, returns: List[float]) -> int:
        """最大ドローダウン期間の計算（日数）"""
        if len(returns) < 2:
            return 0

        returns_array = np.array(returns, dtype=float)
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        # ドローダウン期間の計算
        in_drawdown = False
        max_duration = 0
        current_duration = 0

        for i in range(len(drawdown)):
            if drawdown[i] < 0:  # ドローダウン中
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:  # ドローダウン終了
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0

        # 最後のドローダウンが終了していない場合
        if in_drawdown:
            max_duration = max(max_duration, current_duration)

        return max_duration


def main() -> None:
    """メイン関数"""

    parser = argparse.ArgumentParser(
        description="Trading RL Model Evaluation and Visualization"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to evaluation data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "visualize", "compare"],
        default="evaluate",
        help="Operation mode",
    )
    parser.add_argument(
        "--compare-models", nargs="+", help="Paths to models for comparison"
    )
    parser.add_argument("--model-names", nargs="+", help="Names for compared models")
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--dsr-trials",
        type=int,
        default=None,
        help="Number of DSR trials (default: min(1000, strategies * windows))",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--bootstrap-block",
        type=int,
        default=None,
        help="Bootstrap block size (default: max(16, ceil(sqrt(n))))",
    )
    parser.add_argument(
        "--bootstrap-overlap",
        action="store_true",
        default=True,
        help="Bootstrap overlap (default: True)",
    )

    args = parser.parse_args()

    # 設定の更新（デフォルト設定を維持しつつ上書き）
    config = {
        "results_dir": "./results/",
        "n_eval_episodes": 20,
        "max_steps_per_episode": 10000,
        "render_mode": None,
        "deterministic": True,
        "plot_style": "seaborn",
    }
    config.update(
        {
            "n_eval_episodes": args.n_episodes,
            "results_dir": "./results/",
        }
    )

    evaluator = TradingEvaluator(args.model, args.data, config)

    if args.mode == "evaluate":
        stats = evaluator.evaluate_model()
        print("\nEvaluation Summary:")
        print(f"Mean Reward: {stats['reward_stats']['mean_total_reward']:.4f}")
        print(f"Mean PnL: {stats['pnl_stats']['mean_total_pnl']:.4f}")
        print(f"Sharpe Ratio: {stats['pnl_stats']['sharpe_ratio']:.4f}")
        print(f"Total Trades: {stats['trading_stats']['total_trades']}")

    elif args.mode == "visualize":
        evaluator.create_visualizations()

    elif args.mode == "compare":
        if not args.compare_models:
            print("Error: --compare-models required for comparison mode")
            return

        evaluator.compare_models(
            args.compare_models,
            args.model_names,
            args.dsr_trials,
            args.bootstrap_resamples,
            args.bootstrap_block,
            args.bootstrap_overlap,
        )


if __name__ == "__main__":
    main()
