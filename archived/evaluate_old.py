# Evaluation and Visualization Script for Trading RL Models
# 取引RLモデルの評価と可視化スクリプト

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from ztb.utils.data_utils import load_csv_data
from ztb.utils.errors import safe_operation
from ztb.types.common import ConfigDict

warnings.filterwarnings("ignore")

# ローカルモジュールのインポート
parent_path = str(Path(__file__).parent.parent)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.policies.policy_utils import predict_with_masks


class SingleEpisodeResultDict(TypedDict):
    """Single episode evaluation result."""

    rewards: List[float]
    positions: List[float]
    pnls: List[float]
    actions: List[int]
    states: List[Any]


class EvaluationResult(TypedDict, total=False):
    """Type definition for comprehensive evaluation results.

    Contains all metrics and statistics from model evaluation including
    risk metrics, trading performance, and data quality assessments.
    """

    # Core performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    # Risk and volatility metrics
    volatility: float
    value_at_risk: float
    expected_shortfall: float

    # Data quality assessment
    data_quality_score: float  # Composite score based on NaN rate, correlation, etc.

    # Metadata
    evaluation_timestamp: str
    feature_count: int
    model_config: ConfigDict

    # Detailed statistics
    reward_stats: Dict[str, Any]
    pnl_stats: Dict[str, Any]
    trading_stats: Dict[str, Any]
    episode_stats: Dict[str, Any]  # Episode-level aggregated statistics

    # Action distribution analysis
    action_distribution: Dict[str, int]

    # Performance stability metrics
    consistency_score: float
    robustness_score: float


class TradingEvaluator:
    """取引モデルの評価クラス"""

    writer: Any  # TensorBoard SummaryWriter
    model: Optional[PPO]
    df: Optional[pd.DataFrame]

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
        self.writer = SummaryWriter(  # type: ignore[no-untyped-call]
            log_dir=str(self.tensorboard_log_dir / "evaluation")
        )


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
        env_config = {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
        }

        env = HeavyTradingEnv(self.df, env_config)
        return env

    def evaluate_model(self) -> EvaluationResult:
        """モデルの包括的な評価"""
        return cast(
            EvaluationResult,
            safe_operation(
                logger=None,  # Use default logger
                operation=self._evaluate_model_impl,
                context="model_evaluation",
                default_result={},  # Empty dict on error
            ),
        )

    def _evaluate_model_impl(self) -> EvaluationResult:
        """Implementation of model evaluation with memory optimization."""
        print("Starting comprehensive model evaluation...")
        all_states: Optional[List[Any]] = (
            [] if self.config.get("save_states", False) else None
        )

        for episode in range(self.config["n_eval_episodes"]):
            print(f"Evaluating episode {episode + 1}/{self.config['n_eval_episodes']}")

            episode_data = self._evaluate_single_episode()
            all_rewards.append(episode_data["rewards"])
        stats = self._calculate_statistics(
            all_rewards, all_positions, all_pnls, all_actions
        )

        # 結果の保存
        self._save_evaluation_results(
            stats, all_rewards, all_positions, all_pnls, all_actions
        )

        # メモリ最適化: 大きなデータを解放
        if memory_optimized:
            del all_rewards, all_positions, all_pnls, all_actions
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
        assert self.df is not None, "DataFrame not loaded"
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
            ]
            all_position_changes.extend(changes)

        # 行動統計（メモリ最適化: ジェネレータ使用）
        def get_all_actions() -> Generator[Any, None, None]:
            for episode_actions in all_actions:
                for action in episode_actions:
                    yield action

        all_episode_actions = (
            list(get_all_actions()) if not memory_optimized else list(get_all_actions())
        )

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

        return cast(EvaluationResult, stats)

                stability_score = max(
                    0.0, 1.0 - std_trend * 50
                )  # Penalize increasing std

            return stability_score
        except Exception:
            return 0.5

    def _save_evaluation_results(
        self,
        stats: EvaluationResult,
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
        self, stats: EvaluationResult, timestamp: Optional[str] = None
    ) -> None:
        """TensorBoardに評価指標を記録"""
        try:
            # 報酬統計
            reward_stats = stats.get("reward_stats", {})
            if reward_stats:
                self.writer.add_scalar(
                    "Evaluation/Mean_Reward",
                    reward_stats.get("mean_total_reward", 0),
                    0,
                )
                self.writer.add_scalar(
                    "Evaluation/Mean_Step_Reward",
                    reward_stats.get("mean_step_reward", 0),
                    0,
                )

            # PnL統計
            pnl_stats = stats.get("pnl_stats", {})
            if pnl_stats:
                self.writer.add_scalar(
                    "Evaluation/Mean_PnL", pnl_stats.get("mean_total_pnl", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/PnL_Std", pnl_stats.get("std_total_pnl", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Sharpe_Ratio", pnl_stats.get("sharpe_ratio", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Sortino_Ratio", pnl_stats.get("sortino_ratio", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Max_Drawdown", pnl_stats.get("max_drawdown", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Calmar_Ratio", pnl_stats.get("calmar_ratio", 0), 0
                )

            # 取引統計
            trading_stats = stats.get("trading_stats", {})
            if trading_stats:
                self.writer.add_scalar(
                    "Evaluation/Total_Trades", trading_stats.get("total_trades", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Mean_Trades_Per_Episode",
                    trading_stats.get("mean_trades_per_episode", 0),
                    0,
                )
                self.writer.add_scalar(
                    "Evaluation/Position_Change_Rate",
                    trading_stats.get("position_change_rate", 0),
                    0,
                )
                self.writer.add_scalar(
                    "Evaluation/Position_Change_Variance",
                    trading_stats["position_change_variance"],
                    0,
                )
                self.writer.add_scalar(
                    "Evaluation/Hold_Ratio", trading_stats.get("hold_ratio", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Buy_Ratio", trading_stats.get("buy_ratio", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Sell_Ratio", trading_stats.get("sell_ratio", 0), 0
                )
                self.writer.add_scalar(
                    "Evaluation/Profit_Per_Trade",
                    trading_stats.get("profit_per_trade", 0),
                    0,
                )

            # エピソード統計
            episode_stats = stats.get("episode_stats", {})
            if episode_stats:
                self.writer.add_scalar(
                    "Evaluation/Mean_Episode_Length",
                    episode_stats.get("mean_episode_length", 0),
                    0,
                )
            self.writer.add_scalar(
                "Evaluation/Total_Steps", episode_stats["total_steps"], 0
            )

            self.writer.flush()
            print(
                f"Metrics logged to TensorBoard: {self.tensorboard_log_dir}/evaluation"
            )

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
        except Exception as e:
            print(f"Error during TradingEvaluator cleanup: {e}")
            import traceback

            print(f"Cleanup traceback: {traceback.format_exc()}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup even if close() wasn't called explicitly."""
        self.close()


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
        reward_stats = stats.get("reward_stats", {})
        pnl_stats = stats.get("pnl_stats", {})
        trading_stats = stats.get("trading_stats", {})
        print(f"Mean Reward: {reward_stats.get('mean_total_reward', 0):.4f}")
        print(f"Mean PnL: {pnl_stats.get('mean_total_pnl', 0):.4f}")
        print(f"Sharpe Ratio: {pnl_stats.get('sharpe_ratio', 0):.4f}")
        print(f"Total Trades: {trading_stats.get('total_trades', 0)}")

        evaluator.close()

    elif args.mode == "visualize":
        evaluator.create_visualizations()
        evaluator.close()

    elif args.mode == "compare":
        if not args.compare_models:
            print("Error: --compare-models required for comparison mode")
            evaluator.close()
            return

        evaluator.compare_models(
            args.compare_models,
            args.model_names,
            args.dsr_trials,
            args.bootstrap_resamples,
            args.bootstrap_block,
            args.bootstrap_overlap,
        )
        evaluator.close()


if __name__ == "__main__":
    main()
