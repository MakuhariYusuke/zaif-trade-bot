#!/usr/bin/env python3
"""
SAC v423 Analysis Script

Analyzes training results from SAC v423 initial test runs.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.sac_types import ModelConfig, TrainingMetrics
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv423Analyzer:
    """Analyzer for SAC v423 training results."""

    def __init__(self, results_dir: str = "results/sac_v423"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports")

    def find_latest_training_results(self) -> list[Path]:
        """Find all recent training results."""
        results = []

        # Look for report files
        if self.reports_dir.exists():
            report_files = list(self.reports_dir.glob("training_report_sac_sac_v423*"))
            # Sort by modification time, most recent first
            report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            results.extend(report_files[:5])  # Get up to 5 most recent

        # Also check for training directories
        if self.results_dir.exists():
            training_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
            training_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            results.extend(training_dirs[:2])  # Get up to 2 most recent directories

        return results[:5]  # Return up to 5 total results

    def load_training_metrics(self, result_path: Path) -> Optional[TrainingMetrics]:
        """Load training metrics from the training directory or report file."""
        try:
            # If it's a report file, load the entire report
            if result_path.suffix == ".json" and "training_report" in result_path.name:
                if not result_path.exists():
                    logger.error(f"Report file does not exist: {result_path}")
                    return None

                with open(result_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)

                # Validate report structure
                if not isinstance(report_data, dict):
                    logger.error(
                        f"Invalid report format: expected dict, got {type(report_data)}"
                    )
                    return None

                # Extract metrics from the report structure
                training_stats = report_data.get("training_stats", {})
                if not isinstance(training_stats, dict):
                    logger.warning(
                        "training_stats section missing or invalid in report"
                    )
                    training_stats = {}

                action_dist = training_stats.get("action_distribution", {})
                if not isinstance(action_dist, dict):
                    logger.warning(
                        "action_distribution missing or invalid, using empty dict"
                    )
                    action_dist = {}

                metrics: TrainingMetrics = {
                    "final_episode_reward": float(
                        training_stats.get("final_reward", 0)
                    ),
                    "best_episode_reward": float(
                        training_stats.get("final_reward", 0)
                    ),  # Use final_reward as best
                    "training_time_seconds": float(
                        training_stats.get("training_time", 0)
                    ),
                    "action_distribution": {
                        "HOLD": float(action_dist.get("HOLD", 0)),
                        "BUY": float(action_dist.get("BUY", 0)),
                        "SELL": float(action_dist.get("SELL", 0)),
                    },
                    "total_timesteps": int(training_stats.get("total_timesteps", 0)),
                }
                return metrics

            # Original logic for training directories
            metrics_file = result_path / "training_metrics.json"
            if not metrics_file.exists():
                logger.warning(f"Training metrics file not found: {metrics_file}")
                return None

            with open(metrics_file, "r", encoding="utf-8") as f:
                raw_metrics = json.load(f)

            if not isinstance(raw_metrics, dict):
                logger.error(
                    f"Invalid metrics format: expected dict, got {type(raw_metrics)}"
                )
                return None

            return raw_metrics

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {result_path}: {e}")
            return None
        except FileNotFoundError:
            logger.error(f"File not found: {result_path}")
            return None
        except PermissionError:
            logger.error(f"Permission denied accessing file: {result_path}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error loading training metrics from {result_path}: {e}"
            )
            return None

    def load_model_config(self, result_path: Path) -> Optional[ModelConfig]:
        """Load model configuration from the training directory or report file."""
        # If it's a report file, extract config from report
        if result_path.suffix == ".json" and "training_report" in result_path.name:
            try:
                with open(result_path, "r") as f:
                    report_data = json.load(f)
                return report_data.get("configuration", {})
            except Exception as e:
                logger.warning(
                    f"Failed to load config from report file {result_path}: {e}"
                )
                return None

        # Original logic for training directories
        config_file = result_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
        return None

    def analyze_learning_stability(self, metrics: TrainingMetrics) -> None:
        """Analyze learning stability and convergence."""
        print("\n🧠 学習安定性分析:")

        action_dist = metrics.get("action_distribution", {})
        if not action_dist:
            print("   ⚠️ 行動分布データが利用できません")
            return

        # Calculate action diversity (higher is better)
        total_actions = sum(action_dist.values())
        if total_actions == 0:
            print("   ❌ 有効な行動データがありません")
            return

        # Normalize distribution
        normalized_dist = {k: v / total_actions for k, v in action_dist.items()}

        # Calculate entropy (diversity measure)
        import math

        entropy = -sum(
            p * math.log(p) if p > 0 else 0 for p in normalized_dist.values()
        )

        # Maximum possible entropy for 3 actions
        max_entropy = math.log(3)

        diversity_ratio = entropy / max_entropy if max_entropy > 0 else 0

        print(f"   行動多様性: {diversity_ratio:.3f} (1.0が理想)")
        print(f"   エントロピー: {entropy:.3f}")

        # Stability assessment
        dominant_action = max(action_dist.items(), key=lambda x: x[1])
        dominant_ratio = dominant_action[1] / total_actions

        if diversity_ratio > 0.8:
            print("   ✅ 高い行動多様性（学習が安定）")
        elif diversity_ratio > 0.5:
            print("   ⚠️ 中程度の行動多様性")
        else:
            print("   ❌ 低い行動多様性（学習が収束しすぎの可能性）")

        if dominant_ratio > 0.7:
            print(
                f"   ⚠️ 主要行動 '{dominant_action[0]}' が{dominant_ratio:.1%}を占める"
            )
            print("      行動が偏りすぎている可能性があります")

    def analyze_training_results(self) -> None:
        """Analyze the latest training results with detailed explanations."""
        print("🔍 SAC v423 Training Results Analysis")
        print("=" * 50)
        print("この分析では、SAC v423モデルの最新のトレーニング結果を評価します。")
        print("パフォーマンス指標、行動分布、学習の安定性を確認します。\n")

        result_paths = self.find_latest_training_results()
        if not result_paths:
            print("❌ トレーニング結果が見つかりません")
            print("   検索場所: results/sac_v423/ または reports/")
            print("   トレーニングを実行してから再度お試しください。")
            return

        # Use the most recent result
        result_path = result_paths[0]
        print(f"📁 分析対象: {result_path}")
        print(
            f"   ファイルタイプ: {'レポートファイル' if result_path.suffix == '.json' else 'トレーニングディレクトリ'}"
        )
        if len(result_paths) > 1:
            print(f"   📋 他の利用可能な結果: {len(result_paths) - 1}件")

        # Load data
        metrics = self.load_training_metrics(result_path)
        config = self.load_model_config(result_path)

        if not metrics:
            print("❌ トレーニングメトリクスが見つかりません")
            return

        if not config:
            print("❌ モデル設定が見つかりません")
            return

        # Display basic information
        print("\n📊 トレーニング概要:")
        print(f"   モデル名: {config.get('model_name', 'Unknown')}")
        print(f"   アルゴリズム: {config.get('algorithm', 'Unknown')}")
        print(
            f"   総タイムステップ: {config.get('training', {}).get('total_timesteps', 'Unknown'):,}"
        )
        print(f"   最終エピソード報酬: {metrics.get('final_episode_reward', 'N/A')}")
        print(f"   最高エピソード報酬: {metrics.get('best_episode_reward', 'N/A')}")
        print(f"   トレーニング時間: {metrics.get('training_time_seconds', 'N/A')} 秒")

        # Performance interpretation
        training_time = metrics.get("training_time_seconds", 0)
        total_timesteps = config.get("training", {}).get("total_timesteps", 0)
        if training_time > 0 and total_timesteps > 0:
            steps_per_sec = total_timesteps / training_time
            print(f"   ステップ/秒: {steps_per_sec:.2f}")
            if steps_per_sec > 100:
                print("   ✅ 高速トレーニング（良好なパフォーマンス）")
            elif steps_per_sec > 50:
                print("   ⚠️ 中程度のトレーニング速度")
            else:
                print("   ❌ 低速トレーニング（最適化が必要）")

        # Analyze reward progression
        if "episode_rewards" in metrics:
            rewards = metrics["episode_rewards"]
            print("\n📈 Reward Analysis:")
            print(f"   Episodes: {len(rewards)}")
            print(f"   Average Reward: {sum(rewards)/len(rewards):.2f}")
            print(f"   Min Reward: {min(rewards):.2f}")
            print(f"   Max Reward: {max(rewards):.2f}")
            # Plot reward progression
            self.plot_reward_progression(
                rewards,
                result_path.parent if result_path.suffix == ".json" else result_path,
            )

        # Analyze action distribution if available
        if "action_distribution" in metrics:
            actions = metrics["action_distribution"]
            print("\n🎯 Action Distribution:")
            total_actions = sum(actions.values())
            for action, count in actions.items():
                percentage = (count / total_actions) * 100
                print(f"   {action}: {count} ({percentage:.1f}%)")

        # Analyze learning stability
        self.analyze_learning_stability(metrics)

        print("\n✅ Analysis complete!")
        print(f"📁 Results analyzed from: {result_path}")

    def plot_reward_progression(self, rewards: list, output_dir: Path) -> None:
        """Plot reward progression over episodes."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards, label="Episode Reward", alpha=0.7)
            plt.title("SAC v423 Training - Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            plot_file = output_dir / "reward_progression.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"   📊 Reward plot saved: {plot_file}")

        except Exception as e:
            logger.warning(f"Failed to create reward plot: {e}")

        # Analyze learning stability
        self.analyze_learning_stability(metrics)

        print("\n" + "=" * 50)
        print("🎉 SAC v423分析完了")
        print("改善点:")
        print("- 学習安定性を確認してください")
        print("- 行動分布が偏りすぎていないかチェック")
        print("- トレーニング時間を最適化")
        print("=" * 50)


def main():
    """Main analysis function."""
    analyzer = SACv423Analyzer()
    analyzer.analyze_training_results()


if __name__ == "__main__":
    main()
