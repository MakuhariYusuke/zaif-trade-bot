#!/usr/bin/env python3
"""
Trade Analysis and Action Distribution Visualization

This script analyzes trading logs and visualizes action distributions
to understand model behavior and improvements.

Features:
- Action distribution analysis (BUY/SELL/HOLD)
- Trade sequence analysis
- HOLD bias improvement tracking
- Action transition matrices
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment import HeavyTradingEnv  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


@dataclass
class TradeAnalysisResult:
    """Results of trade analysis."""

    model_name: str
    total_actions: int
    action_counts: Dict[str, int]
    action_percentages: Dict[str, float]
    hold_streaks: List[int]
    avg_hold_streak: float
    max_hold_streak: int
    transition_matrix: Dict[Tuple[str, str], int]
    action_sequences: List[List[str]]


class TradeAnalyzer:
    """Analyzer for trading actions and behavior."""

    def __init__(
        self,
        data_path: Path,
        output_dir: Path = Path("trade_analysis")
    ):
        """
        Initialize trade analyzer.

        Args:
            data_path: Path to evaluation data
            output_dir: Output directory for results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Load evaluation data
        self.eval_data = self._load_data()

        LOGGER.info(f"Initialized trade analyzer with {len(self.eval_data)} data points")

    def _load_data(self) -> pd.DataFrame:
        """Load evaluation data."""
        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")

        LOGGER.info(f"Loaded {len(df)} rows from {self.data_path}")
        return df

    def analyze_model_actions(
        self,
        model_path: Path,
        model_name: str,
        n_episodes: int = 5
    ) -> TradeAnalysisResult:
        """
        Analyze actions taken by a model.

        Args:
            model_path: Path to the model
            model_name: Name for the model
            n_episodes: Number of episodes to analyze

        Returns:
            TradeAnalysisResult with action analysis
        """
        LOGGER.info(f"Analyzing actions for model: {model_name}")

        # Load model
        model = PPO.load(str(model_path))

        # Create environment
        env_config = {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.02,
            "feature_set": "full",
        }

        all_actions = []
        action_sequences = []

        for episode in range(n_episodes):
            env = HeavyTradingEnv(
                df=self.eval_data,
                config=env_config,
                streaming_pipeline=None,
                stream_batch_size=1000,
                max_features=100
            )

            obs = env.reset()
            done = False
            episode_actions = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action.item()) if hasattr(action, 'item') else int(action)
                action_str = self._action_to_string(action_int)

                obs, reward, terminated, truncated, _ = env.step(action_int)
                done = terminated or truncated

                episode_actions.append(action_str)
                all_actions.append(action_str)

            action_sequences.append(episode_actions)

        # Analyze actions
        result = self._analyze_action_patterns(all_actions, action_sequences, model_name)

        return result

    def _action_to_string(self, action: int) -> str:
        """Convert action index to string."""
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_map.get(action, "UNKNOWN")

    def _analyze_action_patterns(
        self,
        all_actions: List[str],
        action_sequences: List[List[str]],
        model_name: str
    ) -> TradeAnalysisResult:
        """Analyze action patterns and create result."""
        # Basic action counts
        action_counts = Counter(all_actions)
        total_actions = len(all_actions)
        action_percentages = {
            action: count / total_actions
            for action, count in action_counts.items()
        }

        # HOLD streak analysis
        hold_streaks = []
        current_streak = 0

        for action in all_actions:
            if action == "HOLD":
                current_streak += 1
            else:
                if current_streak > 0:
                    hold_streaks.append(current_streak)
                    current_streak = 0

        if current_streak > 0:
            hold_streaks.append(current_streak)

        avg_hold_streak = float(np.mean(hold_streaks)) if hold_streaks else 0.0
        max_hold_streak = max(hold_streaks) if hold_streaks else 0

        # Transition matrix
        transition_matrix = defaultdict(int)
        for sequence in action_sequences:
            for i in range(len(sequence) - 1):
                transition = (sequence[i], sequence[i + 1])
                transition_matrix[transition] += 1

        return TradeAnalysisResult(
            model_name=model_name,
            total_actions=total_actions,
            action_counts=dict(action_counts),
            action_percentages=action_percentages,
            hold_streaks=hold_streaks,
            avg_hold_streak=avg_hold_streak,
            max_hold_streak=max_hold_streak,
            transition_matrix=dict(transition_matrix),
            action_sequences=action_sequences
        )

    def compare_models(
        self,
        model_configs: List[Dict[str, Any]],
        n_episodes: int = 5
    ) -> List[TradeAnalysisResult]:
        """
        Compare action patterns across multiple models.

        Args:
            model_configs: List of model configurations with 'path' and 'name'
            n_episodes: Number of episodes per model

        Returns:
            List of analysis results
        """
        results = []

        for config in model_configs:
            model_path = Path(config["path"])
            model_name = config["name"]

            result = self.analyze_model_actions(
                model_path, model_name, n_episodes
            )
            results.append(result)

        return results

    def generate_analysis_report(
        self,
        results: List[TradeAnalysisResult],
        output_prefix: str = "trade_analysis"
    ) -> Path:
        """
        Generate comprehensive analysis report.

        Args:
            results: Analysis results to compare
            output_prefix: Output file prefix

        Returns:
            Path to generated report
        """
        LOGGER.info("Generating trade analysis report")

        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append({
                "Model": result.model_name,
                "Total Actions": result.total_actions,
                "HOLD %": f"{result.action_percentages.get('HOLD', 0):.1%}",
                "BUY %": f"{result.action_percentages.get('BUY', 0):.1%}",
                "SELL %": f"{result.action_percentages.get('SELL', 0):.1%}",
                "Avg HOLD Streak": f"{result.avg_hold_streak:.1f}",
                "Max HOLD Streak": result.max_hold_streak,
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_path = self.output_dir / f"{output_prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Generate visualizations
        self._generate_action_distribution_plot(results, output_prefix)
        self._generate_hold_streak_analysis(results, output_prefix)
        self._generate_transition_heatmap(results, output_prefix)

        # Save detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "model_name": result.model_name,
                "total_actions": result.total_actions,
                "action_counts": result.action_counts,
                "action_percentages": result.action_percentages,
                "hold_streaks": result.hold_streaks,
                "avg_hold_streak": result.avg_hold_streak,
                "max_hold_streak": result.max_hold_streak,
                "transition_matrix": result.transition_matrix,
            })

        detailed_path = self.output_dir / f"{output_prefix}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        LOGGER.info(f"Analysis report generated: {summary_path}")
        return summary_path

    def _generate_action_distribution_plot(
        self,
        results: List[TradeAnalysisResult],
        prefix: str
    ) -> None:
        """Generate action distribution comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 6))

        model_names = [r.model_name for r in results]
        hold_pct = [r.action_percentages.get('HOLD', 0) for r in results]
        buy_pct = [r.action_percentages.get('BUY', 0) for r in results]
        sell_pct = [r.action_percentages.get('SELL', 0) for r in results]

        x = np.arange(len(model_names))
        width = 0.25

        ax.bar(x - width, hold_pct, width, label='HOLD', alpha=0.8, color='blue')
        ax.bar(x, buy_pct, width, label='BUY', alpha=0.8, color='green')
        ax.bar(x + width, sell_pct, width, label='SELL', alpha=0.8, color='red')

        ax.set_title("Action Distribution Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Action Percentage")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add percentage labels
        for i, (h, b, s) in enumerate(zip(hold_pct, buy_pct, sell_pct)):
            ax.text(i - width, h + 0.01, f'{h:.1%}', ha='center', va='bottom', fontsize=8)
            ax.text(i, b + 0.01, f'{b:.1%}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, s + 0.01, f'{s:.1%}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plot_path = self.output_dir / f"{prefix}_action_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"Action distribution plot saved: {plot_path}")

    def _generate_hold_streak_analysis(
        self,
        results: List[TradeAnalysisResult],
        prefix: str
    ) -> None:
        """Generate HOLD streak analysis plot."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Average HOLD streak comparison
        model_names = [r.model_name for r in results]
        avg_streaks = [r.avg_hold_streak for r in results]
        max_streaks = [r.max_hold_streak for r in results]

        x = np.arange(len(model_names))
        width = 0.35

        axes[0].bar(x - width/2, avg_streaks, width, label='Avg Streak', alpha=0.7, color='skyblue')
        axes[0].bar(x + width/2, max_streaks, width, label='Max Streak', alpha=0.7, color='orange')

        axes[0].set_title("HOLD Streak Analysis")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("Streak Length")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # HOLD streak distribution (first model only for simplicity)
        if results and results[0].hold_streaks:
            axes[1].hist(results[0].hold_streaks, bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_title(f"HOLD Streak Distribution ({results[0].model_name})")
            axes[1].set_xlabel("Streak Length")
            axes[1].set_ylabel("Frequency")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{prefix}_hold_streaks.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"HOLD streak analysis plot saved: {plot_path}")

    def _generate_transition_heatmap(
        self,
        results: List[TradeAnalysisResult],
        prefix: str
    ) -> None:
        """Generate action transition heatmap."""
        if not results:
            return

        # Use first result for transition matrix
        result = results[0]
        actions = ['HOLD', 'BUY', 'SELL']

        # Create transition matrix
        transition_data = np.zeros((len(actions), len(actions)))

        for (from_action, to_action), count in result.transition_matrix.items():
            if from_action in actions and to_action in actions:
                i = actions.index(from_action)
                j = actions.index(to_action)
                transition_data[i, j] = count

        # Normalize by row sums
        row_sums = transition_data.sum(axis=1, keepdims=True)
        transition_data = np.divide(transition_data, row_sums,
                                  out=np.zeros_like(transition_data),
                                  where=row_sums != 0)

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            transition_data,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=actions,
            yticklabels=actions,
            cbar_kws={'label': 'Transition Probability'}
        )

        plt.title(f"Action Transition Matrix ({result.model_name})")
        plt.xlabel("To Action")
        plt.ylabel("From Action")

        plt.tight_layout()
        plot_path = self.output_dir / f"{prefix}_transitions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"Transition heatmap saved: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Trade Analysis and Action Distribution Visualization"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to models to analyze (format: name:path)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="trade_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes per model"
    )

    args = parser.parse_args()

    # Parse model configurations
    model_configs = []
    for model_spec in args.models:
        if ":" in model_spec:
            name, path = model_spec.split(":", 1)
        else:
            name = Path(model_spec).stem
            path = model_spec

        model_configs.append({"name": name, "path": path})

    # Initialize analyzer
    analyzer = TradeAnalyzer(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir)
    )

    # Analyze models
    results = analyzer.compare_models(model_configs, n_episodes=args.episodes)

    # Generate report
    analyzer.generate_analysis_report(results)

    LOGGER.info("Trade analysis completed!")


if __name__ == "__main__":
    main()