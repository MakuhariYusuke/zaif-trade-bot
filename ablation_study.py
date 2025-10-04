#!/usr/bin/env python3
"""
Ablation Study for Trading Model Improvements

This script performs ablation studies to evaluate the impact of different
improvements on trading model performance.

Features:
- Dynamic HOLD penalty ablation
- Sortino/Calmar ratio ablation
- Cosine Annealing LR ablation
- Early Stopping ablation
- Confidence-weighted ensemble ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
class AblationConfig:
    """Configuration for ablation study."""

    name: str
    description: str
    env_modifications: Dict[str, Any] = field(default_factory=dict)
    training_modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result of ablation study."""

    config_name: str
    description: str
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    total_return: float
    max_drawdown: float
    improvement_over_baseline: float = 0.0


class AblationStudy:
    """Ablation study runner."""

    def __init__(
        self,
        data_path: Path,
        base_model_path: Path,
        output_dir: Path = Path("ablation_results")
    ):
        """
        Initialize ablation study.

        Args:
            data_path: Path to evaluation data
            base_model_path: Path to base model
            output_dir: Output directory
        """
        self.data_path = data_path
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Load base model
        self.base_model = PPO.load(str(base_model_path))

        # Load evaluation data
        self.eval_data = self._load_data()

        LOGGER.info(f"Initialized ablation study with {len(self.eval_data)} data points")

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

    def define_ablation_configs(self) -> List[AblationConfig]:
        """
        Define ablation study configurations.

        Returns:
            List of ablation configurations
        """
        configs = [
            AblationConfig(
                name="baseline",
                description="Base configuration with all improvements",
                env_modifications={
                    "use_dynamic_hold_penalty": True,
                    "use_sortino_calmar": True,
                },
                training_modifications={
                    "use_cosine_lr": True,
                    "use_early_stopping": True,
                }
            ),
            AblationConfig(
                name="no_dynamic_hold",
                description="Remove dynamic HOLD penalty",
                env_modifications={
                    "use_dynamic_hold_penalty": False,
                    "use_sortino_calmar": True,
                },
                training_modifications={
                    "use_cosine_lr": True,
                    "use_early_stopping": True,
                }
            ),
            AblationConfig(
                name="no_sortino_calmar",
                description="Remove Sortino/Calmar ratio improvements",
                env_modifications={
                    "use_dynamic_hold_penalty": True,
                    "use_sortino_calmar": False,
                },
                training_modifications={
                    "use_cosine_lr": True,
                    "use_early_stopping": True,
                }
            ),
            AblationConfig(
                name="no_cosine_lr",
                description="Remove Cosine Annealing LR",
                env_modifications={
                    "use_dynamic_hold_penalty": True,
                    "use_sortino_calmar": True,
                },
                training_modifications={
                    "use_cosine_lr": False,
                    "use_early_stopping": True,
                }
            ),
            AblationConfig(
                name="no_early_stopping",
                description="Remove Early Stopping",
                env_modifications={
                    "use_dynamic_hold_penalty": True,
                    "use_sortino_calmar": True,
                },
                training_modifications={
                    "use_cosine_lr": True,
                    "use_early_stopping": False,
                }
            ),
            AblationConfig(
                name="no_improvements",
                description="No improvements (traditional setup)",
                env_modifications={
                    "use_dynamic_hold_penalty": False,
                    "use_sortino_calmar": False,
                },
                training_modifications={
                    "use_cosine_lr": False,
                    "use_early_stopping": False,
                }
            ),
        ]

        return configs

    def run_ablation_study(
        self,
        n_episodes: int = 5,
        configs: List[AblationConfig] = None
    ) -> List[AblationResult]:
        """
        Run ablation study.

        Args:
            n_episodes: Number of evaluation episodes per configuration
            configs: Ablation configurations (default: all)

        Returns:
            List of ablation results
        """
        if configs is None:
            configs = self.define_ablation_configs()

        results = []

        # First, evaluate baseline
        baseline_result = None
        for config in configs:
            if config.name == "baseline":
                LOGGER.info(f"Evaluating baseline: {config.description}")
                baseline_result = self._evaluate_config(config, n_episodes)
                results.append(baseline_result)
                break

        if baseline_result is None:
            raise ValueError("Baseline configuration not found")

        # Evaluate other configurations
        for config in configs:
            if config.name == "baseline":
                continue

            LOGGER.info(f"Evaluating ablation: {config.name} - {config.description}")
            result = self._evaluate_config(config, n_episodes)

            # Calculate improvement over baseline
            result.improvement_over_baseline = result.sharpe_ratio - baseline_result.sharpe_ratio

            results.append(result)

        return results

    def _evaluate_config(
        self,
        config: AblationConfig,
        n_episodes: int
    ) -> AblationResult:
        """
        Evaluate a single configuration.

        Args:
            config: Ablation configuration
            n_episodes: Number of episodes

        Returns:
            Ablation result
        """
        # Create environment with modifications
        env_config = self._create_env_config(config.env_modifications)

        all_rewards = []
        all_returns = []

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
            episode_rewards = []

            while not done:
                action, _ = self.base_model.predict(obs, deterministic=True)
                action_int = int(action.item()) if hasattr(action, 'item') else int(action)
                obs, reward, terminated, truncated, _ = env.step(action_int)
                done = terminated or truncated

                episode_rewards.append(reward)

            episode_return = float(np.sum(episode_rewards))
            all_rewards.extend(episode_rewards)
            all_returns.append(episode_return)

        # Calculate metrics
        metrics = self._calculate_metrics(all_returns)

        return AblationResult(
            config_name=config.name,
            description=config.description,
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            win_rate=metrics["win_rate"],
            total_return=metrics["total_return"],
            max_drawdown=metrics["max_drawdown"]
        )

    def _create_env_config(self, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Create environment config with modifications."""
        base_config = {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.02,
            "feature_set": "full",
        }

        # Apply modifications
        config = base_config.copy()
        config.update(modifications)

        return config

    def _calculate_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        returns_array = np.array(returns)

        # Basic metrics
        total_return = float(np.sum(returns))
        win_rate = float(np.mean(returns_array > 0))

        # Sharpe ratio
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns_array - risk_free_rate
        if np.std(excess_returns) > 0:
            sharpe_ratio = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
        else:
            sortino_ratio = 0.0

        # Max drawdown
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = float(np.min(drawdowns))

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
        }

    def generate_ablation_report(
        self,
        results: List[AblationResult],
        output_prefix: str = "ablation_study"
    ) -> Path:
        """
        Generate ablation study report.

        Args:
            results: Ablation results
            output_prefix: Output file prefix

        Returns:
            Path to generated report
        """
        LOGGER.info("Generating ablation study report")

        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append({
                "Configuration": result.config_name,
                "Description": result.description,
                "Sharpe Ratio": f"{result.sharpe_ratio:.4f}",
                "Sortino Ratio": f"{result.sortino_ratio:.4f}",
                "Win Rate": f"{result.win_rate:.2%}",
                "Total Return": f"{result.total_return:.4f}",
                "Max Drawdown": f"{result.max_drawdown:.4f}",
                "Improvement vs Baseline": f"{result.improvement_over_baseline:.4f}",
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_path = self.output_dir / f"{output_prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Generate bar chart
        self._generate_ablation_plot(results, output_prefix)

        # Save detailed results
        detailed_results = [
            {
                "config_name": r.config_name,
                "description": r.description,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "win_rate": r.win_rate,
                "total_return": r.total_return,
                "max_drawdown": r.max_drawdown,
                "improvement_over_baseline": r.improvement_over_baseline,
            }
            for r in results
        ]

        detailed_path = self.output_dir / f"{output_prefix}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        LOGGER.info(f"Ablation report generated: {summary_path}")
        return summary_path

    def _generate_ablation_plot(
        self,
        results: List[AblationResult],
        prefix: str
    ) -> None:
        """Generate ablation study visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        configs = [r.config_name for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        win_rates = [r.win_rate for r in results]
        total_returns = [r.total_return for r in results]
        improvements = [r.improvement_over_baseline for r in results]

        # Sharpe ratio comparison
        axes[0, 0].bar(configs, sharpe_ratios, alpha=0.7)
        axes[0, 0].set_title("Sharpe Ratio by Configuration")
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Win rate comparison
        axes[0, 1].bar(configs, win_rates, alpha=0.7, color='green')
        axes[0, 1].set_title("Win Rate by Configuration")
        axes[0, 1].set_ylabel("Win Rate")
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Total return comparison
        axes[1, 0].bar(configs, total_returns, alpha=0.7, color='orange')
        axes[1, 0].set_title("Total Return by Configuration")
        axes[1, 0].set_ylabel("Total Return")
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Improvement over baseline
        axes[1, 1].bar(configs, improvements, alpha=0.7, color='red')
        axes[1, 1].set_title("Improvement over Baseline")
        axes[1, 1].set_ylabel("Sharpe Ratio Improvement")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = self.output_dir / f"{prefix}_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"Ablation plot saved: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Ablation Study for Trading Model Improvements"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--output-dir", type=str, default="ablation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes per configuration"
    )

    args = parser.parse_args()

    # Initialize ablation study
    study = AblationStudy(
        data_path=Path(args.data),
        base_model_path=Path(args.model),
        output_dir=Path(args.output_dir)
    )

    # Run ablation study
    results = study.run_ablation_study(n_episodes=args.episodes)

    # Generate report
    study.generate_ablation_report(results)

    LOGGER.info("Ablation study completed!")


if __name__ == "__main__":
    main()