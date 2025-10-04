#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Trading Models

This script provides comprehensive evaluation and comparison between single PPO models
and ensemble models across multiple evaluation metrics and visualization.

Features:
- Single model vs Ensemble model comparison
- Multiple evaluation metrics (Sharpe, Sortino, Calmar, max drawdown, etc.)
- Cross-validation support
- Ablation study capabilities
- Trade analysis and action distribution visualization
- Automated graph generation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from stable_baselines3 import PPO

from ztb.utils.data_utils import load_csv_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

TRADING_DAYS_PER_YEAR = 252

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment import HeavyTradingEnv  # noqa: E402
from ztb.trading.ensemble import EnsemblePredictor  # noqa: E402

# Import evaluation modules (lazy import to avoid circular imports)
evaluation_modules_available = True
evaluation_analyzers = {}

def _import_evaluation_modules():
    """Lazy import of evaluation modules to avoid circular imports."""
    global evaluation_analyzers, evaluation_modules_available
    try:
        from performance_attribution import PerformanceAttributionAnalyzer
        from monte_carlo_simulation import MonteCarloSimulator
        from strategy_robustness import StrategyRobustnessAnalyzer
        # Temporarily disable benchmark_comparison due to import issues
        # from benchmark_comparison import BenchmarkComparisonAnalyzer
        from risk_parity_analysis import RiskParityAnalyzer
        from cost_sensitivity import CostSensitivityAnalyzer

        evaluation_analyzers = {
            'performance_attribution': PerformanceAttributionAnalyzer(),
            'monte_carlo': MonteCarloSimulator(),
            'strategy_robustness': StrategyRobustnessAnalyzer(),
            # 'benchmark_comparison': BenchmarkComparisonAnalyzer(risk_free_rate=0.02),  # Disabled due to import issues
            'risk_parity': RiskParityAnalyzer(),
            'cost_sensitivity': CostSensitivityAnalyzer(base_fee=0.001)  # Default value
        }
        evaluation_modules_available = True
    except ImportError as e:
        print(f"Warning: Some evaluation modules not available: {e}")
        evaluation_modules_available = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Comprehensive evaluation metrics for trading models."""

    # Basic metrics
    total_return: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Action distribution
    buy_actions: int = 0
    sell_actions: int = 0
    hold_actions: int = 0

    # Time series data for plotting
    returns: List[float] = field(default_factory=list)
    cumulative_returns: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Extended evaluation results
    extended_evaluation: Dict[str, Any] = field(default_factory=dict)
    comprehensive_score: float = 0.0
    risk_adjusted_score: float = 0.0
    robustness_score: float = 0.0
    consistency_score: float = 0.0


@dataclass
class BenchmarkResult:
    """Result container for benchmark evaluation."""

    model_name: str
    model_type: str  # 'single' or 'ensemble'
    metrics: BenchmarkMetrics
    config: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for trading model evaluation."""

    def __init__(
        self,
        data_path: Path,
        output_dir: Optional[Path] = None,
        n_episodes: int = 10,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02
    ) -> None:
        """
        Initialize benchmark suite.

        Args:
            data_path: Path to evaluation data
            output_dir: Directory to save results and plots
            n_episodes: Number of evaluation episodes
            transaction_cost: Transaction cost per trade
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.data_path = data_path
        self.output_dir = output_dir or Path("benchmark_results")
        self.n_episodes = n_episodes
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        self.output_dir.mkdir(exist_ok=True)

        # Initialize evaluation modules if available (lazy import)
        self.evaluation_analyzers = {}
        if evaluation_modules_available:
            try:
                _import_evaluation_modules()
                # Update analyzers with instance parameters
                # benchmark_comparison disabled due to import issues
                # if 'benchmark_comparison' in evaluation_analyzers:
                #     evaluation_analyzers['benchmark_comparison'] = BenchmarkComparisonAnalyzer(risk_free_rate=risk_free_rate)
                if 'cost_sensitivity' in evaluation_analyzers:
                    from cost_sensitivity import CostSensitivityAnalyzer
                    evaluation_analyzers['cost_sensitivity'] = CostSensitivityAnalyzer(base_fee=transaction_cost)
                self.evaluation_analyzers = evaluation_analyzers.copy()
                LOGGER.info("Extended evaluation modules initialized")
            except Exception as e:
                LOGGER.warning(f"Failed to initialize some evaluation modules: {e}")
        else:
            LOGGER.info("Extended evaluation modules not available")

        # Load evaluation data
        self.eval_data = self._load_data()

        # Create evaluation environment
        self.env_config = {
            "reward_scaling": 1.0,
            "transaction_cost": transaction_cost,
            "max_position_size": 1.0,
            "risk_free_rate": risk_free_rate,
            "feature_set": "full",  # Use full features
            "initial_portfolio_value": 1000000.0,  # Default initial portfolio value
        }
        
        self.initial_portfolio_value = self.env_config["initial_portfolio_value"]

        LOGGER.info(f"Initialized benchmark with {len(self.eval_data)} data points")

    def _load_data(self) -> pd.DataFrame:
        """Load evaluation data."""
        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            df = load_csv_data(self.data_path)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")

        LOGGER.info(f"Loaded {len(df)} rows from {self.data_path}")
        return df

    def _create_env(self, model_path: Optional[Path] = None) -> HeavyTradingEnv:
        """Create evaluation environment."""
        # Determine max_features from model if provided
        max_features = 26  # Default
        if model_path:
            try:
                from stable_baselines3 import PPO
                model = PPO.load(model_path, device='cpu')
                max_features = model.observation_space.shape[0]
                LOGGER.info(f"Using max_features={max_features} to match model observation space")
            except Exception as e:
                LOGGER.warning(f"Could not determine model observation space, using default max_features=26: {e}")

        return HeavyTradingEnv(
            df=self.eval_data,
            config=self.env_config,
            streaming_pipeline=None,
            stream_batch_size=1000,
            max_features=max_features
        )

    def evaluate_single_model(
        self,
        model_path: Path,
        model_name: str = "single_model"
    ) -> BenchmarkResult:
        """
        Evaluate a single PPO model.

        Args:
            model_path: Path to the model file
            model_name: Name for the model

        Returns:
            BenchmarkResult with evaluation metrics
        """
        LOGGER.info(f"Evaluating single model: {model_name}")

        # Load model
        model = PPO.load(str(model_path))

        # Run evaluation episodes
        all_rewards = []
        all_returns = []
        all_actions = []

        for episode in tqdm(range(self.n_episodes), desc=f"Evaluating {model_name}", unit="episode"):
            env = self._create_env(model_path)
            obs, info = env.reset()  # Unpack obs and info
            initial_portfolio_value = info.get('portfolio_value', self.initial_portfolio_value)
            done = False
            episode_rewards = []
            episode_actions = []
            final_portfolio_value = initial_portfolio_value

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action.item()) if hasattr(action, 'item') else int(action)
                obs, reward, terminated, truncated, info = env.step(action_int)  # Unpack all values
                done = terminated or truncated

                episode_rewards.append(reward)
                episode_actions.append(self._action_to_string(action_int))
                final_portfolio_value = info.get('portfolio_value', final_portfolio_value)

            # Calculate episode metrics based on portfolio value change
            episode_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
            all_rewards.extend(episode_rewards)
            all_returns.append(episode_return)
            all_actions.extend(episode_actions)

            LOGGER.debug(f"Episode {episode + 1}: Return = {episode_return:.4f}")

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(all_rewards, all_returns, all_actions)

        # Run extended evaluation if available
        if self.evaluation_analyzers:
            extended_results = self._run_extended_evaluation(metrics, model_name)
            metrics.extended_evaluation = extended_results
            metrics.comprehensive_score = self._calculate_comprehensive_score(metrics)
            metrics.risk_adjusted_score = self._calculate_risk_adjusted_score(metrics)
            metrics.robustness_score = self._calculate_robustness_score(metrics)

        return BenchmarkResult(
            model_name=model_name,
            model_type="single",
            metrics=metrics,
            config={"model_path": str(model_path)}
        )

    def evaluate_ensemble_model(
        self,
        model_paths: List[Path],
        ensemble_name: str = "ensemble",
        voting_method: str = "confidence_weighted"
    ) -> BenchmarkResult:
        """
        Evaluate an ensemble of models.

        Args:
            model_paths: List of paths to ensemble model files
            ensemble_name: Name for the ensemble
            voting_method: Voting method ('soft', 'hard', 'confidence_weighted')

        Returns:
            BenchmarkResult with evaluation metrics
        """
        LOGGER.info(f"Evaluating ensemble: {ensemble_name} with {len(model_paths)} models")

        # Load models
        models = [PPO.load(str(path)) for path in model_paths]

        # Create ensemble predictor with model configs
        model_configs = [
            {"path": str(path), "weight": 1.0, "feature_set": "full"}
            for path in model_paths
        ]
        ensemble = EnsemblePredictor(model_configs)

        # Run evaluation episodes
        all_rewards = []
        all_returns = []
        all_actions = []

        for episode in tqdm(range(self.n_episodes), desc=f"Evaluating {ensemble_name}", unit="episode"):
            env = self._create_env(model_paths[0] if model_paths else None)
            obs, info = env.reset()  # Unpack obs and info
            initial_portfolio_value = info.get('portfolio_value', self.initial_portfolio_value)
            done = False
            episode_rewards = []
            episode_actions = []
            final_portfolio_value = initial_portfolio_value

            while not done:
                action, _ = ensemble.predict(obs)
                action_int = int(action.item()) if hasattr(action, 'item') else int(action)
                obs, reward, terminated, truncated, info = env.step(action_int)  # Unpack all values
                done = terminated or truncated

                episode_rewards.append(reward)
                episode_actions.append(self._action_to_string(action_int))
                final_portfolio_value = info.get('portfolio_value', final_portfolio_value)

            # Calculate episode metrics based on portfolio value change
            episode_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
            all_rewards.extend(episode_rewards)
            all_returns.append(episode_return)
            all_actions.extend(episode_actions)

            LOGGER.debug(f"Episode {episode + 1}: Return = {episode_return:.4f}")

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(all_rewards, all_returns, all_actions)

        return BenchmarkResult(
            model_name=ensemble_name,
            model_type="ensemble",
            metrics=metrics,
            config={
                "model_paths": [str(p) for p in model_paths],
                "voting_method": voting_method,
                "n_models": len(model_paths)
            }
        )

    def _action_to_string(self, action: int) -> str:
        """Convert action index to string."""
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_map.get(action, "UNKNOWN")

    def _calculate_metrics(
        self,
        rewards: List[float],
        returns: List[float],
        actions: List[str]
    ) -> BenchmarkMetrics:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            rewards: List of step rewards
            returns: List of episode returns
            actions: List of actions taken

        Returns:
            BenchmarkMetrics object
        """
        metrics = BenchmarkMetrics()

        # Basic metrics
        metrics.total_return = float(np.sum(returns))
        metrics.avg_return = float(np.mean(returns))
        metrics.volatility = float(np.std(returns))

        # Win rate (positive returns)
        metrics.win_rate = float(np.mean([r > 0 for r in returns]))

        # Total trades (BUY + SELL actions)
        metrics.total_trades = sum(1 for a in actions if a in ["BUY", "SELL"])

        # Action distribution
        metrics.buy_actions = actions.count("BUY")
        metrics.sell_actions = actions.count("SELL")
        metrics.hold_actions = actions.count("HOLD")

        # Risk-adjusted metrics
        if len(returns) > 1:
            # Sharpe ratio
            excess_returns = np.array(returns) - self.risk_free_rate / TRADING_DAYS_PER_YEAR
            if np.std(excess_returns) > 0:
                metrics.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

            # Sortino ratio (downside deviation)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    metrics.sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)

            # Calmar ratio (return / max drawdown)
            metrics.max_drawdown = self._calculate_max_drawdown(returns)
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = metrics.total_return / abs(metrics.max_drawdown)

        # Store time series data
        metrics.returns = returns
        metrics.cumulative_returns = np.cumsum(returns).tolist()
        metrics.drawdowns = self._calculate_drawdowns(returns)
        metrics.actions = actions

        # Calculate consistency score (inverse of coefficient of variation)
        if len(returns) > 1 and np.std(returns) > 0:
            metrics.consistency_score = float(np.mean(returns) / np.std(returns))
        else:
            metrics.consistency_score = 0.0

        return metrics

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return np.min(drawdowns)

    def _calculate_drawdowns(self, returns: List[float]) -> List[float]:
        """Calculate drawdown series."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return drawdowns.tolist()

    def _run_extended_evaluation(self, metrics: BenchmarkMetrics, model_name: str) -> Dict[str, Any]:
        """Run extended evaluation using specialized analyzers."""
        extended_results = {}
        
        for analyzer_name, analyzer in self.evaluation_analyzers.items():
            try:
                if hasattr(analyzer, 'analyze'):
                    result = analyzer.analyze(metrics)
                    extended_results[analyzer_name] = result
                    LOGGER.debug(f"Completed {analyzer_name} analysis for {model_name}")
                else:
                    LOGGER.warning(f"Analyzer {analyzer_name} does not have analyze method")
            except Exception as e:
                LOGGER.warning(f"Failed to run {analyzer_name} analysis: {e}")
                extended_results[analyzer_name] = {"error": str(e)}
        
        return extended_results

    def cross_validate(
        self,
        model_paths: List[Path],
        n_splits: int = 5,
        model_type: str = "single"
    ) -> BenchmarkResult:
        """
        Perform cross-validation evaluation.

        Args:
            model_paths: Model paths (single model or ensemble)
            n_splits: Number of CV splits
            model_type: 'single' or 'ensemble'

        Returns:
            BenchmarkResult with CV metrics
        """
        LOGGER.info(f"Performing {n_splits}-fold cross-validation")

        # Split data into folds
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(self.eval_data)),
                                              desc="Cross-validation folds",
                                              total=n_splits,
                                              unit="fold"):
            LOGGER.info(f"Evaluating fold {fold + 1}/{n_splits}")

            # Create fold-specific data
            fold_data = self.eval_data.iloc[val_idx].copy()

            # Temporarily replace eval data
            original_data = self.eval_data
            self.eval_data = fold_data

            try:
                if model_type == "single":
                    result = self.evaluate_single_model(model_paths[0], f"cv_fold_{fold}")
                else:  # ensemble
                    result = self.evaluate_ensemble_model(model_paths, f"cv_fold_{fold}")

                cv_scores.append(result.metrics.total_return)

            finally:
                # Restore original data
                self.eval_data = original_data

        # Calculate CV statistics
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Create result with CV metrics
        metrics = BenchmarkMetrics(
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std
        )

        return BenchmarkResult(
            model_name=f"{model_type}_cv",
            model_type=model_type,
            metrics=metrics,
            config={"n_splits": n_splits, "cv_scores": cv_scores}
        )

    def ablation_study(
        self,
        base_config: Dict[str, Any],
        ablation_configs: Dict[str, Dict[str, Any]],
        model_path: Path
    ) -> Dict[str, BenchmarkResult]:
        """
        Perform ablation study by testing different configurations.

        Args:
            base_config: Base configuration
            ablation_configs: Dictionary of ablation configurations
            model_path: Path to the model

        Returns:
            Dictionary of results for each configuration
        """
        LOGGER.info("Performing ablation study")

        results = {}

        # Evaluate base configuration
        LOGGER.info("Evaluating base configuration")
        # Note: This would require modifying environment/reward logic
        # For now, we'll create placeholder results
        results["base"] = self.evaluate_single_model(model_path, "base_config")

        # Evaluate ablation configurations
        for ablation_name, config_diff in ablation_configs.items():
            LOGGER.info(f"Evaluating ablation: {ablation_name}")
            # Note: Actual implementation would modify the environment/trainer
            # based on config_diff and evaluate
            results[ablation_name] = self.evaluate_single_model(
                model_path,
                f"ablation_{ablation_name}"
            )

        return results

    def generate_comparison_report(
        self,
        results: List[BenchmarkResult],
        output_prefix: str = "benchmark_comparison"
    ) -> Path:
        """
        Generate comprehensive comparison report with plots.

        Args:
            results: List of benchmark results to compare
            output_prefix: Prefix for output files

        Returns:
            Path to the generated report
        """
        LOGGER.info("Generating comparison report")

        # Create summary table
        summary_data = []
        for result in results:
            m = result.metrics
            summary_data.append({
                "Model": result.model_name,
                "Type": result.model_type,
                "Total Return": f"{m.total_return:.4f}",
                "Avg Return": f"{m.avg_return:.4f}",
                "Win Rate": f"{m.win_rate:.2%}",
                "Sharpe": f"{m.sharpe_ratio:.4f}",
                "Sortino": f"{m.sortino_ratio:.4f}",
                "Calmar": f"{m.calmar_ratio:.4f}",
                "Max DD": f"{m.max_drawdown:.4f}",
                "Total Trades": m.total_trades,
                "Buy Actions": m.buy_actions,
                "Sell Actions": m.sell_actions,
                "Hold Actions": m.hold_actions,
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        summary_path = self.output_dir / f"{output_prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Generate plots
        self._generate_comparison_plots(results, output_prefix)

        # Generate action distribution plot
        self._generate_action_distribution_plot(results, output_prefix)

        # Save detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "model_name": result.model_name,
                "model_type": result.model_type,
                "metrics": {
                    "total_return": result.metrics.total_return,
                    "avg_return": result.metrics.avg_return,
                    "win_rate": result.metrics.win_rate,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "sortino_ratio": result.metrics.sortino_ratio,
                    "calmar_ratio": result.metrics.calmar_ratio,
                    "max_drawdown": result.metrics.max_drawdown,
                    "total_trades": result.metrics.total_trades,
                    "buy_actions": result.metrics.buy_actions,
                    "sell_actions": result.metrics.sell_actions,
                    "hold_actions": result.metrics.hold_actions,
                    "returns": result.metrics.returns,
                    "cumulative_returns": result.metrics.cumulative_returns,
                    "drawdowns": result.metrics.drawdowns,
                    "extended_evaluation": result.metrics.extended_evaluation,
                    "comprehensive_score": getattr(result.metrics, 'comprehensive_score', None),
                    "risk_adjusted_score": getattr(result.metrics, 'risk_adjusted_score', None),
                    "robustness_score": getattr(result.metrics, 'robustness_score', None),
                },
                "config": result.config
            })

        detailed_path = self.output_dir / f"{output_prefix}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        LOGGER.info(f"Report generated: {summary_path}, {detailed_path}")
        return summary_path

    def _generate_comparison_plots(
        self,
        results: List[BenchmarkResult],
        prefix: str
    ) -> None:
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Cumulative returns
        ax = axes[0, 0]
        for result in results:
            if result.metrics.cumulative_returns:
                ax.plot(result.metrics.cumulative_returns,
                       label=result.model_name, linewidth=2)
        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True)

        # Drawdown comparison
        ax = axes[0, 1]
        for result in results:
            if result.metrics.drawdowns:
                ax.plot(result.metrics.drawdowns,
                       label=result.model_name, linewidth=2)
        ax.set_title("Drawdown Analysis")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Drawdown")
        ax.legend()
        ax.grid(True)

        # Sharpe vs Return scatter
        ax = axes[1, 0]
        for result in results:
            ax.scatter(result.metrics.sharpe_ratio, result.metrics.total_return,
                      s=100, label=result.model_name, alpha=0.7)
        ax.set_title("Risk-Return Analysis")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Total Return")
        ax.legend()
        ax.grid(True)

        # Win rate comparison
        ax = axes[1, 1]
        model_names = [r.model_name for r in results]
        win_rates = [r.metrics.win_rate for r in results]
        bars = ax.bar(model_names, win_rates, alpha=0.7)
        ax.set_title("Win Rate Comparison")
        ax.set_ylabel("Win Rate")
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{rate:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = self.output_dir / f"{prefix}_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"Comparison plots saved: {plot_path}")

    def _generate_action_distribution_plot(
        self,
        results: List[BenchmarkResult],
        prefix: str
    ) -> None:
        """Generate action distribution comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for grouped bar chart
        model_names = [r.model_name for r in results]
        buy_counts = [r.metrics.buy_actions for r in results]
        sell_counts = [r.metrics.sell_actions for r in results]
        hold_counts = [r.metrics.hold_actions for r in results]

        x = np.arange(len(model_names))
        width = 0.25

        ax.bar(x - width, buy_counts, width, label='BUY', alpha=0.8)
        ax.bar(x, hold_counts, width, label='HOLD', alpha=0.8)
        ax.bar(x + width, sell_counts, width, label='SELL', alpha=0.8)

        ax.set_title("Action Distribution Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Action Count")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"{prefix}_actions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _run_extended_evaluation(
        self,
        metrics: BenchmarkMetrics,
        model_name: str
    ) -> Dict[str, Any]:
        """Run extended evaluation using all available analyzers."""
        extended_results = {}

        for analyzer_name, analyzer in tqdm(self.evaluation_analyzers.items(),
                                           desc=f"Extended evaluation for {model_name}",
                                           unit="analyzer"):
            try:
                LOGGER.info(f"Running {analyzer_name} analysis for {model_name}")
                result = analyzer.analyze(metrics)
                extended_results[analyzer_name] = result
            except Exception as e:
                LOGGER.warning(f"Failed to run {analyzer_name} analysis: {e}")
                extended_results[analyzer_name] = {"error": str(e)}

        return extended_results

    def _calculate_comprehensive_score(self, metrics: BenchmarkMetrics) -> float:
        """Calculate comprehensive score combining all evaluation results."""
        if not metrics.extended_evaluation:
            return metrics.total_return

        scores = []

        # Base score from traditional metrics
        base_score = (
            metrics.total_return * 0.4 +
            metrics.sharpe_ratio * 0.3 +
            metrics.win_rate * 0.3
        )
        scores.append(base_score)

        # Add scores from extended evaluations
        for result in metrics.extended_evaluation.values():
            if isinstance(result, dict) and "score" in result:
                scores.append(result["score"] * 0.2)  # Weight each extended analysis

        return float(np.mean(scores)) if scores else base_score

    def _calculate_risk_adjusted_score(self, metrics: BenchmarkMetrics) -> float:
        """Calculate risk-adjusted score focusing on risk management."""
        if not metrics.extended_evaluation:
            return metrics.sharpe_ratio

        risk_scores = [metrics.sharpe_ratio, metrics.sortino_ratio]

        # Add risk-focused scores from extended evaluations
        for analyzer_name, result in metrics.extended_evaluation.items():
            if isinstance(result, dict):
                if "risk_adjusted_score" in result:
                    risk_scores.append(result["risk_adjusted_score"])
                elif analyzer_name in ["risk_parity", "monte_carlo"]:
                    # Extract risk metrics from these analyzers
                    if "var_95" in result:
                        risk_scores.append(1.0 / (1.0 + abs(result["var_95"])))  # Lower VaR is better

        return float(np.mean(risk_scores)) if risk_scores else metrics.sharpe_ratio

    def _calculate_robustness_score(self, metrics: BenchmarkMetrics) -> float:
        """Calculate robustness score based on stability and consistency."""
        if not metrics.extended_evaluation:
            return metrics.max_drawdown * -1  # Negative because lower drawdown is better

        robustness_scores = []

        # Base robustness from drawdown and consistency
        base_robustness = (
            (1.0 - abs(metrics.max_drawdown)) * 0.5 +  # Lower drawdown = higher robustness
            metrics.consistency_score * 0.5
        )
        robustness_scores.append(base_robustness)

        # Add robustness from extended evaluations
        for analyzer_name, result in metrics.extended_evaluation.items():
            if isinstance(result, dict):
                if "robustness_score" in result:
                    robustness_scores.append(result["robustness_score"])
                elif analyzer_name == "strategy_robustness":
                    # Extract robustness metrics
                    if "stress_test_score" in result:
                        robustness_scores.append(result["stress_test_score"])

        return float(np.mean(robustness_scores)) if robustness_scores else base_robustness

        LOGGER.info(f"Action distribution plot saved: {plot_path}")


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark Suite for Trading Models"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to evaluation data (parquet or csv)"
    )
    parser.add_argument(
        "--single-model", type=str,
        help="Path to single model for evaluation"
    )
    parser.add_argument(
        "--ensemble-models", nargs="+",
        help="Paths to ensemble models"
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--transaction-cost", type=float, default=0.001,
        help="Transaction cost per trade"
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = ComprehensiveBenchmark(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        n_episodes=args.episodes,
        transaction_cost=args.transaction_cost
    )

    results = []

    # Evaluate single model
    if args.single_model:
        result = benchmark.evaluate_single_model(
            Path(args.single_model),
            "single_model"
        )

        # Cross-validation for single model
        cv_result = benchmark.cross_validate(
            [Path(args.single_model)],
            n_splits=args.cv_folds,
            model_type="single"
        )

        # Merge CV results into main result
        result.metrics.cv_scores = cv_result.metrics.cv_scores
        result.metrics.cv_mean = cv_result.metrics.cv_mean
        result.metrics.cv_std = cv_result.metrics.cv_std
        result.config["cv_scores"] = cv_result.config["cv_scores"]

        results.append(result)

    # Evaluate ensemble
    if args.ensemble_models:
        ensemble_paths = [Path(p) for p in args.ensemble_models]
        result = benchmark.evaluate_ensemble_model(
            ensemble_paths,
            "ensemble"
        )
        results.append(result)

        # Cross-validation for ensemble
        cv_result = benchmark.cross_validate(
            ensemble_paths,
            n_splits=args.cv_folds,
            model_type="ensemble"
        )
        results.append(cv_result)

    # Generate comparison report
    if results:
        benchmark.generate_comparison_report(results)
        LOGGER.info("Benchmark evaluation completed!")
    else:
        LOGGER.error("No models specified for evaluation")


if __name__ == "__main__":
    main()