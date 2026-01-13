"""
Reward Function Display Manager

Manages display and visualization of reward function optimization results.
Separated from the main optimizer to follow Single Responsibility Principle.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ztb.analysis.common.plot_utils import save_plot
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RewardFunctionDisplayManager:
    """
    Manages display and visualization of reward function optimization results.

    Responsibilities:
    - Displaying optimization results
    - Creating plots and visualizations
    - Formatting output for different display modes
    - Managing display configuration
    """

    def __init__(self, output_dir: str = "optimization_results"):
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def display_optimization_results(
        self,
        results: Dict[str, Any],
        stage_name: str,
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> None:
        """
        Display optimization results for a given stage.

        Args:
            results: Optimization results dictionary
            stage_name: Name of the optimization stage
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
        """
        self.logger.info(f"Displaying results for stage: {stage_name}")

        # Display summary statistics
        self._display_summary_stats(results, stage_name)

        # Display best parameters
        self._display_best_parameters(results, stage_name)

        # Create and display plots
        if show_plots or save_plots:
            self._create_optimization_plots(results, stage_name, show_plots, save_plots)

    def _display_summary_stats(self, results: Dict[str, Any], stage_name: str) -> None:
        """Display summary statistics of optimization results."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION RESULTS - {stage_name.upper()}")
        print(f"{'='*60}")

        if "best_value" in results:
            print(f"Best Objective Value: {results['best_value']:.6f}")

        if "best_params" in results:
            print(f"Number of Parameters: {len(results['best_params'])}")

        if "n_trials" in results:
            print(f"Total Trials: {results['n_trials']}")

        if "elapsed_time" in results:
            print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")

        print(f"{'='*60}\n")

    def _display_best_parameters(
        self, results: Dict[str, Any], stage_name: str
    ) -> None:
        """Display the best parameters found during optimization."""
        if "best_params" not in results:
            return

        print(f"BEST PARAMETERS - {stage_name.upper()}")
        print("-" * 40)

        best_params = results["best_params"]
        for param_name, param_value in best_params.items():
            if isinstance(param_value, float):
                print(f"{param_name}: {param_value:.6f}")
            else:
                print(f"{param_name}: {param_value}")

        print()

    def _create_optimization_plots(
        self,
        results: Dict[str, Any],
        stage_name: str,
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> None:
        """Create optimization plots."""
        try:
            # Create parameter importance plot
            self._plot_parameter_importance(results, stage_name, show_plots, save_plots)

            # Create optimization history plot
            self._plot_optimization_history(results, stage_name, show_plots, save_plots)

            # Create parameter correlation plot
            self._plot_parameter_correlations(
                results, stage_name, show_plots, save_plots
            )

        except Exception as e:
            self.logger.warning(f"Failed to create plots: {e}")

    def _plot_parameter_importance(
        self,
        results: Dict[str, Any],
        stage_name: str,
        show_plots: bool,
        save_plots: bool,
    ) -> None:
        """Plot parameter importance."""
        if "param_importance" not in results:
            return

        importance_data = results["param_importance"]

        plt.figure(figsize=(12, 8))
        params = list(importance_data.keys())
        importance = list(importance_data.values())

        plt.barh(params, importance)
        plt.xlabel("Importance")
        plt.ylabel("Parameters")
        plt.title(f"Parameter Importance - {stage_name}")
        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir
                / f"parameter_importance_{stage_name.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved parameter importance plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _plot_optimization_history(
        self,
        results: Dict[str, Any],
        stage_name: str,
        show_plots: bool,
        save_plots: bool,
    ) -> None:
        """Plot optimization history."""
        if "trials" not in results:
            return

        trials = results["trials"]

        plt.figure(figsize=(12, 6))

        # Plot objective values over trials
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]

        plt.plot(trial_numbers, values, "b-", alpha=0.7)
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value")
        plt.title(f"Optimization History - {stage_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir
                / f"optimization_history_{stage_name.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved optimization history plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _plot_parameter_correlations(
        self,
        results: Dict[str, Any],
        stage_name: str,
        show_plots: bool,
        save_plots: bool,
    ) -> None:
        """Plot parameter correlations."""
        if "trials" not in results:
            return

        trials = results["trials"]
        if len(trials) < 2:
            return

        # Extract parameter values
        param_data = {}
        for trial in trials:
            for param_name, param_value in trial.params.items():
                if param_name not in param_data:
                    param_data[param_name] = []
                param_data[param_name].append(param_value)

        if len(param_data) < 2:
            return

        # Create correlation matrix
        df = pd.DataFrame(param_data)
        correlation_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
        )
        plt.title(f"Parameter Correlations - {stage_name}")
        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir
                / f"parameter_correlations_{stage_name.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved parameter correlations plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def display_comparison_results(
        self,
        comparison_data: Dict[str, Any],
        metric_names: List[str],
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> None:
        """
        Display comparison results between different configurations.

        Args:
            comparison_data: Comparison data dictionary
            metric_names: List of metric names to display
            show_plots: Whether to display plots
            save_plots: Whether to save plots
        """
        print(f"\n{'='*60}")
        print("CONFIGURATION COMPARISON RESULTS")
        print(f"{'='*60}")

        # Display comparison table
        self._display_comparison_table(comparison_data, metric_names)

        # Create comparison plots
        if show_plots or save_plots:
            self._create_comparison_plots(
                comparison_data, metric_names, show_plots, save_plots
            )

    def _display_comparison_table(
        self, comparison_data: Dict[str, Any], metric_names: List[str]
    ) -> None:
        """Display comparison table."""
        if "configurations" not in comparison_data:
            return

        configs = comparison_data["configurations"]

        print(f"{'Configuration':<20} {' | '.join(f'{m:<12}' for m in metric_names)}")
        print("-" * (20 + len(metric_names) * 14))

        for config_name, metrics in configs.items():
            metric_values = [f"{metrics.get(m, 0):<12.6f}" for m in metric_names]
            print(f"{config_name:<20} {' | '.join(metric_values)}")

        print()

    def _create_comparison_plots(
        self,
        comparison_data: Dict[str, Any],
        metric_names: List[str],
        show_plots: bool,
        save_plots: bool,
    ) -> None:
        """Create comparison plots."""
        if "configurations" not in comparison_data:
            return

        configs = comparison_data["configurations"]
        config_names = list(configs.keys())

        # Create bar plot for each metric
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metric_names):
            values = [configs[config].get(metric, 0) for config in config_names]

            axes[i].bar(config_names, values)
            axes[i].set_title(f"{metric} Comparison")
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_plots:
            filename = self.output_dir / "configuration_comparison.png"
            save_plot(filename)
            self.logger.info(f"Saved comparison plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()
