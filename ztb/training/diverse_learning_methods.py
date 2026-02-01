"""
Diverse Learning Methods Integration

Provides integration with multiple optimization frameworks for efficient hyperparameter tuning:
- Ray Tune: Distributed hyperparameter optimization
- Hyperopt: Bayesian optimization
- BOHB: Budget-aware hyperparameter optimization
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.io.data_loader import DataLoader
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DiverseLearningMethods:
    """
    Integration of diverse learning methods for optimization.

    Supports multiple optimization frameworks for comprehensive hyperparameter tuning.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/optimization_config.json"
        self.results_cache = {}
        self.frameworks = {
            "ray_tune": self._setup_ray_tune,
            "hyperopt": self._setup_hyperopt,
            "bohb": self._setup_bohb,
        }

    def optimize_hyperparameters(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        framework: str = "ray_tune",
        max_evals: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using specified framework.

        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            framework: Optimization framework ('ray_tune', 'hyperopt', 'bohb')
            max_evals: Maximum number of evaluations
            **kwargs: Framework-specific parameters

        Returns:
            Optimization results
        """
        if framework not in self.frameworks:
            raise ValueError(f"Unsupported framework: {framework}")

        logger.info(f"Starting hyperparameter optimization with {framework}")

        # Setup framework
        optimizer = self.frameworks[framework]()

        # Run optimization
        start_time = time.time()
        results = optimizer.optimize(
            objective_function=objective_function,
            search_space=search_space,
            max_evals=max_evals,
            **kwargs,
        )
        end_time = time.time()

        results["optimization_time"] = end_time - start_time
        results["framework"] = framework
        results["max_evals"] = max_evals

        logger.info(
            f"Optimization completed in {results['optimization_time']:.2f} seconds"
        )
        logger.info(f"Best value: {results.get('best_value', 'N/A')}")

        return results

    def _setup_ray_tune(self):
        """Setup Ray Tune optimizer."""
        try:
            from ray import tune
            from ray.tune.schedulers import ASHAScheduler
            from ray.tune.search.hyperopt import HyperOptSearch

            class RayTuneOptimizer:
                def optimize(
                    self, objective_function, search_space, max_evals, **kwargs
                ):
                    def trainable(config):
                        return objective_function(config)

                    # Convert search space to Ray Tune format
                    ray_search_space = self._convert_to_ray_space(search_space)

                    # Setup scheduler and search algorithm
                    scheduler = ASHAScheduler(
                        max_t=max_evals, grace_period=10, reduction_factor=2
                    )

                    search_alg = HyperOptSearch(
                        space=ray_search_space, metric="objective", mode="min"
                    )

                    # Run optimization
                    analysis = tune.run(
                        trainable,
                        config=ray_search_space,
                        num_samples=max_evals,
                        scheduler=scheduler,
                        search_alg=search_alg,
                        metric="objective",
                        mode="min",
                        **kwargs,
                    )

                    return {
                        "best_params": analysis.best_config,
                        "best_value": analysis.best_result["objective"],
                        "all_results": analysis.results_df.to_dict()
                        if hasattr(analysis, "results_df")
                        else {},
                    }

                def _convert_to_ray_space(self, space):
                    """Convert generic search space to Ray Tune format."""
                    ray_space = {}
                    for param, spec in space.items():
                        if isinstance(spec, dict):
                            if spec.get("type") == "choice":
                                ray_space[param] = tune.choice(spec["values"])
                            elif spec.get("type") == "uniform":
                                ray_space[param] = tune.uniform(
                                    spec["low"], spec["high"]
                                )
                            elif spec.get("type") == "loguniform":
                                ray_space[param] = tune.loguniform(
                                    spec["low"], spec["high"]
                                )
                        else:
                            ray_space[param] = spec
                    return ray_space

            return RayTuneOptimizer()

        except ImportError:
            raise ImportError(
                "Ray Tune not available. Install with: pip install ray[tune]"
            )

    def _setup_hyperopt(self):
        """Setup Hyperopt optimizer."""
        try:
            from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

            class HyperoptOptimizer:
                def optimize(
                    self, objective_function, search_space, max_evals, **kwargs
                ):
                    def objective(config):
                        result = objective_function(config)
                        return {"loss": result, "status": STATUS_OK}

                    # Convert search space to Hyperopt format
                    hyperopt_space = self._convert_to_hyperopt_space(search_space)

                    # Setup trials object
                    trials = Trials()

                    # Run optimization
                    best = fmin(
                        fn=objective,
                        space=hyperopt_space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials,
                        **kwargs,
                    )

                    return {
                        "best_params": best,
                        "best_value": min([t["result"]["loss"] for t in trials.trials]),
                        "all_results": trials.results,
                    }

                def _convert_to_hyperopt_space(self, space):
                    """Convert generic search space to Hyperopt format."""
                    hyperopt_space = {}
                    for param, spec in space.items():
                        if isinstance(spec, dict):
                            if spec.get("type") == "choice":
                                hyperopt_space[param] = hp.choice(param, spec["values"])
                            elif spec.get("type") == "uniform":
                                hyperopt_space[param] = hp.uniform(
                                    param, spec["low"], spec["high"]
                                )
                            elif spec.get("type") == "loguniform":
                                hyperopt_space[param] = hp.loguniform(
                                    param, np.log(spec["low"]), np.log(spec["high"])
                                )
                        else:
                            hyperopt_space[param] = spec
                    return hyperopt_space

            return HyperoptOptimizer()

        except ImportError:
            raise ImportError(
                "Hyperopt not available. Install with: pip install hyperopt"
            )

    def _setup_bohb(self):
        """Setup BOHB optimizer."""
        try:
            import hpbandster.core.nameserver as hpns
            from hpbandster.core.worker import Worker
            from hpbandster.optimizers import BOHB

            class BOHBOptimizer:
                def optimize(
                    self, objective_function, search_space, max_evals, **kwargs
                ):
                    # BOHB requires more complex setup, simplified version
                    class MyWorker(Worker):
                        def __init__(self, objective_fn, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.objective_fn = objective_fn

                        def compute(self, config, budget, **kwargs):
                            result = self.objective_fn(config)
                            return {"loss": result, "info": {"budget": budget}}

                    # Setup nameserver
                    NS = hpns.NameServer(
                        run_id="bohb_optimization", host="127.0.0.1", port=0
                    )
                    NS.start()

                    # Setup worker
                    worker = MyWorker(
                        objective_function,
                        nameserver="127.0.0.1",
                        run_id="bohb_optimization",
                    )
                    worker.run(background=True)

                    # Convert search space
                    bohb_space = self._convert_to_bohb_space(search_space)

                    # Setup optimizer
                    optimizer = BOHB(
                        configspace=bohb_space,
                        run_id="bohb_optimization",
                        nameserver="127.0.0.1",
                        min_budget=1,
                        max_budget=max_evals,
                    )

                    # Run optimization
                    result = optimizer.run(n_iterations=max_evals)

                    # Shutdown
                    optimizer.shutdown(shutdown_workers=True)
                    NS.shutdown()

                    return {
                        "best_params": result.get_incumbent_id(),
                        "best_value": result.get_incumbent_trajectory()[-1][1]
                        if result.get_incumbent_trajectory()
                        else None,
                        "all_results": result.get_all_runs(),
                    }

                def _convert_to_bohb_space(self, space):
                    """Convert generic search space to BOHB format."""
                    from hpbandster.core.configspace import ConfigurationSpace
                    from hpbandster.core.configspace.hyperparameters import (
                        CategoricalHyperparameter,
                        UniformFloatHyperparameter,
                        UniformIntegerHyperparameter,
                    )

                    cs = ConfigurationSpace()
                    for param, spec in space.items():
                        if isinstance(spec, dict):
                            if spec.get("type") == "choice":
                                cs.add_hyperparameter(
                                    CategoricalHyperparameter(param, spec["values"])
                                )
                            elif spec.get("type") == "uniform":
                                if isinstance(spec["low"], int) and isinstance(
                                    spec["high"], int
                                ):
                                    cs.add_hyperparameter(
                                        UniformIntegerHyperparameter(
                                            param, spec["low"], spec["high"]
                                        )
                                    )
                                else:
                                    cs.add_hyperparameter(
                                        UniformFloatHyperparameter(
                                            param, spec["low"], spec["high"]
                                        )
                                    )
                    return cs

            return BOHBOptimizer()

        except ImportError:
            raise ImportError(
                "BOHB not available. Install with: pip install hpbandster ConfigSpace"
            )

    def compare_frameworks(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        frameworks: List[str] = ["ray_tune", "hyperopt", "bohb"],
        max_evals: int = 50,
    ) -> Dict[str, Any]:
        """
        Compare performance of different optimization frameworks.

        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            frameworks: List of frameworks to compare
            max_evals: Maximum evaluations per framework

        Returns:
            Comparison results
        """
        logger.info(f"Comparing optimization frameworks: {frameworks}")

        results = {}
        for framework in frameworks:
            try:
                logger.info(f"Running {framework} optimization...")
                result = self.optimize_hyperparameters(
                    objective_function=objective_function,
                    search_space=search_space,
                    framework=framework,
                    max_evals=max_evals,
                )
                results[framework] = result
                logger.info(
                    f"{framework} completed: best_value={result.get('best_value', 'N/A')}"
                )
            except Exception as e:
                logger.error(f"{framework} failed: {e}")
                results[framework] = {"error": str(e)}

        # Generate comparison summary
        comparison = self._generate_comparison_summary(results)
        results["comparison"] = comparison

        return results

    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary of optimization results."""
        summary = {
            "frameworks_tested": list(results.keys()),
            "best_performers": [],
            "average_times": {},
            "success_rate": {},
        }

        successful_results = {k: v for k, v in results.items() if "error" not in v}

        if successful_results:
            # Find best performers
            best_value = min(
                [r.get("best_value", float("inf")) for r in successful_results.values()]
            )
            summary["best_performers"] = [
                k
                for k, v in successful_results.items()
                if v.get("best_value") == best_value
            ]

            # Calculate average times
            summary["average_times"] = {
                k: v.get("optimization_time", 0) for k, v in successful_results.items()
            }

        summary["success_rate"] = {
            "successful": len(successful_results),
            "total": len(results),
            "rate": len(successful_results) / len(results) if results else 0,
        }

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save optimization results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def load_results(self, input_path: str) -> Dict[str, Any]:
        """Load optimization results from file."""
        with open(input_path, "r") as f:
            results = json.load(f)

        logger.info(f"Results loaded from {input_path}")
        return results


def create_trading_objective_function(backtest_data: pd.DataFrame):
    """
    Create objective function for trading strategy optimization.

    Args:
        backtest_data: Historical market data for backtesting

    Returns:
        Objective function for optimization
    """

    def objective(params):
        """
        Objective function that evaluates trading strategy performance.

        Args:
            params: Strategy parameters

        Returns:
            Negative Sharpe ratio (for minimization)
        """
        try:
            # Import here to avoid circular imports
            from ztb.trading.backtest.adapters import RLPolicyAdapter
            from ztb.metrics.metrics import MetricsCalculator
            from ztb.trading.backtest.runner import BacktestEngine

            # Create adapter with parameters
            adapter = RLPolicyAdapter(
                enable_150d_features=params.get("enable_150d_features", True)
            )

            # Create backtest engine
            engine = BacktestEngine(
                initial_capital=params.get("initial_capital", 10000),
                slippage_bps=params.get("slippage_bps", 5.0),
                commission_bps=params.get("commission_bps", 0.0),
            )

            # Run backtest
            equity_series, orders_df, adaptation_summary = engine.run_backtest(
                adapter, backtest_data
            )

            # Calculate metrics
            returns = MetricsCalculator.calculate_returns(equity_series)
            sharpe_ratio = MetricsCalculator.calculate_sharpe_ratio(returns)

            # Return negative Sharpe ratio for minimization
            return -sharpe_ratio if not np.isnan(sharpe_ratio) else 1000

        except Exception as e:
            logger.error(f"Objective function error: {e}")
            return 1000  # High penalty for failures

    return objective


# Example usage and predefined search spaces
def get_trading_search_space() -> Dict[str, Any]:
    """Get predefined search space for trading strategy optimization."""
    return {
        "initial_capital": {"type": "choice", "values": [1000, 5000, 10000, 50000]},
        "slippage_bps": {"type": "uniform", "low": 1.0, "high": 10.0},
        "commission_bps": {"type": "uniform", "low": 0.0, "high": 2.0},
        "enable_150d_features": {"type": "choice", "values": [True, False]},
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "choice", "values": [16, 32, 64, 128]},
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load sample data
    try:
        data = DataLoader.load_csv_strict("data/btc_jpy_real_dataset.csv")
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")

        # Create optimizer
        optimizer = DiverseLearningMethods()

        # Define objective function
        objective_fn = create_trading_objective_function(data)

        # Get search space
        search_space = get_trading_search_space()

        # Compare frameworks
        results = optimizer.compare_frameworks(
            objective_function=objective_fn,
            search_space=search_space,
            frameworks=["hyperopt"],  # Start with hyperopt for simplicity
            max_evals=10,
        )

        # Save results
        optimizer.save_results(results, "results/optimization_comparison.json")

        print("Optimization comparison completed!")

    except Exception as e:
        print(f"Example failed: {e}")
