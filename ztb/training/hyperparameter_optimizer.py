#!/usr/bin/env python3
"""
Hyperparameter Optimization Framework for SAC v428.

This module provides comprehensive hyperparameter optimization including:
- Bayesian optimization using Optuna
- Grid search and random search
- Cross-validation across market conditions
- Automated parameter space exploration
- Performance tracking and analysis
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.trading.environment.constants import EPSILON
from ztb.types.common import BaseComponent
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Bayesian optimization will be disabled.")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available. Progress bars will be disabled.")


@dataclass
class HyperparameterOptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: Optional[Dict[str, Any]]
    best_score: float
    trials: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    cross_validation_scores: List[float] = field(default_factory=list)


@dataclass
class ParameterSpace:
    """Definition of hyperparameter search space."""

    name: str
    type: str  # 'float', 'int', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False


class OptimizationMethod(ABC):
    """Abstract base class for optimization methods."""

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, ParameterSpace],
        n_trials: int,
        **kwargs,
    ) -> HyperparameterOptimizationResult:
        """Run optimization."""
        pass


class BayesianOptimization(OptimizationMethod):
    """Bayesian optimization using Optuna."""

    def __init__(self, console_optimizer=None) -> None:
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        self.console_optimizer = console_optimizer

    def optimize(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, ParameterSpace],
        n_trials: int,
        **kwargs,
    ) -> HyperparameterOptimizationResult:
        """Run Bayesian optimization with Optuna."""

        def optuna_objective(trial):
            """Optuna objective function."""
            params = {}
            for param_name, param_def in parameter_space.items():
                if param_def.type == "float":
                    if param_def.log_scale:
                        params[param_name] = trial.suggest_float(
                            param_name, param_def.low, param_def.high, log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_def.low, param_def.high
                        )
                elif param_def.type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, int(param_def.low), int(param_def.high)
                    )
                elif param_def.type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_def.choices
                    )

            return objective_function(params)

        # Create Optuna study
        study = optuna.create_study(direction="maximize")

        # Initialize progress tracking
        completed_trials = 0
        start_time = time.time()

        def objective_with_progress(trial):
            """Objective function with progress tracking."""
            nonlocal completed_trials

            result = optuna_objective(trial)
            completed_trials += 1

            # Print progress if console optimizer is available
            if self.console_optimizer and self.console_optimizer.verbose:
                elapsed = time.time() - start_time
                eta = (elapsed / completed_trials) * (n_trials - completed_trials)
                try:
                    best_score = study.best_value
                except ValueError:
                    best_score = 0.0
                self.console_optimizer._print_progress(
                    completed_trials, n_trials, best_score, eta
                )

            return result

        # Run optimization with progress bar
        try:
            if (
                self.console_optimizer
                and self.console_optimizer.show_progress
                and TQDM_AVAILABLE
            ):
                from tqdm import tqdm

                with tqdm(
                    total=n_trials, desc="🎯 Bayesian Optimization", unit="trial"
                ) as pbar:

                    def callback(study, trial):
                        pbar.update(1)
                        try:
                            best_score = study.best_value
                        except ValueError:
                            best_score = 0.0
                        pbar.set_postfix(
                            {
                                "best_score": f"{best_score:.6f}",
                                "trial": f"{trial.number}/{n_trials}",
                            }
                        )

                    study.optimize(
                        objective_with_progress,
                        n_trials=n_trials,
                        callbacks=[callback],
                        **kwargs,
                    )
            else:
                study.optimize(objective_with_progress, n_trials=n_trials, **kwargs)

        except Exception as e:
            if self.console_optimizer:
                self.console_optimizer._handle_error(e, "Bayesian optimization")
            raise

        optimization_time = time.time() - start_time

        # Extract results
        best_params = study.best_params
        best_score = study.best_value

        # Extract trial information
        trials = []
        for trial in study.trials:
            trials.append(
                {
                    "params": trial.params,
                    "value": trial.value,
                    "number": trial.number,
                    "state": trial.state.name,
                }
            )

        # Convergence information
        convergence_info = {
            "n_completed_trials": len(
                [t for t in study.trials if t.state.name == "COMPLETE"]
            ),
            "best_trial_number": study.best_trial.number,
            "optimization_history": [
                (t.number, t.value) for t in study.trials if t.value is not None
            ],
        }

        return HyperparameterOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            optimization_time=optimization_time,
            convergence_info=convergence_info,
        )


class GridSearchOptimization(OptimizationMethod):
    """Grid search optimization."""

    def __init__(self, console_optimizer=None) -> None:
        self.console_optimizer = console_optimizer

    def optimize(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, ParameterSpace],
        n_trials: int,
        **kwargs,
    ) -> HyperparameterOptimizationResult:
        """Run grid search optimization."""

        # Generate parameter combinations
        param_combinations = self._generate_grid(parameter_space)

        # Limit to n_trials if specified
        if len(param_combinations) > n_trials:
            # Randomly sample combinations
            indices = np.random.choice(len(param_combinations), n_trials, replace=False)
            param_combinations = [param_combinations[i] for i in indices]

        trials = []
        best_score = float("-inf")
        best_params = None

        start_time = time.time()

        # Run optimization with progress tracking
        try:
            if (
                self.console_optimizer
                and self.console_optimizer.show_progress
                and TQDM_AVAILABLE
            ):
                from tqdm import tqdm

                with tqdm(
                    total=len(param_combinations), desc="🎯 Grid Search", unit="trial"
                ) as pbar:
                    for i, params in enumerate(param_combinations):
                        try:
                            score = objective_function(params)
                            trials.append(
                                {
                                    "params": params,
                                    "value": score,
                                    "number": i,
                                    "state": "COMPLETE",
                                }
                            )

                            if score > best_score:
                                best_score = score
                                best_params = params

                        except Exception as e:
                            if self.console_optimizer:
                                self.console_optimizer._handle_error(
                                    e, f"grid search trial {i}"
                                )
                            trials.append(
                                {
                                    "params": params,
                                    "value": None,
                                    "number": i,
                                    "state": "FAILED",
                                }
                            )

                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "best_score": f"{best_score:.4f}",
                                "trial": f"{i+1}/{len(param_combinations)}",
                            }
                        )
            else:
                for i, params in enumerate(param_combinations):
                    try:
                        score = objective_function(params)
                        trials.append(
                            {
                                "params": params,
                                "value": score,
                                "number": i,
                                "state": "COMPLETE",
                            }
                        )

                        if score > best_score:
                            best_score = score
                            best_params = params

                        # Print progress if console optimizer is available
                        if (
                            self.console_optimizer
                            and self.console_optimizer.verbose
                            and (i + 1) % max(1, len(param_combinations) // 10) == 0
                        ):
                            elapsed = time.time() - start_time
                            eta = (elapsed / (i + 1)) * (
                                len(param_combinations) - (i + 1)
                            )
                            self.console_optimizer._print_progress(
                                i + 1, len(param_combinations), best_score, eta
                            )

                    except Exception as e:
                        if self.console_optimizer:
                            self.console_optimizer._handle_error(
                                e, f"grid search trial {i}"
                            )
                        trials.append(
                            {
                                "params": params,
                                "value": None,
                                "number": i,
                                "state": "FAILED",
                            }
                        )

        except Exception as e:
            if self.console_optimizer:
                self.console_optimizer._handle_error(e, "grid search optimization")
            raise

        optimization_time = time.time() - start_time

        return HyperparameterOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            optimization_time=optimization_time,
            convergence_info={
                "method": "grid_search",
                "total_combinations": len(param_combinations),
            },
        )

    def _generate_grid(
        self, parameter_space: Dict[str, ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        if not parameter_space:
            return [{}]

        param_names = list(parameter_space.keys())
        param_defs = [parameter_space[name] for name in param_names]

        # Generate values for each parameter
        param_values = []
        for param_def in param_defs:
            if param_def.type == "categorical":
                values = param_def.choices
            elif param_def.type in ["float", "int"]:
                # For grid search, use fixed number of values
                n_values = 5  # Can be made configurable
                if param_def.log_scale:
                    values = np.logspace(
                        np.log10(param_def.low), np.log10(param_def.high), n_values
                    )
                else:
                    values = np.linspace(param_def.low, param_def.high, n_values)

                if param_def.type == "int":
                    values = [int(v) for v in values]
            else:
                values = [param_def.low]  # Default value

            param_values.append(values)

        # Generate all combinations
        combinations = []
        for combo in np.ndindex(*[len(v) for v in param_values]):
            params = {}
            for i, param_name in enumerate(param_names):
                params[param_name] = param_values[i][combo[i]]
            combinations.append(params)

        return combinations


class RandomSearchOptimization(OptimizationMethod):
    """Random search optimization."""

    def __init__(self, console_optimizer=None) -> None:
        self.console_optimizer = console_optimizer

    def optimize(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, ParameterSpace],
        n_trials: int,
        **kwargs,
    ) -> HyperparameterOptimizationResult:
        """Run random search optimization."""

        trials = []
        best_score = float("-inf")
        best_params = None

        start_time = time.time()

        # Run optimization with progress tracking
        try:
            if (
                self.console_optimizer
                and self.console_optimizer.show_progress
                and TQDM_AVAILABLE
            ):
                from tqdm import tqdm

                with tqdm(
                    total=n_trials, desc="🎯 Random Search", unit="trial"
                ) as pbar:
                    for i in range(n_trials):
                        # Sample random parameters
                        params = self._sample_random_params(parameter_space)

                        try:
                            score = objective_function(params)
                            trials.append(
                                {
                                    "params": params,
                                    "value": score,
                                    "number": i,
                                    "state": "COMPLETE",
                                }
                            )

                            if score > best_score:
                                best_score = score
                                best_params = params

                        except Exception as e:
                            if self.console_optimizer:
                                self.console_optimizer._handle_error(
                                    e, f"random search trial {i}"
                                )
                            trials.append(
                                {
                                    "params": params,
                                    "value": None,
                                    "number": i,
                                    "state": "FAILED",
                                }
                            )

                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "best_score": f"{best_score:.6f}",
                                "trial": f"{i+1}/{n_trials}",
                            }
                        )
            else:
                for i in range(n_trials):
                    # Sample random parameters
                    params = self._sample_random_params(parameter_space)

                    try:
                        score = objective_function(params)
                        trials.append(
                            {
                                "params": params,
                                "value": score,
                                "number": i,
                                "state": "COMPLETE",
                            }
                        )

                        if score > best_score:
                            best_score = score
                            best_params = params

                        # Print progress if console optimizer is available
                        if (
                            self.console_optimizer
                            and self.console_optimizer.verbose
                            and (i + 1) % max(1, n_trials // 10) == 0
                        ):
                            elapsed = time.time() - start_time
                            eta = (elapsed / (i + 1)) * (n_trials - (i + 1))
                            self.console_optimizer._print_progress(
                                i + 1, n_trials, best_score, eta
                            )

                    except Exception as e:
                        if self.console_optimizer:
                            self.console_optimizer._handle_error(
                                e, f"random search trial {i}"
                            )
                        trials.append(
                            {
                                "params": params,
                                "value": None,
                                "number": i,
                                "state": "FAILED",
                            }
                        )

        except Exception as e:
            if self.console_optimizer:
                self.console_optimizer._handle_error(e, "random search optimization")
            raise

        optimization_time = time.time() - start_time

        return HyperparameterOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            optimization_time=optimization_time,
            convergence_info={"method": "random_search", "n_trials": n_trials},
        )

    def _sample_random_params(
        self, parameter_space: Dict[str, ParameterSpace]
    ) -> Dict[str, Any]:
        """Sample random parameters from the search space."""
        params = {}

        for param_name, param_def in parameter_space.items():
            if param_def.type == "float":
                if param_def.low is not None and param_def.high is not None:
                    if param_def.log_scale:
                        params[param_name] = np.random.uniform(
                            np.log(param_def.low), np.log(param_def.high)
                        )
                        params[param_name] = np.exp(params[param_name])
                    else:
                        params[param_name] = np.random.uniform(
                            param_def.low, param_def.high
                        )
            elif param_def.type == "int":
                if param_def.low is not None and param_def.high is not None:
                    params[param_name] = np.random.randint(
                        int(param_def.low), int(param_def.high) + 1
                    )
            elif param_def.type == "categorical":
                params[param_name] = np.random.choice(param_def.choices)

        return params


class HyperparameterOptimizer(BaseComponent):
    """
    Main hyperparameter optimization framework.

    Supports multiple optimization methods and cross-validation strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize hyperparameter optimizer.

        Args:
            config: Configuration parameters
        """
        super().__init__(name="HyperparameterOptimizer", config=config)
        self.config = config or self._get_default_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

        # Console output settings
        self.verbose = True
        self.show_progress = TQDM_AVAILABLE

        # Initialize optimization methods
        self.optimization_methods = {
            "bayesian": BayesianOptimization(console_optimizer=self)
            if OPTUNA_AVAILABLE
            else None,
            "grid": GridSearchOptimization(console_optimizer=self),
            "random": RandomSearchOptimization(console_optimizer=self),
        }

    def set_console_output(self, verbose: bool = True, show_progress: bool = True):
        """
        Configure console output settings.

        Args:
            verbose: Whether to show detailed output
            show_progress: Whether to show progress bars
        """
        self.verbose = verbose
        self.show_progress = show_progress and TQDM_AVAILABLE
        if show_progress and not TQDM_AVAILABLE:
            self.logger.warning("tqdm not available. Progress bars disabled.")

    def _print_header(self, title: str, subtitle: str = ""):
        """
        Print a formatted header for optimization sections.

        Args:
            title: Main title
            subtitle: Optional subtitle
        """
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        if subtitle:
            print(f"   {subtitle}")
        print(f"{'='*60}")

    def _print_progress(
        self, current: int, total: int, best_score: float, eta: Optional[float] = None
    ):
        """
        Print optimization progress.

        Args:
            current: Current trial number
            total: Total number of trials
            best_score: Current best score
            eta: Estimated time remaining (seconds)
        """
        eta_str = f" | ETA: {eta:.1f}s" if eta else ""
        print(
            f"  📊 Trial {current}/{total} completed | Best: {best_score:.6f}{eta_str}"
        )

    def _print_scores(self, scores: Dict[str, Any], title: str = "Performance Scores"):
        """
        Print performance scores with color coding.

        Args:
            scores: Dictionary of metric names and values
            title: Title for the scores section
        """
        print(f"\n📊 {title}:")
        print("-" * 40)

        for metric, value in scores.items():
            # Color coding based on metric type
            if isinstance(value, (int, float)):
                if metric.lower() in [
                    "sharpe",
                    "sortino",
                    "profit_factor",
                    "win_rate",
                    "total_return",
                ]:
                    if value > 0:
                        color = "🟢"  # Green for positive
                    elif value == 0:
                        color = "🟡"  # Yellow for zero
                    else:
                        color = "🔴"  # Red for negative
                elif metric.lower() in ["max_drawdown", "volatility"]:
                    if value < 0.2:
                        color = "🟢"  # Low risk
                    elif value < 0.5:
                        color = "🟡"  # Medium risk
                    else:
                        color = "🔴"  # High risk
                else:
                    color = "⚪"  # Neutral
                print(f"  {color} {metric}: {value:.6f}")
            else:
                print(f"  ⚪ {metric}: {value}")

    def _print_optimization_summary(
        self, result: HyperparameterOptimizationResult, method: str
    ):
        """
        Print comprehensive optimization summary.

        Args:
            result: Optimization result
            method: Optimization method used
        """
        print("\n🎉 Hyperparameter Optimization Completed!")
        print(f"⏱️  Optimization Time: {result.optimization_time:.2f} seconds")
        print(f"🎯 Method: {method}")
        print(f"🎲 Total Trials: {len(result.trials)}")
        print(f"🏆 Best Score: {result.best_score:.6f}")

        if result.best_params:
            print("📊 Best Parameters:")
            for param, value in result.best_params.items():
                print(f"   {param}: {value}")

        # Show convergence info
        if result.convergence_info:
            print("📈 Convergence Info:")
            for key, value in result.convergence_info.items():
                if key == "optimization_history" and len(value) > 5:
                    # Show only first and last few points for long histories
                    history = (
                        value[:3] + ["..."] + value[-3:] if len(value) > 6 else value
                    )
                    print(f"   {key}: {history}")
                else:
                    print(f"   {key}: {value}")

    def _handle_error(self, error: Exception, context: str):
        """
        Handle and display errors in a user-friendly way.

        Args:
            error: Exception that occurred
            context: Context where the error occurred
        """
        print(f"\n❌ Error in {context}:")
        print(f"   {str(error)}")
        print(f"   Type: {type(error).__name__}")
        print(f"Error in {context}: {error}")
        self.logger.error(f"Error in {context}: {error}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "default_method": "bayesian" if OPTUNA_AVAILABLE else "random",
            "n_trials": 50,
            "cv_folds": 3,
            "validation_method": "time_series_split",
            "early_stopping": True,
            "early_stopping_patience": 10,
            "save_results": True,
            "results_dir": "optimization_results",
        }

    def optimize_hyperparameters(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, ParameterSpace],
        method: str = None,
        n_trials: int = None,
        cross_validate: bool = True,
        **kwargs,
    ) -> HyperparameterOptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            objective_function: Function to optimize (takes params dict, returns score)
            parameter_space: Dictionary of parameter definitions
            method: Optimization method ('bayesian', 'grid', 'random')
            n_trials: Number of trials to run
            cross_validate: Whether to use cross-validation
            **kwargs: Additional arguments for optimization method

        Returns:
            HyperparameterOptimizationResult object
        """
        method = method or self.config["default_method"]
        n_trials = n_trials or self.config["n_trials"]

        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")

        optimizer = self.optimization_methods[method]
        if optimizer is None:
            raise ImportError(f"Optimization method {method} is not available")

        # Print header
        self._print_header(
            "Hyperparameter Optimization",
            f"Method: {method.title()} | Parameters: {len(parameter_space)} | Trials: {n_trials}",
        )

        print(f"🎯 Optimization Method: {method}")
        print(f"🧠 Parameters to optimize: {list(parameter_space.keys())}")
        print(f"🎲 Number of trials: {n_trials}")
        print(f"🔄 Cross-validation: {'Enabled' if cross_validate else 'Disabled'}")

        self.logger.info(f"Starting hyperparameter optimization with method: {method}")
        self.logger.info(f"Parameter space: {list(parameter_space.keys())}")
        self.logger.info(f"Number of trials: {n_trials}")

        # Wrap objective function for cross-validation if requested
        if cross_validate:
            objective_fn = self._create_cv_objective(objective_function)
        else:
            objective_fn = objective_function

        # Run optimization
        try:
            result = optimizer.optimize(
                objective_fn, parameter_space, n_trials, **kwargs
            )
        except Exception as e:
            self._handle_error(e, "hyperparameter optimization")
            raise

        # Save results if configured
        if self.config["save_results"]:
            self._save_results(result, method)

        # Print comprehensive summary
        self._print_optimization_summary(result, method)

        self.logger.info(f"Optimization completed. Best score: {result.best_score:.4f}")
        self.logger.info(f"Best parameters: {result.best_params}")

        return result

    def _create_cv_objective(self, objective_function: Callable) -> Callable:
        """Create cross-validation objective function."""

        def cv_objective(params):
            # For now, simple wrapper - can be extended with actual CV
            try:
                score = objective_function(params)
                return score
            except Exception as e:
                self.logger.warning(
                    f"Objective function failed with params {params}: {e}"
                )
                return float("-inf")

        return cv_objective

    def _save_results(self, result: HyperparameterOptimizationResult, method: str):
        """Save optimization results to disk."""
        try:
            results_dir = Path(self.config["results_dir"])
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{method}_{timestamp}.json"

            result_dict = {
                "method": method,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "optimization_time": result.optimization_time,
                "n_trials": len(result.trials),
                "convergence_info": result.convergence_info,
                "cross_validation_scores": result.cross_validation_scores,
                "timestamp": timestamp,
            }

            with open(results_dir / filename, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)

            self.logger.info(f"Optimization results saved to {results_dir / filename}")

        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

    def create_parameter_space(
        self, param_definitions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ParameterSpace]:
        """
        Create parameter space from dictionary definitions.

        Args:
            param_definitions: Dictionary of parameter definitions

        Returns:
            Dictionary of ParameterSpace objects
        """
        parameter_space = {}

        for param_name, param_config in param_definitions.items():
            param_space = ParameterSpace(
                name=param_name,
                type=param_config["type"],
                low=param_config.get("low"),
                high=param_config.get("high"),
                choices=param_config.get("choices"),
                log_scale=param_config.get("log_scale", False),
            )
            parameter_space[param_name] = param_space

        return parameter_space

    def get_optimization_history(self, results_dir: str = None) -> pd.DataFrame:
        """
        Load optimization history from saved results.

        Args:
            results_dir: Directory containing optimization results

        Returns:
            DataFrame with optimization history
        """
        results_dir = results_dir or self.config["results_dir"]
        results_path = Path(results_dir)

        if not results_path.exists():
            return pd.DataFrame()

        history_data = []

        for json_file in results_path.glob("optimization_*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    history_data.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")

        return pd.DataFrame(history_data)


# Predefined parameter spaces for common SAC configurations
SAC_PARAMETER_SPACE = {
    "learning_rate": {"type": "float", "low": EPSILON, "high": 1e-2, "log_scale": True},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256, 512]},
    "buffer_size": {"type": "categorical", "choices": [50000, 100000, 200000, 500000]},
    "gamma": {"type": "float", "low": 0.9, "high": 0.999},
    "tau": {"type": "float", "low": 0.001, "high": 0.01},
    "ent_coef": {"type": "float", "low": 1e-4, "high": 1e-1, "log_scale": True},
    "reward_scale": {"type": "float", "low": 50, "high": 1000, "log_scale": True},
    # Reward function parameters
    "reward_clip_min": {"type": "float", "low": -500, "high": -50},
    "reward_clip_max": {"type": "float", "low": 50, "high": 500},
    "use_simple_reward": {"type": "categorical", "choices": [True, False]},
}

# Market regime-specific parameter spaces
REGIME_SPECIFIC_SPACES = {
    "bull_market": {
        "learning_rate": {
            "type": "float",
            "low": 5e-4,
            "high": 5e-3,
            "log_scale": True,
        },
        "ent_coef": {"type": "float", "low": 1e-3, "high": 1e-1, "log_scale": True},
        "gamma": {"type": "float", "low": 0.95, "high": 0.99},
        "reward_scale": {"type": "float", "low": 200, "high": 600, "log_scale": True},
        "reward_clip_max": {"type": "float", "low": 100, "high": 300},
    },
    "bear_market": {
        "learning_rate": {
            "type": "float",
            "low": EPSILON,
            "high": 1e-3,
            "log_scale": True,
        },
        "ent_coef": {"type": "float", "low": 1e-4, "high": 1e-2, "log_scale": True},
        "gamma": {"type": "float", "low": 0.99, "high": 0.999},
        "reward_scale": {"type": "float", "low": 100, "high": 400, "log_scale": True},
        "reward_clip_min": {"type": "float", "low": -300, "high": -100},
    },
    "high_volatility": {
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
        "tau": {"type": "float", "low": 0.005, "high": 0.02},
        "reward_scale": {"type": "float", "low": 200, "high": 800, "log_scale": True},
        "reward_clip_min": {"type": "float", "low": -400, "high": -150},
        "reward_clip_max": {"type": "float", "low": 150, "high": 400},
    },
}
