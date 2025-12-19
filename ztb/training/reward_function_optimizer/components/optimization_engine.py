"""
Optimization Engine - Core optimization logic for reward functions.

This module separates optimization-related logic from the main optimizer class,
including parameter search, objective evaluation, and convergence checking.
"""

from typing import Any, Callable, Dict, List, Optional

from ztb.training.reward_function_optimizer.constants import (
    DEFAULT_OPTIMIZATION_TRIALS,
    SAMPLER_SEED,
    EVALUATION_TIMEOUT_SECONDS,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OptimizationEngine:
    """
    Handles core optimization logic for reward function parameters.

    This class manages:
    - Parameter space exploration
    - Objective function evaluation
    - Optimization algorithm execution
    - Convergence detection
    """

    def __init__(self):
        """Initialize OptimizationEngine."""
        self.logger = get_logger(__name__)
        self.optuna_available = self._check_optuna_available()
        self.parameter_spaces = {}
        self.optimization_history = []

    def _check_optuna_available(self) -> bool:
        """Check if Optuna is available."""
        import importlib.util

        available = importlib.util.find_spec("optuna") is not None
        if not available:
            self.logger.warning("Optuna not available for optimization")
        return available

    def define_parameter_space(
        self,
        stage: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Define parameter space for optimization.

        Args:
            stage: Optimization stage name
            parameters: Parameter definitions

        Returns:
            Parameter space configuration

        Raises:
            ValueError: If parameter space is invalid
        """
        try:
            if not isinstance(parameters, dict):
                raise ValueError("Parameters must be a dictionary")

            space_config = {
                "stage": stage,
                "parameters": parameters,
                "bounds": self._extract_bounds(parameters),
                "types": self._extract_types(parameters),
            }

            self.parameter_spaces[stage] = space_config
            self.logger.info(f"Defined parameter space for stage: {stage}")

            return space_config

        except Exception as e:
            self.logger.error(f"Failed to define parameter space: {e}")
            raise

    def _extract_bounds(self, parameters: Dict[str, Any]) -> Dict[str, tuple]:
        """Extract parameter bounds from definitions."""
        bounds = {}
        for param_name, param_config in parameters.items():
            if isinstance(param_config, dict):
                if "low" in param_config and "high" in param_config:
                    bounds[param_name] = (param_config["low"], param_config["high"])
                elif "choices" in param_config:
                    # For categorical parameters, use index bounds
                    choices = param_config["choices"]
                    bounds[param_name] = (0, len(choices) - 1)
            else:
                # Default bounds for numeric parameters
                bounds[param_name] = (0.0, 1.0)

        return bounds

    def _extract_types(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Extract parameter types from definitions."""
        types = {}
        for param_name, param_config in parameters.items():
            if isinstance(param_config, dict):
                if "choices" in param_config:
                    types[param_name] = "categorical"
                else:
                    types[param_name] = "float"
            else:
                types[param_name] = "float"

        return types

    def optimize(
        self,
        stage: str,
        evaluation_function: Callable,
        n_trials: int = DEFAULT_OPTIMIZATION_TRIALS,
        objectives: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        parameter_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization for reward function parameters.

        Args:
            stage: Reward function stage to optimize
            evaluation_function: Function that evaluates parameter performance
            n_trials: Number of optimization trials
            objectives: List of objectives to optimize
            constraints: Optimization constraints
            parameter_spaces: Parameter spaces for different stages

        Returns:
            Optimization results

        Raises:
            RuntimeError: If optimization fails
        """
        if not self.optuna_available:
            raise RuntimeError("Optuna not available for optimization")

        if parameter_spaces is None or stage not in parameter_spaces:
            raise ValueError(f"Unknown reward stage: {stage}")

        objectives = objectives or ["profit", "sharpe", "win_rate"]
        constraints = constraints or {}

        try:
            import optuna
            import time

            start_time = time.time()
            param_space = parameter_spaces[stage]

            def optuna_objective(trial):
                """Objective function for Optuna optimization."""
                params = {}

                # Sample parameters for the stage
                for param_name, param_def in param_space.items():
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
                        if param_def.low is not None and param_def.high is not None:
                            params[param_name] = trial.suggest_int(
                                param_name, int(param_def.low), int(param_def.high)
                            )
                    elif param_def.type == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_def.choices
                        )

                # Evaluate parameters using actual backtest
                try:
                    scores = evaluation_function(params)
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for trial {trial.number}: {e}")
                    # Return poor scores for failed evaluations
                    scores = dict.fromkeys(objectives, -999)

                # Store trial information
                trial_info = {
                    "trial_number": trial.number,
                    "parameters": params.copy(),
                    "scores": scores.copy(),
                    "timestamp": time.time(),
                }
                self.optimization_history.append(trial_info)

                # Return composite score (weighted average of objectives)
                # For simplicity, use profit as primary objective
                return scores.get("profit", 0.0)

            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
                pruner=optuna.pruners.MedianPruner(),
            )

            # Run optimization
            study.optimize(optuna_objective, n_trials=n_trials, timeout=EVALUATION_TIMEOUT_SECONDS)

            # Extract best results
            best_params = study.best_params
            best_score = study.best_value

            # Get best scores by re-evaluating best parameters
            try:
                best_scores = evaluation_function(best_params)
            except Exception as e:
                self.logger.warning(f"Final evaluation failed: {e}")
                best_scores = {"profit": best_score}

            # Create result dictionary
            result = {
                "best_config": {
                    "stage": stage,
                    "parameters": best_params,
                    "objectives": objectives,
                    "constraints": constraints,
                },
                "best_scores": best_scores,
                "optimization_history": self.optimization_history.copy(),
                "optimization_time": time.time() - start_time,
                "convergence_info": {
                    "best_trial_number": study.best_trial.number if study.best_trial else None,
                    "study_best_value": best_score,
                    "n_trials": len(study.trials),
                    "n_completed_trials": len(study.trials),  # All trials are completed
                },
            }

            self.logger.info(f"Optimization completed for stage '{stage}'. Best score: {best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Optimization failed: {e}") from e

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Optimization failed: {e}") from e

    def evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        evaluation_function: Callable,
        n_evaluations: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate parameter set with multiple runs.

        Args:
            parameters: Parameters to evaluate
            evaluation_function: Function to evaluate parameters
            n_evaluations: Number of evaluation runs

        Returns:
            Evaluation results

        Raises:
            RuntimeError: If evaluation fails
        """
        try:
            scores = []

            for i in range(n_evaluations):
                self.logger.debug(f"Evaluation run {i + 1}/{n_evaluations}")
                score = evaluation_function(parameters)
                scores.append(score)

            import numpy as np

            evaluation_results = {
                "parameters": parameters,
                "scores": scores,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "n_evaluations": n_evaluations,
            }

            self.logger.info(f"Parameter evaluation completed. Mean score: {evaluation_results['mean_score']:.4f}")
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            raise RuntimeError(f"Parameter evaluation failed: {e}") from e

    def check_convergence(
        self,
        history: List[Dict[str, Any]],
        window_size: int = 10,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Check if optimization has converged.

        Args:
            history: Optimization history
            window_size: Window size for convergence check
            tolerance: Convergence tolerance

        Returns:
            True if converged, False otherwise
        """
        try:
            if len(history) < window_size:
                return False

            recent_scores = [h["score"] for h in history[-window_size:]]
            max_score = max(recent_scores)
            min_score = min(recent_scores)

            # Check if improvement is below tolerance
            improvement = max_score - min_score
            converged = improvement < tolerance

            if converged:
                self.logger.info(f"Optimization converged. Improvement in last {window_size} trials: {improvement}")

            return converged

        except Exception as e:
            self.logger.error(f"Convergence check failed: {e}")
            return False
