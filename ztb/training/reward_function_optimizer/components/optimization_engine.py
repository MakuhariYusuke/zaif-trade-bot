"""
Optimization Engine - Core optimization logic for reward functions.

This module separates optimization-related logic from the main optimizer class,
including parameter search, objective evaluation, and convergence checking.
"""

from collections import deque
from typing import Callable, Protocol

from ztb.training.hyperparameter_optimizer import ParameterSpace
from ztb.training.reward_function_optimizer.constants import (
    DEFAULT_OPTIMIZATION_TRIALS,
    SAMPLER_SEED,
    EVALUATION_TIMEOUT_SECONDS,
)
from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import ensure_dict, safe_to_float

logger = get_logger(__name__)

ConfigMap = dict[str, object]
ScoreMap = dict[str, float]
HistoryRecord = dict[str, object]
ParameterDefinition = ParameterSpace | ConfigMap


class TrialLike(Protocol):
    """Minimal Optuna trial protocol used by the optimizer."""

    number: int

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False
    ) -> float: ...

    def suggest_int(self, name: str, low: int, high: int) -> int: ...

    def suggest_categorical(self, name: str, choices: list[object]) -> object: ...


class OptimizationEngine:
    """
    Handles core optimization logic for reward function parameters.

    This class manages:
    - Parameter space exploration
    - Objective function evaluation
    - Optimization algorithm execution
    - Convergence detection
    """

    def __init__(self, history_limit: int = 5000):
        """Initialize OptimizationEngine."""
        self.logger = get_logger(__name__)
        self.optuna_available = self._check_optuna_available()
        self.parameter_spaces: dict[str, dict[str, ParameterDefinition]] = {}
        self.optimization_history: deque[HistoryRecord] = deque(maxlen=history_limit)

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
        parameters: dict[str, ParameterDefinition],
    ) -> ConfigMap:
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

    def _extract_bounds(
        self, parameters: dict[str, ParameterDefinition]
    ) -> dict[str, tuple[object, object]]:
        """Extract parameter bounds from definitions."""
        bounds: dict[str, tuple[object, object]] = {}
        for param_name, param_config in parameters.items():
            param_def = self._parameter_definition_to_map(param_config)
            if "low" in param_def and "high" in param_def:
                bounds[param_name] = (param_def["low"], param_def["high"])
                continue
            choices = param_def.get("choices")
            if isinstance(choices, list) and choices:
                # For categorical parameters, use index bounds
                bounds[param_name] = (0, len(choices) - 1)
                continue
            # Default bounds for numeric parameters
            bounds[param_name] = (0.0, 1.0)

        return bounds

    def _extract_types(self, parameters: dict[str, ParameterDefinition]) -> dict[str, str]:
        """Extract parameter types from definitions."""
        types: dict[str, str] = {}
        for param_name, param_config in parameters.items():
            param_def = self._parameter_definition_to_map(param_config)
            param_type = str(param_def.get("type", "float"))
            if param_type in {"float", "int", "categorical"}:
                types[param_name] = param_type
                continue
            choices = param_def.get("choices")
            types[param_name] = "categorical" if isinstance(choices, list) else "float"

        return types

    def sample_parameters_for_trial(
        self,
        trial: TrialLike,
        parameter_space: dict[str, ParameterDefinition],
    ) -> ConfigMap:
        """
        Sample parameters for a trial from mixed parameter definitions.

        Supports both `ParameterSpace` instances and dict-based definitions.
        """
        params: ConfigMap = {}
        for param_name, param_def in parameter_space.items():
            sampled = self._sample_parameter_value(trial, param_name, param_def)
            if sampled is not None:
                params[param_name] = sampled
        return params

    @staticmethod
    def _parameter_definition_to_map(param_def: ParameterDefinition) -> ConfigMap:
        if isinstance(param_def, ParameterSpace):
            return {
                "type": param_def.type,
                "low": param_def.low,
                "high": param_def.high,
                "choices": param_def.choices,
                "log_scale": param_def.log_scale,
            }
        return param_def if isinstance(param_def, dict) else {}

    def _sample_parameter_value(
        self,
        trial: TrialLike,
        param_name: str,
        param_def: ParameterDefinition,
    ) -> object | None:
        definition = self._parameter_definition_to_map(param_def)
        param_type = str(definition.get("type", "float"))

        if param_type == "float":
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                self.logger.warning(
                    "Skipping float parameter '%s' due to missing bounds", param_name
                )
                return None
            low_f = safe_to_float(low, 0.0)
            high_f = safe_to_float(high, low_f)
            if low_f > high_f:
                low_f, high_f = high_f, low_f
            return trial.suggest_float(
                param_name, low_f, high_f, log=bool(definition.get("log_scale", False))
            )

        if param_type == "int":
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                self.logger.warning(
                    "Skipping int parameter '%s' due to missing bounds", param_name
                )
                return None
            low_i = int(round(safe_to_float(low, 0.0)))
            high_i = int(round(safe_to_float(high, float(low_i))))
            if low_i > high_i:
                low_i, high_i = high_i, low_i
            return trial.suggest_int(param_name, low_i, high_i)

        if param_type == "categorical":
            choices = definition.get("choices")
            if not isinstance(choices, list) or not choices:
                self.logger.warning(
                    "Skipping categorical parameter '%s' due to missing choices",
                    param_name,
                )
                return None
            return trial.suggest_categorical(param_name, choices)

        self.logger.warning(
            "Skipping unsupported parameter type '%s' for '%s'", param_type, param_name
        )
        return None

    def optimize(
        self,
        stage: str,
        evaluation_function: Callable[[ConfigMap], ScoreMap],
        n_trials: int = DEFAULT_OPTIMIZATION_TRIALS,
        objectives: list[str] | None = None,
        constraints: ConfigMap | None = None,
        parameter_spaces: dict[str, dict[str, ParameterDefinition]] | None = None,
    ) -> ConfigMap:
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

            def optuna_objective(trial: TrialLike) -> float:
                """Objective function for Optuna optimization."""
                params = self.sample_parameters_for_trial(trial, param_space)

                # Evaluate parameters using actual backtest
                try:
                    scores = ensure_dict(evaluation_function(params))
                except Exception as e:
                    self.logger.warning(
                        f"Evaluation failed for trial {getattr(trial, 'number', -1)}: {e}"
                    )
                    # Return poor scores for failed evaluations
                    scores = {objective: -999.0 for objective in objectives}
                    scores.setdefault("profit", -999.0)

                # Store trial information
                trial_info: HistoryRecord = {
                    "trial_number": getattr(trial, "number", -1),
                    "parameters": params.copy(),
                    "scores": scores.copy(),
                    "timestamp": time.time(),
                }
                self.optimization_history.append(trial_info)

                # Return composite score (weighted average of objectives)
                # For simplicity, use profit as primary objective
                return safe_to_float(scores.get("profit", 0.0), 0.0)

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
                best_scores = ensure_dict(evaluation_function(best_params))
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
                "optimization_history": list(self.optimization_history),
                "optimization_time": time.time() - start_time,
                "convergence_info": {
                    "best_trial_number": study.best_trial.number if study.best_trial else None,
                    "study_best_value": best_score,
                    "n_trials": len(study.trials),
                    "n_completed_trials": len(
                        [t for t in study.trials if getattr(t.state, "name", "") == "COMPLETE"]
                    ),
                },
            }

            self.logger.info(f"Optimization completed for stage '{stage}'. Best score: {best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Optimization failed: {e}") from e

    def evaluate_parameters(
        self,
        parameters: ConfigMap,
        evaluation_function: Callable[[ConfigMap], float],
        n_evaluations: int = 5,
    ) -> ConfigMap:
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
        history: list[HistoryRecord],
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

            recent_scores = [
                self._extract_history_score(h)
                for h in history[-window_size:]
            ]
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

    @staticmethod
    def _extract_history_score(record: HistoryRecord) -> float:
        if "score" in record:
            return safe_to_float(record.get("score", 0.0), 0.0)
        scores = ensure_dict(record.get("scores"))
        return safe_to_float(scores.get("profit", 0.0), 0.0)
