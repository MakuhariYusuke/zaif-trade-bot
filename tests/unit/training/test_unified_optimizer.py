#!/usr/bin/env python3
"""
Unit tests for unified_optimizer.py

Tests for UnifiedOptimizer, MultiTimeframeOptimizer, ABTestingFramework,
AutomaticOptimizationPipeline, OptimizationResultPersistence, and ParallelOptimizer.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import (
    make_scalar_objective,
    make_scalar_search_space,
    make_timeframe_objectives,
    make_timeframe_search_spaces,
)
from ztb.training.unified_optimizer import (
    ABTestingFramework,
    AutomaticOptimizationPipeline,
    BayesianOptimizer,
    MultiTimeframeOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationResultPersistence,
    ParallelOptimizer,
    UnifiedOptimizer,
)
from ztb.utils.system_utils import check_library_availability


class _StubOptimizer:
    def __init__(self, param_name: str = "param", value: float = 0.25):
        self.param_name = param_name
        self.value = value
        self.config = OptimizationConfig(max_trials=1)

    def optimize(self, objective, search_space):
        params = {self.param_name: self.value}
        return OptimizationResult(
            best_params=params,
            best_score=objective(params),
            optimization_history=[],
            execution_time=0.0,
            convergence_info={},
            recommendations=[],
        )


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.enable_hyperparameter_optimization is True
        assert config.optimization_method == "bayesian"
        assert config.max_trials == 100
        assert config.enable_system_optimization is True
        assert config.max_parallel_trials == 4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            enable_hyperparameter_optimization=False,
            optimization_method="grid",
            max_trials=50,
            enable_system_optimization=False,
            max_parallel_trials=2
        )

        assert config.enable_hyperparameter_optimization is False
        assert config.optimization_method == "grid"
        assert config.max_trials == 50
        assert config.enable_system_optimization is False
        assert config.max_parallel_trials == 2


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating OptimizationResult instance."""
        result = OptimizationResult(
            best_params={"x": 1.0, "y": 2.0},
            best_score=-0.5,
            optimization_history=[{"trial": 1, "score": -0.5}],
            execution_time=1.23,
            convergence_info={"converged": True},
            recommendations=["Use these parameters"]
        )

        assert result.best_params == {"x": 1.0, "y": 2.0}
        assert result.best_score == -0.5
        assert len(result.optimization_history) == 1
        assert result.execution_time == 1.23
        assert result.convergence_info == {"converged": True}
        assert result.recommendations == ["Use these parameters"]


class TestBayesianOptimizer:
    """Test BayesianOptimizer class."""

    def test_initialization(self):
        """Test BayesianOptimizer initialization."""
        config = OptimizationConfig()
        optimizer = BayesianOptimizer(config)

        assert optimizer.config == config
        assert hasattr(optimizer, 'logger')

    @unittest.skipUnless(check_library_availability('optuna', 'optimization'), "optuna not available")
    def test_optimize_with_optuna(self):
        """Test optimization with Optuna available."""
        config = OptimizationConfig(max_trials=2)
        objective = make_scalar_objective("x")
        search_space = make_scalar_search_space("x")

        optimizer = BayesianOptimizer(config)
        result = optimizer.optimize(objective, search_space)

        assert isinstance(result, OptimizationResult)
        assert 'best_params' in result.__dict__
        assert 'best_score' in result.__dict__
        assert 'optimization_history' in result.__dict__
        assert 'execution_time' in result.__dict__
        assert 'convergence_info' in result.__dict__
        assert 'recommendations' in result.__dict__

    @patch('ztb.training.unified_optimizer.OPTUNA_AVAILABLE', False)
    def test_optimize_without_optuna_raises_error(self):
        """Test that BayesianOptimizer raises error when Optuna is not available."""
        config = OptimizationConfig(max_trials=2)

        with pytest.raises(ImportError, match="Optuna is required for Bayesian optimization"):
            BayesianOptimizer(config)


class TestMultiTimeframeOptimizer:
    """Test MultiTimeframeOptimizer class."""

    def test_initialization(self):
        """Test MultiTimeframeOptimizer initialization."""
        config = OptimizationConfig()
        optimizer = MultiTimeframeOptimizer(config)

        assert optimizer.config == config
        assert optimizer.timeframes == ["1m", "5m", "15m"]
        assert optimizer.timeframe_weights == {"1m": 0.5, "5m": 0.3, "15m": 0.2}
        assert len(optimizer.timeframe_optimizers) == 3

    def test_optimize_multi_timeframe(self):
        """Test multi-timeframe optimization."""
        config = OptimizationConfig(max_trials=1)
        objective_functions = make_timeframe_objectives(["1m", "5m", "15m"], param_name="param")
        search_spaces = make_timeframe_search_spaces(["1m", "5m", "15m"], param_name="param")

        optimizer = MultiTimeframeOptimizer(config)
        optimizer.timeframe_optimizers = {
            timeframe: _StubOptimizer("param", value=0.25)
            for timeframe in optimizer.timeframes
        }
        result = optimizer.optimize_multi_timeframe(objective_functions, search_spaces)

        assert 'integrated' in result
        assert isinstance(result['integrated'], OptimizationResult)
        assert '1m' in result
        assert '5m' in result
        assert '15m' in result


class TestABTestingFramework:
    """Test ABTestingFramework class."""

    def test_initialization(self):
        """Test ABTestingFramework initialization."""
        config = OptimizationConfig()
        framework = ABTestingFramework(config)

        assert framework.config == config
        assert hasattr(framework, 'logger')

    def test_create_ab_test(self):
        """Test creating A/B test."""
        config = OptimizationConfig()
        framework = ABTestingFramework(config)

        test_id = framework.create_ab_test(
            'test_experiment',
            {'param': 1.0},
            {'param': 2.0},
            lambda p: p.get('param', 0)
        )

        assert test_id in framework.active_tests
        assert framework.active_tests[test_id]['status'] == 'created'

    def test_run_ab_test(self):
        """Test running A/B test."""
        config = OptimizationConfig()
        framework = ABTestingFramework(config)

        test_id = framework.create_ab_test(
            'test_experiment',
            {'param': 1.0},
            {'param': 2.0},
            lambda p: p.get('param', 0)
        )

        result = framework.run_ab_test(test_id, num_iterations=5)

        assert 'status' in result
        assert 'control_results' in result
        assert 'variant_results' in result
        assert len(result['control_results']) == 5
        assert len(result['variant_results']) == 5


class TestOptimizationResultPersistence:
    """Test OptimizationResultPersistence class."""

    def test_initialization(self, tmp_path):
        """Test OptimizationResultPersistence initialization."""
        persistence = OptimizationResultPersistence(base_dir=tmp_path)

        assert persistence.base_dir == Path(tmp_path)
        assert hasattr(persistence, 'logger')

    def test_save_and_load_result(self, tmp_path):
        """Test saving and loading optimization results."""
        persistence = OptimizationResultPersistence(base_dir=tmp_path)

        result = {
            "best_params": {"x": 1.0},
            "best_score": -0.5,
            "optimization_history": [{"trial": 1}],
            "execution_time": 1.0,
            "convergence_info": {},
            "recommendations": []
        }

        version_id = persistence.save_optimization_result(result, "test_experiment", {"test": "data"}, ["tag1"])

        assert version_id is not None
        assert version_id.startswith("v")

        loaded = persistence.load_optimization_result(version_id)
        assert loaded is not None
        assert loaded['result']['best_score'] == -0.5

    def test_search_results(self, tmp_path):
        """Test searching optimization results."""
        persistence = OptimizationResultPersistence(base_dir=tmp_path)

        result = {
            "best_params": {"x": 1.0},
            "best_score": -0.5,
            "optimization_history": [{"trial": 1}],
            "execution_time": 1.0,
            "convergence_info": {},
            "recommendations": []
        }

        persistence.save_optimization_result(result, "test_experiment", {"test": "data"}, ["tag1"])

        results = persistence.search_results(tags=["tag1"])
        assert len(results) > 0


class TestParallelOptimizer:
    """Test ParallelOptimizer class."""

    def test_initialization(self):
        """Test ParallelOptimizer initialization."""
        config = OptimizationConfig(max_parallel_trials=2)
        optimizer = ParallelOptimizer(config)

        assert optimizer.config == config
        assert optimizer.max_workers == 2

    def test_run_parallel_optimization(self):
        """Test parallel optimization execution."""
        config = OptimizationConfig(max_trials=1, max_parallel_trials=2)
        objective = make_scalar_objective("x")
        search_space = make_scalar_search_space("x")

        tasks = [
            {
                'task_id': 'task1',
                'optimizer': _StubOptimizer("x", value=0.1),
                'objective': objective,
                'search_space': search_space
            },
            {
                'task_id': 'task2',
                'optimizer': _StubOptimizer("x", value=0.2),
                'objective': objective,
                'search_space': search_space
            }
        ]

        optimizer = ParallelOptimizer(config)
        result = optimizer.run_parallel_optimization(tasks)

        assert result['total_tasks'] == 2
        assert result['completed_tasks'] == 2
        assert 'results' in result

    def test_task_specific_max_trials_override_is_restored(self):
        """Per-task max_trials should apply only to that task execution."""
        config = OptimizationConfig(max_trials=9, max_parallel_trials=1)

        class RecordingOptimizer:
            def __init__(self):
                self.config = OptimizationConfig(max_trials=9)
                self.observed_max_trials = []

            def optimize(self, objective, search_space):
                self.observed_max_trials.append(self.config.max_trials)
                return OptimizationResult(
                    best_params={},
                    best_score=objective({}),
                    optimization_history=[],
                    execution_time=0.0,
                    convergence_info={},
                    recommendations=[],
                )

        recording_optimizer = RecordingOptimizer()
        optimizer = ParallelOptimizer(config)
        result = optimizer._execute_optimization_task(
            {
                "task_id": "recording",
                "optimizer": recording_optimizer,
                "objective": lambda params: 0.0,
                "search_space": {},
                "max_trials": 1,
            }
        )

        assert result["success"] is True
        assert recording_optimizer.observed_max_trials == [1]
        assert recording_optimizer.config.max_trials == 9


class TestAutomaticOptimizationPipeline:
    """Test AutomaticOptimizationPipeline class."""

    def test_initialization(self):
        """Test AutomaticOptimizationPipeline initialization."""
        config = OptimizationConfig()
        pipeline = AutomaticOptimizationPipeline(config)

        assert pipeline.config == config
        assert hasattr(pipeline, 'logger')

    def test_run_pipeline(self):
        """Test running optimization pipeline."""
        config = OptimizationConfig(max_trials=2)
        objective = make_scalar_objective("x")
        search_space = make_scalar_search_space("x")

        pipeline = AutomaticOptimizationPipeline(config)
        # Skip system optimization stage for testing
        pipeline.stages = ["hyperparameter_optimization"]
        result = pipeline.run_full_pipeline({}, objective, search_space)

        assert 'success' in result
        assert 'stages' in result
        assert 'hyperparameter_optimization' in result['stages']


class TestUnifiedOptimizer:
    """Test UnifiedOptimizer class."""

    def test_initialization(self):
        """Test UnifiedOptimizer initialization."""
        config = OptimizationConfig()
        optimizer = UnifiedOptimizer(config)

        assert optimizer.config == config
        assert hasattr(optimizer, 'hyperparameter_optimizer')
        assert hasattr(optimizer, 'multi_timeframe_optimizer')
        assert hasattr(optimizer, 'ab_testing_framework')
        assert hasattr(optimizer, 'automatic_pipeline')
        assert hasattr(optimizer, 'persistence')
        assert hasattr(optimizer, 'parallel_optimizer')

    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization."""
        config = OptimizationConfig(max_trials=2)
        objective = make_scalar_objective("x")
        search_space = make_scalar_search_space("x")

        optimizer = UnifiedOptimizer(config)
        result = optimizer.optimize_hyperparameters(objective, search_space)

        assert isinstance(result, OptimizationResult)

    def test_optimize_multi_timeframe(self):
        """Test multi-timeframe optimization."""
        config = OptimizationConfig(max_trials=1)
        objective_functions = make_timeframe_objectives(["1m", "5m", "15m"], param_name="param")
        search_spaces = make_timeframe_search_spaces(["1m", "5m", "15m"], param_name="param")

        optimizer = UnifiedOptimizer(config)
        optimizer.multi_timeframe_optimizer.timeframe_optimizers = {
            timeframe: _StubOptimizer("param", value=0.25)
            for timeframe in optimizer.multi_timeframe_optimizer.timeframes
        }
        result = optimizer.optimize_multi_timeframe(objective_functions, search_spaces)

        assert 'integrated' in result

    def test_create_and_run_ab_test(self):
        """Test A/B testing functionality."""
        config = OptimizationConfig()

        optimizer = UnifiedOptimizer(config)
        test_id = optimizer.create_ab_test(
            'test_experiment',
            {'param': 1.0},
            {'param': 2.0},
            lambda p: p.get('param', 0)
        )

        result = optimizer.run_ab_test(test_id, num_iterations=5)

        assert 'status' in result

    def test_run_parallel_optimization(self):
        """Test parallel optimization."""
        config = OptimizationConfig(max_trials=1, max_parallel_trials=2)
        objective = make_scalar_objective("x")
        search_space = make_scalar_search_space("x")

        tasks = [
            {
                'task_id': 'task1',
                'optimizer': _StubOptimizer("x", value=0.1),
                'objective': objective,
                'search_space': search_space
            }
        ]

        optimizer = UnifiedOptimizer(config)
        result = optimizer.run_parallel_optimization(tasks)

        assert result['total_tasks'] == 1
        assert result['completed_tasks'] == 1

    def test_save_and_load_result(self, tmp_path):
        """Test result persistence."""
        config = OptimizationConfig()

        result_dict = {
            "best_params": {"x": 1.0},
            "best_score": -0.5,
            "optimization_history": [{"trial": 1}],
            "execution_time": 1.0,
            "convergence_info": {},
            "recommendations": []
        }

        optimizer = UnifiedOptimizer(config)
        optimizer.persistence.base_dir = Path(tmp_path)

        version_id = optimizer.save_result_to_version_control(
            result_dict, "test_experiment", {"test": "metadata"}, ["test"]
        )

        assert version_id is not None

        loaded = optimizer.load_result_from_version_control(version_id)
        assert loaded is not None
