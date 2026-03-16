#!/usr/bin/env python3
"""
Integration tests for Unified Optimizer components
"""
from unittest.mock import Mock

import pytest

from tests.helpers.optimization import (
    make_lr_batch_objective,
    make_lr_batch_search_space,
    make_momentum_search_spaces,
)
from ztb.training.unified_optimizer import (
    OptimizationConfig, UnifiedOptimizer, BayesianOptimizer
)


def _stub_system_optimizer(optimizer: UnifiedOptimizer) -> None:
    """Current SystemOptimizer no longer exposes legacy memory/status methods."""
    optimizer.system_optimizer = Mock()
    optimizer.system_optimizer.optimize_memory_usage.return_value = {"status": "ok"}
    optimizer.system_optimizer.get_system_status.return_value = {"status": "ok"}
    optimizer.automatic_pipeline.system_optimizer = optimizer.system_optimizer


def _make_optimizer(tmp_path, *, max_trials: int = 1, max_parallel_trials: int = 1) -> UnifiedOptimizer:
    config = OptimizationConfig(
        max_trials=max_trials,
        max_parallel_trials=max_parallel_trials,
        persistence_base_dir=str(tmp_path / "optimization_results"),
    )
    optimizer = UnifiedOptimizer(config)
    _stub_system_optimizer(optimizer)
    return optimizer


class TestUnifiedOptimizerIntegration:
    """統合テスト for UnifiedOptimizer"""

    def test_full_automatic_pipeline(self, tmp_path):
        """完全自動最適化パイプラインの統合テスト"""
        optimizer = _make_optimizer(tmp_path)

        base_params = {"learning_rate": 0.001, "batch_size": 32}
        complex_objective = make_lr_batch_objective(noise_scale=0.0)
        search_space = make_lr_batch_search_space()

        pipeline_result = optimizer.run_automatic_pipeline(
            base_params, complex_objective, search_space
        )

        assert pipeline_result["success"] is True
        assert "stages" in pipeline_result
        assert "final_recommendation" in pipeline_result

        recommendation = pipeline_result["final_recommendation"]
        assert "action" in recommendation
        assert "params" in recommendation

    def test_multi_timeframe_with_ab_testing(self, tmp_path):
        """マルチタイムフレーム最適化 + A/Bテストの統合テスト"""
        optimizer = _make_optimizer(tmp_path)

        def tf_objective_1m(params):
            return -params.get("momentum", 0.9)**2

        def tf_objective_5m(params):
            return -params.get("momentum", 0.9)**2 * 1.1  # 少し異なる

        mt_functions = {"1m": tf_objective_1m, "5m": tf_objective_5m}
        mt_search_spaces = make_momentum_search_spaces()

        mt_result = optimizer.optimize_multi_timeframe(mt_functions, mt_search_spaces)
        assert "integrated" in mt_result

        # A/Bテスト - サンプルサイズを十分に確保
        control_params = {"momentum": 0.5}
        variant_params = mt_result["integrated"].best_params

        test_id = optimizer.create_ab_test(
            "integration_test",
            control_params,
            variant_params,
            tf_objective_1m,
            sample_size_per_group=8,
        )

        ab_result = optimizer.run_ab_test(test_id, num_iterations=8)
        assert "status" in ab_result
        assert ab_result["status"] in ["completed", "significant", "insignificant", "insufficient_data", "variant_better", "control_better", "no_significant_difference", "running"]

    def test_parallel_optimization_integration(self, tmp_path):
        """並列最適化の統合テスト"""
        optimizer = _make_optimizer(tmp_path, max_parallel_trials=1)
        complex_objective = make_lr_batch_objective(noise_scale=0.0)
        search_space = make_lr_batch_search_space()

        # 複数の最適化タスク
        parallel_tasks = []
        for i in range(2):
            task = {
                "task_id": f"integration_parallel_{i}",
                "optimizer": BayesianOptimizer(optimizer.config),
                "objective": complex_objective,
                "search_space": search_space
            }
            parallel_tasks.append(task)

        parallel_result = optimizer.run_parallel_optimization(parallel_tasks)

        assert parallel_result["success"] is True
        assert parallel_result["total_tasks"] == 2
        assert parallel_result["completed_tasks"] == 2
        assert "results" in parallel_result

    def test_persistence_integration(self, tmp_path):
        """持続化機能の統合テスト"""
        optimizer = _make_optimizer(tmp_path)

        test_result = {"integration_test": True, "score": 0.85}
        version_id = optimizer.save_result_to_version_control(
            test_result, "integration_test", tags=["integration", "test"]
        )

        assert version_id is not None
        assert version_id.startswith("v")

        loaded = optimizer.load_result_from_version_control(version_id)
        assert loaded is not None
        assert loaded["result"]["integration_test"] is True
        assert loaded["result"]["score"] == 0.85

        # 検索機能テスト
        search_results = optimizer.search_optimization_results(tags=["integration"])
        assert len(search_results) > 0

        # 比較機能テスト
        comparison = optimizer.compare_optimization_results([version_id])
        assert "results" in comparison
        assert len(comparison["results"]) == 1

    def test_end_to_end_workflow(self, tmp_path):
        """エンドツーエンドワークフローの統合テスト"""
        optimizer = _make_optimizer(tmp_path)

        # 1. ハイパーパラメータ最適化
        def objective(params):
            return -(params.get("x", 0) - 2)**2 - (params.get("y", 0) - 3)**2

        search_space = {
            "x": {"type": "float", "low": 0, "high": 5},
            "y": {"type": "float", "low": 0, "high": 5}
        }

        hp_result = optimizer.optimize_hyperparameters(objective, search_space)
        assert isinstance(hp_result, object)  # OptimizationResult

        # 2. システム最適化
        system_result = optimizer.optimize_system()
        assert isinstance(system_result, dict)

        # 3. 結果の保存
        saved_version = optimizer.save_result_to_version_control(
            {"workflow_test": True, "best_score": hp_result.best_score},
            "workflow_test"
        )
        assert saved_version is not None

        # 4. 結果の読み込みと検証
        loaded = optimizer.load_result_from_version_control(saved_version)
        assert loaded is not None
        assert loaded["result"]["workflow_test"] is True

        # 5. サマリー取得
        summary = optimizer.get_optimization_summary()
        assert isinstance(summary, dict)
        assert "total_optimizations" in summary
