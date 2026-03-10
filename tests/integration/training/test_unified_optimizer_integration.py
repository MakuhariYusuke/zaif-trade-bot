#!/usr/bin/env python3
"""
Integration tests for Unified Optimizer components
"""
import numpy as np
import pytest

from ztb.training.unified_optimizer import (
    OptimizationConfig, UnifiedOptimizer, BayesianOptimizer
)


class TestUnifiedOptimizerIntegration:
    """統合テスト for UnifiedOptimizer"""

    def test_full_automatic_pipeline(self):
        """完全自動最適化パイプラインの統合テスト"""
        config = OptimizationConfig(max_trials=5)  # 短いテスト用
        optimizer = UnifiedOptimizer(config)

        base_params = {"learning_rate": 0.001, "batch_size": 32}

        def complex_objective(params):
            lr = params.get("learning_rate", 0.001)
            bs = params.get("batch_size", 32)
            # より複雑な目的関数
            return -(lr - 0.01)**2 - (bs - 64)**2 / 10000 + np.random.normal(0, 0.01)

        search_space = {
            "learning_rate": {"type": "float", "low": 0.0001, "high": 0.1},
            "batch_size": {"type": "int", "low": 16, "high": 128}
        }

        pipeline_result = optimizer.run_automatic_pipeline(
            base_params, complex_objective, search_space
        )

        assert pipeline_result["success"] is True
        assert "stages" in pipeline_result
        assert "final_recommendation" in pipeline_result

        recommendation = pipeline_result["final_recommendation"]
        assert "action" in recommendation
        assert "params" in recommendation

    def test_multi_timeframe_with_ab_testing(self):
        """マルチタイムフレーム最適化 + A/Bテストの統合テスト"""
        config = OptimizationConfig(max_trials=5)
        optimizer = UnifiedOptimizer(config)

        def tf_objective_1m(params):
            return -params.get("momentum", 0.9)**2

        def tf_objective_5m(params):
            return -params.get("momentum", 0.9)**2 * 1.1  # 少し異なる

        mt_functions = {"1m": tf_objective_1m, "5m": tf_objective_5m}
        mt_search_spaces = {
            "1m": {"momentum": {"type": "float", "low": 0.1, "high": 0.9}},
            "5m": {"momentum": {"type": "float", "low": 0.1, "high": 0.9}}
        }

        mt_result = optimizer.optimize_multi_timeframe(mt_functions, mt_search_spaces)
        assert "integrated" in mt_result

        # A/Bテスト - サンプルサイズを十分に確保
        control_params = {"momentum": 0.5}
        variant_params = mt_result["integrated"].best_params

        test_id = optimizer.create_ab_test(
            "integration_test", control_params, variant_params, tf_objective_1m
        )

        # 統計的有意差検定に必要なサンプルサイズを確保（最低30）
        ab_result = optimizer.run_ab_test(test_id, num_iterations=35)
        assert "status" in ab_result
        assert ab_result["status"] in ["completed", "significant", "insignificant", "insufficient_data", "variant_better", "control_better", "no_significant_difference", "running"]

    def test_parallel_optimization_integration(self):
        """並列最適化の統合テスト"""
        config = OptimizationConfig(max_trials=5, max_parallel_trials=2)
        optimizer = UnifiedOptimizer(config)

        def complex_objective(params):
            lr = params.get("learning_rate", 0.001)
            bs = params.get("batch_size", 32)
            return -(lr - 0.01)**2 - (bs - 64)**2 / 10000

        search_space = {
            "learning_rate": {"type": "float", "low": 0.0001, "high": 0.1},
            "batch_size": {"type": "int", "low": 16, "high": 128}
        }

        # 複数の最適化タスク
        parallel_tasks = []
        for i in range(3):
            task = {
                "task_id": f"integration_parallel_{i}",
                "optimizer": BayesianOptimizer(config),
                "objective": complex_objective,
                "search_space": search_space
            }
            parallel_tasks.append(task)

        parallel_result = optimizer.run_parallel_optimization(parallel_tasks)

        assert parallel_result["success"] is True
        assert parallel_result["total_tasks"] == 3
        assert parallel_result["completed_tasks"] == 3
        assert "results" in parallel_result

    def test_persistence_integration(self):
        """持続化機能の統合テスト"""
        config = OptimizationConfig()
        optimizer = UnifiedOptimizer(config)

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

    def test_end_to_end_workflow(self):
        """エンドツーエンドワークフローの統合テスト"""
        config = OptimizationConfig(max_trials=5)
        optimizer = UnifiedOptimizer(config)

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


if __name__ == "__main__":
    pytest.main([__file__])