"""
Phase 3 ABTestingFramework統合テスト
既存クラス（ResultComparator, StatisticalValidator）の統合検証
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from ztb.training.unified_optimizer import ABTestingFramework, OptimizationConfig


@pytest.fixture
def ab_framework():
    """ABTestingFrameworkインスタンス"""
    config = OptimizationConfig()
    return ABTestingFramework(config)


@pytest.fixture
def sample_scores():
    """テスト用スコアデータ"""
    np.random.seed(42)
    return {
        "control": np.random.normal(0.5, 0.1, 50).tolist(),
        "variant_better": np.random.normal(0.6, 0.1, 50).tolist(),  # 有意に良い
        "variant_similar": np.random.normal(0.51, 0.1, 50).tolist(),  # 有意差なし
    }


class TestABTestingPhase3Integration:
    """Phase 3統合テスト: 既存クラス活用の検証"""
    
    def test_framework_initialization(self, ab_framework):
        """既存クラスのインスタンス化確認"""
        # ResultComparator, StatisticalValidatorが正しくインスタンス化されているか
        assert hasattr(ab_framework, 'result_comparator')
        assert hasattr(ab_framework, 'statistical_validator')
        assert hasattr(ab_framework, 'p_mean_method')
        
        # 既存設定が維持されているか
        assert ab_framework.confidence_level == 0.95
        assert ab_framework.min_sample_size == 30
    
    def test_significance_test_with_result_comparator(self, ab_framework, sample_scores):
        """ResultComparator統合: Mann-Whitney U + t-test"""
        test_config = {
            "control_results": [{"score": s} for s in sample_scores["control"]],
            "variant_results": [{"score": s} for s in sample_scores["variant_better"]],
            "n_splits": 4
        }
        
        result = ab_framework._perform_significance_test(test_config)
        
        # 基本構造確認
        assert result["test_performed"] is True
        assert "t_statistic" in result
        assert "p_value" in result
        assert "is_significant" in result
        assert "effect_size" in result
        
        # Phase 3拡張確認
        assert "statistical_tests" in result
        assert "p_mean" in result
        assert "combined_decision" in result
        
        # ResultComparatorの結果確認
        assert "t_test" in result["statistical_tests"]
        assert "mann_whitney" in result["statistical_tests"]
        assert "levene" in result["statistical_tests"]
        
        # 有意差が検出されるはず（variant_betterは0.6, controlは0.5）
        assert result["is_significant"] == True  # numpy.True_対応
    
    def test_p_mean_method_integration(self, ab_framework, sample_scores):
        """p平均法統合: 既存p_mean_method活用"""
        result = ab_framework._perform_p_mean_method(
            sample_scores["control"],
            sample_scores["variant_better"],
            n_splits=4
        )
        
        assert result["success"] is True
        assert result["n_splits"] == 4
        assert len(result["p_values"]) == 4
        assert "p_mean_geometric" in result
        assert "p_mean_arithmetic" in result
        assert "is_significant" in result
        assert result["method"] == "richmanbtc_original"
    
    def test_combined_decision_logic(self, ab_framework):
        """三位一体検定の統合判定"""
        # 3つ全て有意
        decision = ab_framework._make_combined_decision(
            {"p_value": 0.01},
            {"p_value": 0.02},
            {"success": True, "is_significant": True}
        )
        assert decision["significant_count"] == 3
        assert decision["evidence_strength"] == "strong"
        
        # 2つ有意
        decision = ab_framework._make_combined_decision(
            {"p_value": 0.01},
            {"p_value": 0.10},  # 有意でない
            {"success": True, "is_significant": True}
        )
        assert decision["significant_count"] == 2
        assert decision["evidence_strength"] == "moderate"
        
        # 1つ有意
        decision = ab_framework._make_combined_decision(
            {"p_value": 0.01},
            {"p_value": 0.10},
            {"success": True, "is_significant": False}
        )
        assert decision["significant_count"] == 1
        assert decision["evidence_strength"] == "weak"
        
        # 0個有意
        decision = ab_framework._make_combined_decision(
            {"p_value": 0.10},
            {"p_value": 0.20},
            {"success": True, "is_significant": False}
        )
        assert decision["significant_count"] == 0
        assert decision["evidence_strength"] == "none"
    
    def test_compare_multiple_conditions(self, ab_framework, sample_scores):
        """複数条件比較: StatisticalValidator統合"""
        conditions = ["control", "variant_better", "variant_similar"]
        metric_results = {
            "control": sample_scores["control"],
            "variant_better": sample_scores["variant_better"],
            "variant_similar": sample_scores["variant_similar"]
        }
        
        result = ab_framework.compare_multiple_conditions(
            conditions, metric_results, alpha=0.05
        )
        
        assert result["success"] is True
        assert result["n_comparisons"] == 3  # C(3,2) = 3ペア
        assert "pairwise_results" in result
        assert "correction" in result
        assert result["method"] == "holm_bonferroni"
        
        # StatisticalValidatorの補正結果確認
        assert "adjusted_p_values" in result["correction"]
        assert "rejected" in result["correction"]
        
        # control vs variant_betterは有意差ありのはず
        assert result["significant_pairs"] >= 1
    
    def test_no_code_duplication_verification(self, ab_framework):
        """コード重複なし確認: 統計ロジックは既存クラスのみ"""
        import inspect
        
        # _perform_significance_testのソースコード取得
        source = inspect.getsource(ab_framework._perform_significance_test)
        
        # 既存クラスを呼び出しているか確認
        assert "self.result_comparator._run_statistical_tests" in source
        assert "_perform_p_mean_method" in source  # メソッド呼び出し確認
        
        # Mann-Whitney Uの直接実装がないことを確認（既存クラスを使うべき）
        # Note: _perform_p_mean_method内でmannwhitneyuを使うのはOK（p平均法の一部）
        
        # compare_multiple_conditionsも同様
        source2 = inspect.getsource(ab_framework.compare_multiple_conditions)
        assert "self.statistical_validator._apply_multiple_testing_correction" in source2
    
    def test_backward_compatibility(self, ab_framework, sample_scores):
        """既存API互換性確認"""
        # create_ab_test, run_ab_testが既存通り動作するか
        test_id = ab_framework.create_ab_test(
            test_name="phase3_compat_test",
            control_params={"condition": "control"},
            variant_params={"condition": "variant"},
            evaluation_function=lambda p: 0.5 if p["condition"] == "control" else 0.6,
            sample_size_per_group=50
        )
        
        assert test_id.startswith("phase3_compat_test_")
        assert test_id in ab_framework.active_tests
        
        # run_ab_testで三位一体検定が動作するか
        result = ab_framework.run_ab_test(test_id, num_iterations=50)
        
        assert "significance_test" in result
        assert result["significance_test"]["test_performed"] is True
        
        # Phase 3拡張が含まれているか
        assert "statistical_tests" in result["significance_test"]
        assert "p_mean" in result["significance_test"]
        assert "combined_decision" in result["significance_test"]


class TestEdgeCases:
    """エッジケーステスト"""
    
    def test_insufficient_samples_for_p_mean(self, ab_framework):
        """p平均法: サンプル数不足"""
        result = ab_framework._perform_p_mean_method(
            [0.1, 0.2, 0.3],  # 3サンプル（4分割不可）
            [0.2, 0.3, 0.4],
            n_splits=4
        )
        
        assert result["success"] is False
        assert "Insufficient data" in result["reason"]
    
    def test_empty_conditions_list(self, ab_framework):
        """複数条件比較: 空のリスト"""
        result = ab_framework.compare_multiple_conditions(
            [],
            {},
            alpha=0.05
        )
        
        assert result["success"] is False
    
    def test_single_condition(self, ab_framework, sample_scores):
        """複数条件比較: 条件1つ（ペアなし）"""
        result = ab_framework.compare_multiple_conditions(
            ["control"],
            {"control": sample_scores["control"]},
            alpha=0.05
        )
        
        # ペアが0個なので失敗するはず
        assert result["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
