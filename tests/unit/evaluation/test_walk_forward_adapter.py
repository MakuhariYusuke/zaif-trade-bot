"""
Walk-Forward統合アダプターのテスト

WindowPerformanceからComprehensiveEvaluationへの変換と
集約統計の計算をテストする。
"""


import pytest

from ztb.analysis.common.types import ComprehensiveEvaluationClass
from ztb.analysis.evaluation.walk_forward_adapter import (
    WalkForwardAggregationStats,
    WalkForwardUnifiedEvaluator,
)
from ztb.evaluation.walk_forward.types import WindowPerformance


class TestWalkForwardUnifiedEvaluator:
    """Walk-Forward統合評価器のテスト"""

    @pytest.fixture
    def evaluator(self) -> WalkForwardUnifiedEvaluator:
        """評価器のセットアップ"""
        return WalkForwardUnifiedEvaluator()

    @pytest.fixture
    def sample_windows(self) -> list[WindowPerformance]:
        """サンプルウィンドウデータ"""
        return [
            WindowPerformance(
                window_id=0,
                val_roi=0.0523,
                test_roi=0.0419,
                val_final_balance=1052300.0,
                test_final_balance=1041900.0,
                sharpe_ratio=1.25,
                max_drawdown=-0.082,
                win_rate=0.65,
                trades=42,
            ),
            WindowPerformance(
                window_id=1,
                val_roi=0.0451,
                test_roi=0.0387,
                val_final_balance=1045100.0,
                test_final_balance=1038700.0,
                sharpe_ratio=1.15,
                max_drawdown=-0.095,
                win_rate=0.62,
                trades=38,
            ),
            WindowPerformance(
                window_id=2,
                val_roi=0.0589,
                test_roi=0.0512,
                val_final_balance=1058900.0,
                test_final_balance=1051200.0,
                sharpe_ratio=1.35,
                max_drawdown=-0.070,
                win_rate=0.68,
                trades=45,
            ),
        ]

    def test_aggregate_windows_produces_comprehensive_evaluation(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """ウィンドウ集約がComprehensiveEvaluationを生成する"""

        # 実行
        result = evaluator.aggregate_windows(
            windows=sample_windows,
            model_name="test_model",
        )

        # 検証
        assert isinstance(result, ComprehensiveEvaluationClass)
        assert result.model_name == "test_model"
        assert result.evaluation_type == "walk_forward"
        assert result.results is not None
        assert len(result.results) > 0

    def test_overfitting_detection(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """過学習検出が機能する"""

        result = evaluator.aggregate_windows(
            windows=sample_windows,
            model_name="test_model",
        )

        # 結果に過学習指標が含まれている
        assert "robustness_tests" in result.__dict__
        robustness = result.robustness_tests
        assert "overfitting_indicator" in robustness
        assert "overfitting_severity" in robustness

        # 過学習指標が計算されている
        overfitting_indicator = robustness["overfitting_indicator"]
        assert isinstance(overfitting_indicator, (int, float))
        assert 0.0 <= overfitting_indicator <= 2.0  # 合理的な範囲

    def test_consistency_score_calculation(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """一貫性スコアが計算される"""

        result = evaluator.aggregate_windows(windows=sample_windows, model_name="test")

        # 一貫性スコアが存在
        performance = result.performance_metrics
        assert "consistency_score" in performance

        # スコアが0-1の範囲
        consistency = performance["consistency_score"]
        assert isinstance(consistency, (int, float))
        assert 0.0 <= consistency <= 1.0

    def test_robustness_score_calculation(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """ロバストネススコアが計算される"""

        result = evaluator.aggregate_windows(windows=sample_windows, model_name="test")

        # ロバストネススコアが存在
        performance = result.performance_metrics
        assert "robustness_score" in performance

        # スコアが0-1の範囲
        robustness = performance["robustness_score"]
        assert isinstance(robustness, (int, float))
        assert 0.0 <= robustness <= 1.0

    def test_aggregate_windows_rejects_empty_list(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
    ) -> None:
        """空リストの場合はValueErrorを発生"""

        with pytest.raises(ValueError, match="Windows list cannot be empty"):
            evaluator.aggregate_windows(windows=[], model_name="test")

    def test_cross_window_statistics(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """ウィンドウ横断的統計が計算される"""

        stats = evaluator._analyze_cross_window_stats(sample_windows)

        assert isinstance(stats, WalkForwardAggregationStats)
        assert stats.window_count == 3
        assert stats.avg_val_roi > 0
        assert stats.avg_test_roi > 0
        assert stats.std_val_roi >= 0
        assert stats.std_test_roi >= 0

    def test_overfitting_severity_determination(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
    ) -> None:
        """過学習の重大度が正しく判定される"""

        # 過学習なし
        assert evaluator._determine_overfitting_severity(0.5) == "none"

        # 軽度
        assert evaluator._determine_overfitting_severity(0.9) == "mild"

        # 中程度
        assert evaluator._determine_overfitting_severity(1.1) == "moderate"

        # 深刻
        assert evaluator._determine_overfitting_severity(1.5) == "severe"

    def test_metric_aggregation_includes_all_required_metrics(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """メトリクス集約に必須メトリクスが全て含まれる"""

        result = evaluator.aggregate_windows(windows=sample_windows, model_name="test")

        required_metrics = {
            "roi_in_sample",
            "roi_out_of_sample",
            "max_drawdown",
            "sharpe_ratio",
            "overfitting_indicator",
            "consistency_score",
            "robustness_score",
            "stability_index",
            "win_rate",
        }

        actual_metrics = set(result.results.keys())

        assert required_metrics.issubset(
            actual_metrics
        ), f"Missing metrics: {required_metrics - actual_metrics}"

    def test_compare_multiple_evaluations(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """複数モデルの比較が機能する"""

        # 複数の評価結果を生成
        eval1 = evaluator.aggregate_windows(sample_windows, "model_a")
        eval2 = evaluator.aggregate_windows(sample_windows, "model_b")

        comparison = evaluator.compare_multiple_evaluations(
            {
                "model_a": eval1,
                "model_b": eval2,
            }
        )

        assert comparison["model_count"] == 2
        assert "models" in comparison
        assert "rankings" in comparison
        assert "model_a" in comparison["models"]
        assert "model_b" in comparison["models"]

    def test_rankings_generation(
        self,
        evaluator: WalkForwardUnifiedEvaluator,
        sample_windows: list[WindowPerformance],
    ) -> None:
        """ランキング生成が機能する"""

        eval1 = evaluator.aggregate_windows(sample_windows, "model_a")
        eval2 = evaluator.aggregate_windows(sample_windows, "model_b")

        comparison = evaluator.compare_multiple_evaluations(
            {
                "model_a": eval1,
                "model_b": eval2,
            }
        )

        rankings = comparison["rankings"]

        # ランキングメトリクスが存在
        assert "roi_out_of_sample" in rankings
        assert "max_drawdown" in rankings
        assert "robustness_score" in rankings
        assert "overfitting_indicator" in rankings

        # 各ランキングに全モデルが含まれている
        for metric, models in rankings.items():
            assert len(models) == 2
            assert set(models) == {"model_a", "model_b"}


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_zero_roi_windows(self) -> None:
        """ROIが0のウィンドウでも処理できる"""

        evaluator = WalkForwardUnifiedEvaluator()

        windows = [
            WindowPerformance(window_id=0, val_roi=0.0, test_roi=0.0),
            WindowPerformance(window_id=1, val_roi=0.0, test_roi=0.0),
        ]

        # エラーなく実行できる
        result = evaluator.aggregate_windows(windows, "zero_roi_model")

        assert result is not None
        assert result.evaluation_type == "walk_forward"

    def test_negative_roi_windows(self) -> None:
        """負のROIのウィンドウで処理できる"""

        evaluator = WalkForwardUnifiedEvaluator()

        windows = [
            WindowPerformance(window_id=0, val_roi=-0.05, test_roi=-0.08),
            WindowPerformance(window_id=1, val_roi=-0.03, test_roi=-0.07),
        ]

        result = evaluator.aggregate_windows(windows, "losing_model")

        assert result is not None
        # 負のROIでも処理可能
        roi_result = result.results.get("roi_in_sample")
        assert roi_result is not None
        if isinstance(roi_result, dict):
            assert roi_result.get("value", 0) < 0
        else:
            assert roi_result.value < 0

    def test_single_window(self) -> None:
        """単一ウィンドウでの処理"""

        evaluator = WalkForwardUnifiedEvaluator()

        windows = [
            WindowPerformance(
                window_id=0,
                val_roi=0.05,
                test_roi=0.04,
                sharpe_ratio=1.0,
                max_drawdown=-0.1,
                win_rate=0.6,
            ),
        ]

        result = evaluator.aggregate_windows(windows, "single_window")

        assert result is not None
        assert result.summary_stats["window_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
