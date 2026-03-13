"""
WalkForwardEvaluationPipelineのテスト
"""

import pytest
from pathlib import Path
import json
import tempfile

from ztb.evaluation.walk_forward.types import WindowPerformance
from ztb.analysis.evaluation.walk_forward_integration_pipeline import WalkForwardEvaluationPipeline


class TestWalkForwardEvaluationPipeline:
    """Walk-Forward評価パイプラインのテスト"""
    
    @pytest.fixture
    def sample_windows(self) -> list[WindowPerformance]:
        """テスト用のサンプルウィンドウ"""
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
        ]
    
    @pytest.fixture
    def pipeline(self, tmp_path) -> WalkForwardEvaluationPipeline:
        """テスト用のパイプライン"""
        return WalkForwardEvaluationPipeline(
            model_name="test_model",
            output_dir=str(tmp_path / "results"),
        )
    
    def test_pipeline_initialization(self, pipeline) -> None:
        """パイプラインが正しく初期化される"""
        assert pipeline.model_name == "test_model"
        assert pipeline.evaluation_result is None
        assert pipeline.output_dir.exists()
    
    def test_integrate_walk_forward_results(self, pipeline, sample_windows) -> None:
        """Walk-Forward結果の統合が機能する"""
        evaluation = pipeline.integrate_walk_forward_results(
            windows=sample_windows,
            model_metadata={"version": "v456"},
        )
        
        assert evaluation is not None
        assert evaluation.model_name == "test_model"
        assert evaluation.evaluation_type == "walk_forward"
        assert len(evaluation.results) > 0
    
    def test_save_evaluation(self, pipeline, sample_windows) -> None:
        """評価結果の保存が機能する"""
        pipeline.integrate_walk_forward_results(
            windows=sample_windows,
        )
        
        output_path = pipeline.save_evaluation(format="json")
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # JSONの内容を検証
        with open(output_path, "r") as f:
            data = json.load(f)
        
        assert "model_name" in data
        assert "evaluation_type" in data
        assert data["evaluation_type"] == "walk_forward"
    
    def test_generate_summary_report(self, pipeline, sample_windows) -> None:
        """サマリーレポート生成が機能する"""
        pipeline.integrate_walk_forward_results(
            windows=sample_windows,
        )
        
        report = pipeline.generate_summary_report()
        
        assert isinstance(report, str)
        assert "Walk-Forward統合評価レポート" in report
        assert "モデル名" in report
        assert "パフォーマンス" in report
        assert "堅牢性評価" in report
        assert "過学習検出" in report
    
    def test_generate_full_report(self, pipeline, sample_windows) -> None:
        """完全レポート生成が機能する"""
        pipeline.integrate_walk_forward_results(
            windows=sample_windows,
        )
        
        report = pipeline.generate_full_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "詳細統計" in report
    
    def test_recommendations_generated(self, pipeline, sample_windows) -> None:
        """推奨事項が生成される"""
        pipeline.integrate_walk_forward_results(
            windows=sample_windows,
        )
        
        report = pipeline.generate_summary_report()
        
        # 推奨事項セクションが存在
        assert "推奨事項" in report
    
    def test_compare_with_baseline(self, pipeline, sample_windows) -> None:
        """ベースラインとの比較が機能する"""
        eval1 = pipeline.integrate_walk_forward_results(
            windows=sample_windows,
        )
        
        # 同じデータでbaseline用の評価を作成
        pipeline2 = WalkForwardEvaluationPipeline(
            model_name="baseline_model",
            output_dir=str(Path(pipeline.output_dir).parent / "baseline"),
        )
        
        baseline_eval = pipeline2.integrate_walk_forward_results(
            windows=sample_windows,
        )
        
        # 比較実行
        comparison = pipeline.compare_with_baseline(baseline_eval)
        
        assert "model" in comparison
        assert "baseline" in comparison
        assert "metrics" in comparison
        assert len(comparison["metrics"]) > 0
    
    def test_pipeline_without_metadata(self, pipeline, sample_windows) -> None:
        """メタデータなしでも動作する"""
        evaluation = pipeline.integrate_walk_forward_results(
            windows=sample_windows,
            model_metadata=None,
        )
        
        assert evaluation is not None
    
    def test_error_on_missing_evaluation(self, pipeline) -> None:
        """評価結果なしでレポート生成すると例外が発生"""
        with pytest.raises(ValueError, match="Evaluation result not available"):
            pipeline.generate_summary_report()
    
    def test_error_on_missing_evaluation_save(self, pipeline) -> None:
        """評価結果なしで保存しようとすると例外が発生"""
        with pytest.raises(ValueError, match="Evaluation result not available"):
            pipeline.save_evaluation()


class TestRecommendationGeneration:
    """推奨事項生成ロジックのテスト"""
    
    def test_excellent_performance_recommendation(self, tmp_path) -> None:
        """優秀な性能の推奨事項"""
        pipeline = WalkForwardEvaluationPipeline(
            model_name="excellent",
            output_dir=str(tmp_path),
        )
        
        recs = pipeline._generate_recommendations(
            roi_in=0.10,
            roi_out=0.12,
            robustness=0.9,
            overfitting=0.05,
        )
        
        assert any("優秀な性能" in rec for rec in recs)
    
    def test_overfitting_severe_recommendation(self, tmp_path) -> None:
        """深刻な過学習の推奨事項"""
        pipeline = WalkForwardEvaluationPipeline(
            model_name="overfit",
            output_dir=str(tmp_path),
        )
        
        recs = pipeline._generate_recommendations(
            roi_in=0.10,
            roi_out=-0.01,
            robustness=0.3,
            overfitting=1.5,
        )
        
        assert any("深刻な過学習" in rec for rec in recs)
    
    def test_roi_diff_recommendation(self, tmp_path) -> None:
        """ROI差分の推奨事項"""
        pipeline = WalkForwardEvaluationPipeline(
            model_name="roi_diff",
            output_dir=str(tmp_path),
        )
        
        recs = pipeline._generate_recommendations(
            roi_in=0.15,
            roi_out=0.08,
            robustness=0.7,
            overfitting=0.47,
        )
        
        assert any("差が大きい" in rec for rec in recs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
