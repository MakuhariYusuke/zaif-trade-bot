"""
Unit tests for Unified Evaluation Framework

統合評価フレームワークの単体テスト
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd

from ztb.evaluation.unified_evaluation import (
    UnifiedEvaluator,
    ComprehensiveEvaluation,
    EvaluationResult,
    EvaluationMetric,
    EvaluationType
)


class TestUnifiedEvaluator(unittest.TestCase):
    """UnifiedEvaluatorクラスのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.evaluator = UnifiedEvaluator()

        # サンプルデータファイルの作成
        self.sample_data = {
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'returns': [0.01] * 50 + [-0.005] * 50,
            'price': [100 + i * 0.1 for i in range(100)]
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_unified_evaluator_creation(self):
        """UnifiedEvaluatorの作成テスト"""
        evaluator = UnifiedEvaluator()
        self.assertIsInstance(evaluator, UnifiedEvaluator)

    def test_unified_evaluator_creation_with_config(self):
        """設定付きUnifiedEvaluatorの作成テスト"""
        config = {"evaluation_threshold": 0.8}
        evaluator = UnifiedEvaluator(config)
        self.assertEqual(evaluator.config, config)

    @patch('pandas.read_csv')
    def test_evaluate_model_basic(self, mock_read_csv):
        """基本的なモデル評価テスト"""
        # モックデータの設定
        mock_read_csv.return_value = self.df

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_data_path = Path(f.name)

        # 評価実行
        result = self.evaluator.evaluate_model(
            model_path="dummy_model",
            data_path=temp_data_path,
            evaluation_type=EvaluationType.BACKTEST
        )

        # 検証
        self.assertIsInstance(result, ComprehensiveEvaluation)
        self.assertEqual(result.model_name, "dummy_model")
        self.assertEqual(result.evaluation_type, EvaluationType.BACKTEST)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertGreater(len(result.results), 0)

        temp_data_path.unlink()

    def test_evaluation_result_creation(self):
        """EvaluationResultの作成テスト"""
        result = EvaluationResult(
            metric=EvaluationMetric.SHARPE_RATIO,
            value=1.5,
            confidence_interval=(1.2, 1.8),
            benchmark_comparison=0.3
        )

        self.assertEqual(result.metric, EvaluationMetric.SHARPE_RATIO)
        self.assertEqual(result.value, 1.5)
        self.assertEqual(result.confidence_interval, (1.2, 1.8))
        self.assertEqual(result.benchmark_comparison, 0.3)

    def test_comprehensive_evaluation_creation(self):
        """ComprehensiveEvaluationの作成テスト"""
        timestamp = datetime.now()

        results = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5
            ),
            EvaluationMetric.MAX_DRAWDOWN: EvaluationResult(
                metric=EvaluationMetric.MAX_DRAWDOWN,
                value=0.15
            )
        }

        evaluation = ComprehensiveEvaluation(
            model_name="test_model",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=timestamp,
            results=results
        )

        self.assertEqual(evaluation.model_name, "test_model")
        self.assertEqual(evaluation.evaluation_type, EvaluationType.BACKTEST)
        self.assertEqual(evaluation.timestamp, timestamp)
        self.assertEqual(len(evaluation.results), 2)

    def test_comprehensive_evaluation_get_metric_value(self):
        """指標値取得テスト"""
        results = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5
            )
        }

        evaluation = ComprehensiveEvaluation(
            model_name="test_model",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=datetime.now(),
            results=results
        )

        self.assertEqual(evaluation.get_metric_value(EvaluationMetric.SHARPE_RATIO), 1.5)
        self.assertIsNone(evaluation.get_metric_value(EvaluationMetric.MAX_DRAWDOWN))

    def test_comprehensive_evaluation_get_summary_score(self):
        """サマリースコア計算テスト"""
        results = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5
            ),
            EvaluationMetric.SORTINO_RATIO: EvaluationResult(
                metric=EvaluationMetric.SORTINO_RATIO,
                value=1.2
            ),
            EvaluationMetric.CALMAR_RATIO: EvaluationResult(
                metric=EvaluationMetric.CALMAR_RATIO,
                value=1.8
            ),
            EvaluationMetric.MAX_DRAWDOWN: EvaluationResult(
                metric=EvaluationMetric.MAX_DRAWDOWN,
                value=0.15
            )
        }

        evaluation = ComprehensiveEvaluation(
            model_name="test_model",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=datetime.now(),
            results=results
        )

        score = evaluation.get_summary_score()
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_comprehensive_evaluation_to_dict(self):
        """辞書変換テスト"""
        timestamp = datetime.now()

        results = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5,
                confidence_interval=(1.2, 1.8)
            )
        }

        evaluation = ComprehensiveEvaluation(
            model_name="test_model",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=timestamp,
            results=results,
            summary_stats={"test": "value"},
            risk_metrics={"volatility": 0.02}
        )

        data = evaluation.to_dict()

        self.assertEqual(data["model_name"], "test_model")
        self.assertEqual(data["evaluation_type"], "backtest")
        self.assertIn("results", data)
        self.assertIn("summary_stats", data)
        self.assertIn("risk_metrics", data)
        self.assertIn("summary_score", data)

    def test_evaluator_save_load_evaluation(self):
        """評価結果の保存・読み込みテスト"""
        results = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5
            )
        }

        evaluation = ComprehensiveEvaluation(
            model_name="test_model",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=datetime.now(),
            results=results
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # 保存
            self.evaluator.save_evaluation(evaluation, temp_path)

            # 読み込み
            loaded_evaluation = self.evaluator.load_evaluation(temp_path)

            # 検証
            self.assertEqual(loaded_evaluation.model_name, evaluation.model_name)
            self.assertEqual(loaded_evaluation.evaluation_type, evaluation.evaluation_type)
            self.assertEqual(len(loaded_evaluation.results), len(evaluation.results))

        finally:
            temp_path.unlink()

    def test_evaluator_compare_evaluations(self):
        """評価結果比較テスト"""
        # 評価結果1
        results1 = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5
            ),
            EvaluationMetric.MAX_DRAWDOWN: EvaluationResult(
                metric=EvaluationMetric.MAX_DRAWDOWN,
                value=0.15
            )
        }

        evaluation1 = ComprehensiveEvaluation(
            model_name="model1",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=datetime.now(),
            results=results1
        )

        # 評価結果2
        results2 = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.8
            ),
            EvaluationMetric.MAX_DRAWDOWN: EvaluationResult(
                metric=EvaluationMetric.MAX_DRAWDOWN,
                value=0.12
            )
        }

        evaluation2 = ComprehensiveEvaluation(
            model_name="model2",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=datetime.now(),
            results=results2
        )

        # 比較
        comparison = self.evaluator.compare_evaluations([evaluation1, evaluation2])

        self.assertIn("model_count", comparison)
        self.assertEqual(comparison["model_count"], 2)
        self.assertIn("best_model", comparison)
        self.assertIn("worst_model", comparison)
        self.assertIn("average_score", comparison)
        self.assertIn("metric_rankings", comparison)

    def test_evaluator_compare_evaluations_empty(self):
        """空の評価結果比較テスト"""
        comparison = self.evaluator.compare_evaluations([])
        self.assertEqual(comparison, {})

    def test_evaluation_metrics_enum(self):
        """評価指標Enumテスト"""
        self.assertEqual(EvaluationMetric.SHARPE_RATIO.value, "sharpe_ratio")
        self.assertEqual(EvaluationMetric.MAX_DRAWDOWN.value, "max_drawdown")
        self.assertEqual(EvaluationMetric.TOTAL_RETURN.value, "total_return")

    def test_evaluation_types_enum(self):
        """評価タイプEnumテスト"""
        self.assertEqual(EvaluationType.BACKTEST.value, "backtest")
        self.assertEqual(EvaluationType.WALK_FORWARD.value, "walk_forward")
        self.assertEqual(EvaluationType.CROSS_VALIDATION.value, "cross_validation")


class TestEvaluationMetrics(unittest.TestCase):
    """評価指標関連のテスト"""

    def test_evaluation_result_with_metadata(self):
        """メタデータ付きEvaluationResultテスト"""
        metadata = {"confidence_level": 0.95, "sample_size": 1000}

        result = EvaluationResult(
            metric=EvaluationMetric.SHARPE_RATIO,
            value=1.5,
            metadata=metadata
        )

        self.assertEqual(result.metadata, metadata)

    def test_comprehensive_evaluation_with_all_fields(self):
        """全フィールド付きComprehensiveEvaluationテスト"""
        results = {
            EvaluationMetric.SHARPE_RATIO: EvaluationResult(
                metric=EvaluationMetric.SHARPE_RATIO,
                value=1.5
            )
        }

        evaluation = ComprehensiveEvaluation(
            model_name="test_model",
            evaluation_type=EvaluationType.BACKTEST,
            timestamp=datetime.now(),
            results=results,
            summary_stats={"total_tests": 100},
            risk_metrics={"var_95": -0.05},
            performance_metrics={"win_rate": 0.55},
            market_regime_analysis={"bull_performance": 0.18},
            robustness_tests={"parameter_sensitivity": "low"}
        )

        self.assertEqual(evaluation.summary_stats["total_tests"], 100)
        self.assertEqual(evaluation.risk_metrics["var_95"], -0.05)
        self.assertEqual(evaluation.performance_metrics["win_rate"], 0.55)
        self.assertEqual(evaluation.market_regime_analysis["bull_performance"], 0.18)
        self.assertEqual(evaluation.robustness_tests["parameter_sensitivity"], "low")


if __name__ == '__main__':
    unittest.main()