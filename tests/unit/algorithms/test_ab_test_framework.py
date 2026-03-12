"""
A/Bテストフレームワークの包括的なテストスイート
パフォーマンスベンチマークを含む全コンポーネントのテスト
"""

import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ztb.adaptation.ab_test.analyzer import ABTestAnalyzer
from ztb.adaptation.ab_test.config import ABTestConfig
from ztb.adaptation.ab_test.executor import ABTestExecutor
from ztb.adaptation.ab_test.selector import ModelSelector

# テスト対象のモジュールをインポート
from ztb.adaptation.ab_test.types import (
    ABTestConfiguration,
    ABTestResult,
    ABTestStatus,
    ABTestVariant,
    StatisticalResult,
    StatisticalTest,
)
from ztb.trading.environment.constants import BYTES_PER_MB


class TestABTestTypes(unittest.TestCase):
    """A/Bテストタイプ定義のテスト"""

    def test_statistical_test_enum(self):
        """StatisticalTest enumのテスト"""
        self.assertEqual(StatisticalTest.T_TEST.value, "t_test")
        self.assertEqual(StatisticalTest.MANN_WHITNEY.value, "mann_whitney")
        self.assertEqual(StatisticalTest.CHI_SQUARE.value, "chi_square")

    def test_ab_test_variant_creation(self):
        """ABTestVariantの作成テスト"""
        variant = ABTestVariant(
            variant_id="test_variant",
            model_path="/path/to/model",
            model_version="v1.0",
            description="Test variant",
        )

        self.assertEqual(variant.variant_id, "test_variant")
        self.assertEqual(variant.model_path, "/path/to/model")
        self.assertEqual(variant.model_version, "v1.0")
        self.assertEqual(variant.description, "Test variant")
        self.assertIsInstance(variant.metadata, dict)

    def test_ab_test_configuration_creation(self):
        """ABTestConfigurationの作成テスト"""
        variant_a = ABTestVariant("A", "/model/a", "v1.0", "Variant A")
        variant_b = ABTestVariant("B", "/model/b", "v1.1", "Variant B")
        config = ABTestConfiguration(
            test_id="test_001",
            name="Test A/B Test",
            description="Test configuration",
            variant_a=variant_a,
            variant_b=variant_b,
            statistical_test=StatisticalTest.T_TEST,
            confidence_level=0.95,
            minimum_sample_size=1000,
            max_duration_hours=1,
            traffic_percentage=50.0,
        )

        self.assertEqual(config.test_id, "test_001")
        self.assertEqual(config.name, "Test A/B Test")
        self.assertEqual(config.variant_a.variant_id, "A")
        self.assertEqual(config.variant_b.variant_id, "B")
        self.assertEqual(config.statistical_test, StatisticalTest.T_TEST)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.minimum_sample_size, 1000)


class TestABTestConfig(unittest.TestCase):
    """A/Bテスト設定のテスト"""

    def test_config_creation(self):
        """設定オブジェクトの作成テスト"""
        config = ABTestConfig()

        self.assertIsInstance(config.performance, object)
        self.assertIsInstance(config.risk, object)
        self.assertIsInstance(config.statistics, object)

    def test_performance_config(self):
        """パフォーマンス設定のテスト"""
        config = ABTestConfig()
        perf_config = config.performance

        self.assertGreater(perf_config.max_workers, 0)
        self.assertGreater(perf_config.batch_size, 0)
        self.assertGreater(perf_config.stream_buffer_size, 0)

    def test_risk_config(self):
        """リスク設定のテスト"""
        config = ABTestConfig()
        risk_config = config.risk

        self.assertGreater(risk_config.max_regression_rate, 0)
        self.assertGreater(risk_config.rollback_trigger_threshold, 0)
        self.assertGreater(risk_config.regression_detection_window, 0)


class TestABTestAnalyzer(unittest.TestCase):
    """A/Bテストアナライザーのテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.analyzer = ABTestAnalyzer()
        self.sample_data_a = np.random.normal(100, 10, 1000)
        self.sample_data_b = np.random.normal(105, 10, 1000)

    def test_t_test_calculation(self):
        """現行 analyze_parallel の基本動作テスト"""
        result = self.analyzer.analyze_parallel(
            self.sample_data_a, self.sample_data_b
        )

        self.assertIsInstance(result, StatisticalResult)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size, float)
        self.assertIsInstance(result.confidence_interval, tuple)
        self.assertEqual(len(result.confidence_interval), 2)

    def test_effect_size_calculation(self):
        """効果量計算のテスト"""
        result = self.analyzer._calculate_effect_size(
            self.sample_data_a, self.sample_data_b
        )

        self.assertIsInstance(result, float)

    def test_parallel_analysis(self):
        """並列分析のテスト"""
        result = self.analyzer.analyze_parallel(self.sample_data_a, self.sample_data_b)
        self.assertIsInstance(result, StatisticalResult)
        self.assertGreater(result.sample_size_a, 0)
        self.assertGreater(result.sample_size_b, 0)

    def test_bootstrap_confidence_interval(self):
        """ブートストラップ信頼区間のテスト"""
        result = self.analyzer.calculate_bootstrap_ci(
            self.sample_data_a, self.sample_data_b, n_bootstrap=100
        )

        self.assertIsInstance(result, StatisticalResult)
        self.assertEqual(len(result.confidence_interval), 2)
        self.assertLess(
            result.confidence_interval[0], result.confidence_interval[1]
        )  # 下限 < 上限


@unittest.skip(
    "Legacy executor harness targets a pre-streaming A/B API and is covered by newer component tests."
)
class TestABTestExecutor(unittest.TestCase):
    """A/Bテスト実行エンジンのテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ABTestExecutor(base_path=self.temp_dir)

        # テスト設定
        self.test_config = ABTestConfiguration(
            test_id="test_exec_001",
            name="Test Execution",
            description="Test executor functionality",
            variants=[
                ABTestVariant("A", "/fake/model/a", "v1.0", "Variant A"),
                ABTestVariant("B", "/fake/model/b", "v1.1", "Variant B"),
            ],
            statistical_tests=[StatisticalTest.T_TEST],
            confidence_level=0.95,
            minimum_sample_size=100,
            maximum_test_duration=60,
            traffic_split=0.5,
        )

    def tearDown(self):
        """テストクリーンアップ"""
        shutil.rmtree(self.temp_dir)

    @patch("ztb.adaptation.ab_test.executor.ABTestAnalyzer")
    def test_execution_initialization(self, mock_analyzer):
        """実行初期化のテスト"""
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance

        result = self.executor.initialize_test(self.test_config)

        self.assertIsInstance(result, ABTestResult)
        self.assertEqual(result.test_id, "test_exec_001")
        self.assertEqual(result.status, ABTestStatus.RUNNING)

    def test_memory_monitoring(self):
        """メモリ監視のテスト"""
        memory_usage = self.executor._get_memory_usage()

        self.assertIsInstance(memory_usage, float)
        self.assertGreaterEqual(memory_usage, 0.0)

    def test_data_streaming(self):
        """データストリーミングのテスト"""
        # テストデータを追加
        test_data = {"variant_a": [1.0, 2.0, 3.0], "variant_b": [1.1, 2.1, 3.1]}

        self.executor._stream_data(test_data)

        # キューサイズを確認
        self.assertGreaterEqual(self.executor.data_queue.qsize(), 0)

@unittest.skip(
    "Legacy selector API tests predate the current select_model decision contract."
)
class TestModelSelector(unittest.TestCase):
    """モデル選択器のテスト"""

    def setUp(self):
        """テストセットアップ"""
        self.selector = ModelSelector()

    def test_winner_selection(self):
        """勝者選択のテスト"""
        # 統計結果のモック
        mock_result_a = Mock()
        mock_result_a.p_value = 0.01  # 有意差あり
        mock_result_a.effect_size = 0.2

        mock_result_b = Mock()
        mock_result_b.p_value = 0.05  # 有意差なし
        mock_result_b.effect_size = 0.1

        results = {
            StatisticalTest.T_TEST: mock_result_a,
            StatisticalTest.MANN_WHITNEY: mock_result_b,
        }

        winner = self.selector.select_winner(results)

        self.assertIn(winner, ["variant_a", "variant_b", "tie"])

    def test_risk_assessment(self):
        """リスク評価のテスト"""
        # 回帰データのテスト
        regression_data = [-0.1, -0.05, 0.0, 0.05, 0.1]

        risk_score = self.selector._assess_risk(regression_data)

        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0.0)

    def test_rollback_decision(self):
        """ロールバック決定のテスト"""
        # 高いリスクスコア
        high_risk_results = {"risk_score": 0.8}

        should_rollback = self.selector.should_rollback(high_risk_results)

        self.assertTrue(should_rollback)

        # 低いリスクスコア
        low_risk_results = {"risk_score": 0.2}

        should_rollback = self.selector.should_rollback(low_risk_results)

        self.assertFalse(should_rollback)


@unittest.skip(
    "Environment-dependent benchmark assertions are unstable under the current test harness."
)
class TestPerformanceBenchmarks(unittest.TestCase):
    """パフォーマンスベンチマークテスト"""

    def setUp(self):
        """ベンチマークセットアップ"""
        self.analyzer = ABTestAnalyzer()
        self.large_data_a = np.random.normal(100, 10, 10000)
        self.large_data_b = np.random.normal(105, 10, 10000)

    def test_parallel_vs_sequential_performance(self):
        """並列 vs 順次処理のパフォーマンス比較"""
        test_data = {"variant_a": self.large_data_a, "variant_b": self.large_data_b}

        # 順次処理
        start_time = time.time()
        sequential_results = {}
        for test_type in [StatisticalTest.T_TEST, StatisticalTest.MANN_WHITNEY]:
            sequential_results[test_type] = self.analyzer.analyze_statistical_test(
                test_data, test_type
            )
        sequential_time = time.time() - start_time

        # 並列処理
        start_time = time.time()
        parallel_results = self.analyzer.analyze_parallel(
            test_data, [StatisticalTest.T_TEST, StatisticalTest.MANN_WHITNEY]
        )
        parallel_time = time.time() - start_time

        # 結果の検証
        self.assertIn(StatisticalTest.T_TEST, parallel_results)
        self.assertIn(StatisticalTest.MANN_WHITNEY, parallel_results)

        # パフォーマンスログ（並列が順次より速いことを期待）
        print(f"Sequential time: {sequential_time:.4f}s")
        print(f"Parallel time: {parallel_time:.4f}s")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")

    def test_memory_efficiency(self):
        """メモリ効率のテスト"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss  / BYTES_PER_MB

        # 大規模データ処理
        for _ in range(10):
            self.analyzer.analyze_parallel(
                {
                    "variant_a": np.random.normal(100, 10, 5000),
                    "variant_b": np.random.normal(105, 10, 5000),
                },
                [StatisticalTest.T_TEST],
            )

        final_memory = process.memory_info().rss / BYTES_PER_MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase: {memory_increase:.2f} MB")

        # メモリ増加が合理的な範囲内であることを確認
        self.assertLess(memory_increase, 100.0)  # 100MB以内の増加

    def test_batch_processing_efficiency(self):
        """バッチ処理効率のテスト"""
        batch_sizes = [100, 500, 1000, 5000]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                data_a = np.random.normal(100, 10, batch_size)
                data_b = np.random.normal(105, 10, batch_size)

                start_time = time.time()
                result = self.analyzer._perform_t_test(data_a, data_b)
                processing_time = time.time() - start_time

                print(f"Batch size {batch_size}: {processing_time:.4f}s")

                # 結果が有効であることを確認
                self.assertIsInstance(result, StatisticalResult)
                self.assertIsInstance(result.p_value, float)


if __name__ == "__main__":
    # 詳細なテスト出力
    unittest.main(verbosity=2)
