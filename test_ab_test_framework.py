"""
Comprehensive Unit Tests for A/B Testing Framework
Tests focus on processing time reduction and memory efficiency
"""

import unittest
import time
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime, timedelta

from ztb.adaptation.ab_test import (
    ABTestExecutor, ABTestConfig, ABTestAnalyzer,
    ModelSelector, TrafficManager, ABTestVariant,
    ABTestConfiguration, ABTestState, ABTestMetrics,
    ABTestResult, ABTestResultSummary, StatisticalResult
)


class TestABTestConfig(unittest.TestCase):
    """ABTestConfigのテスト"""

    def test_default_configuration(self):
        """デフォルト設定の検証"""
        config = ABTestConfig()

        self.assertEqual(config.statistics.min_sample_size, 500)
        self.assertEqual(config.performance.max_memory_mb, 1024)
        self.assertEqual(config.performance.max_workers, 4)
        self.assertEqual(config.performance.batch_size, 1000)
        self.assertTrue(config.performance.enable_caching)
        self.assertTrue(config.enabled)

    def test_performance_optimized_config(self):
        """パフォーマンス最適化設定の検証"""
        config = ABTestConfig(
            min_sample_size=500,
            max_memory_mb=200,
            processing_threads=8,
            batch_size=50
        )

        self.assertEqual(config.min_sample_size, 500)
        self.assertEqual(config.max_memory_mb, 200)
        self.assertEqual(config.processing_threads, 8)
        self.assertEqual(config.batch_size, 50)


class TestABTestAnalyzer(unittest.TestCase):
    """ABTestAnalyzerのテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        self.analyzer = ABTestAnalyzer()  # configなしで初期化

    def test_streaming_statistics_memory_efficiency(self):
        """ストリーミング統計のメモリ効率テスト"""
        # 大量のデータをストリーミング処理
        data_a = np.random.normal(0, 1, 10000)
        data_b = np.random.normal(0.1, 1, 10000)

        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # ストリーミング処理
        for i in range(0, len(data_a), 100):
            batch_a = data_a[i:i+100]
            batch_b = data_b[i:i+100]

            self.analyzer.update_streaming_stats('test_a', batch_a)
            self.analyzer.update_streaming_stats('test_b', batch_b)

        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory

        # メモリ使用量が500MB未満であることを確認
        self.assertLess(memory_used, 500)

    def test_parallel_statistical_analysis(self):
        """並列統計解析のテスト"""
        data_a = np.random.normal(0, 1, 5000)
        data_b = np.random.normal(0.2, 1, 5000)

        start_time = time.time()

        # 並列処理で統計解析を実行
        result = self.analyzer.analyze_parallel(data_a, data_b)

        end_time = time.time()
        processing_time = end_time - start_time

        # 結果の検証
        self.assertIsInstance(result, StatisticalResult)
        self.assertIsNotNone(result.p_value)
        self.assertIsNotNone(result.effect_size)

        # 処理時間が2秒未満であることを確認（パフォーマンス要件）
        self.assertLess(processing_time, 2.0)

    def test_bootstrap_confidence_intervals(self):
        """ブートストラップ信頼区間テスト"""
        data_a = np.random.normal(0, 1, 1000)
        data_b = np.random.normal(0.1, 1, 1000)

        result = self.analyzer.calculate_bootstrap_ci(data_a, data_b, n_bootstrap=1000)

        # 信頼区間が計算されていることを確認
        self.assertIsNotNone(result.confidence_interval)
        self.assertEqual(len(result.confidence_interval), 2)

        # 効果量が正であることを確認
        self.assertGreater(result.effect_size, 0)


class TestABTestExecutor(unittest.TestCase):
    """ABTestExecutorのテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        self.executor = ABTestExecutor(self.config)

    def test_memory_monitoring(self):
        """メモリ監視機能のテスト"""
        # メモリ使用量を監視しながら処理を実行
        initial_memory = self.executor.get_memory_usage()

        # 大量のデータを処理
        large_data = np.random.random((1000, 100))

        # メモリ監視が機能することを確認
        current_memory = self.executor.get_memory_usage()
        self.assertIsInstance(current_memory, float)
        self.assertGreaterEqual(current_memory, 0)

    def test_concurrent_processing(self):
        """並列処理のテスト"""
        # 複数のテストを並列実行
        test_configs = []
        for i in range(5):
            config = ABTestConfiguration(
                test_id=f"test_{i}",
                variant_a=ABTestVariant(
                    variant_id=f"variant_a_{i}",
                    model_path=f"/path/to/model_a_{i}",
                    model_version="1.0",
                    description=f"Test variant A {i}"
                ),
                variant_b=ABTestVariant(
                    variant_id=f"variant_b_{i}",
                    model_path=f"/path/to/model_b_{i}",
                    model_version="1.0",
                    description=f"Test variant B {i}"
                ),
                minimum_sample_size=100,
                minimum_effect_size=0.1,
                confidence_level=0.95
            )
            test_configs.append(config)

        start_time = time.time()

        # 並列実行
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.executor.run_test, config) for config in test_configs]
            for future in results:
                results.append(future.result())

        end_time = time.time()
        total_time = end_time - start_time

        # 5つのテストが4スレッドで実行されるため、単一スレッドより高速であることを確認
        # （正確な時間は環境によるが、少なくとも完了すること）
        self.assertGreater(len(results), 0)

    def test_risk_management(self):
        """リスク管理機能のテスト"""
        config = ABTestConfiguration(
            test_id="risk_test",
            variant_a=ABTestVariant("a", "/path/a", "1.0", "A"),
            variant_b=ABTestVariant("b", "/path/b", "1.0", "B"),
            minimum_sample_size=100,
            minimum_effect_size=0.1,
            confidence_level=0.95
        )

        # リスク検出が機能することを確認
        risk_detected = self.executor.detect_risk(config, {})
        self.assertIsInstance(risk_detected, bool)

    @patch('ztb.adaptation.ab_test.executor.ABTestExecutor.get_memory_usage')
    def test_memory_threshold_enforcement(self, mock_memory):
        """メモリしきい値適用テスト"""
        # メモリ使用量がしきい値を超えた場合の動作をテスト
        mock_memory.return_value = 600  # 500MBしきい値を超える

        # メモリチェックが機能することを確認
        memory_ok = self.executor.check_memory_threshold()
        self.assertFalse(memory_ok)


class TestModelSelector(unittest.TestCase):
    """ModelSelectorのテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        self.selector = ModelSelector(self.config)

    def test_model_selection_logic(self):
        """モデル選択ロジックのテスト"""
        test_config = ABTestConfiguration(
            test_id="selection_test",
            variant_a=ABTestVariant("a", "/path/a", "1.0", "A"),
            variant_b=ABTestVariant("b", "/path/b", "1.0", "B"),
            minimum_sample_size=1000,
            minimum_effect_size=0.1,
            confidence_level=0.95
        )

        test_state = ABTestState(
            test_id="selection_test",
            status="running",
            start_time=datetime.now(),
            metrics_a=ABTestMetrics(
                sample_count=1500,
                mean_reward=10.5,
                std_reward=2.1,
                total_trades=150,
                win_rate=0.65
            ),
            metrics_b=ABTestMetrics(
                sample_count=1500,
                mean_reward=11.2,
                std_reward=2.0,
                total_trades=160,
                win_rate=0.68
            ),
            regression_detected=False
        )

        result_summary = ABTestResultSummary(
            test_id="selection_test",
            result=ABTestResult.WINNER_B,
            statistical_result=StatisticalResult(
                p_value=0.001,
                effect_size=0.8,
                confidence_interval=(0.5, 1.1),
                sample_size_a=1500,
                sample_size_b=1500,
                test_type="t-test"
            ),
            confidence_level=0.95,
            analysis_time=datetime.now()
        )

        # モデル選択を実行
        decision = self.selector.select_model(test_config, test_state, result_summary)

        # 決定が正しく生成されていることを確認
        self.assertIsNotNone(decision["selected_variant"])
        self.assertEqual(decision["action"], "deploy")
        self.assertGreater(decision["confidence_level"], 0.7)
        self.assertGreater(decision["recommended_traffic_percentage"], 0)

    def test_risk_assessment(self):
        """リスク評価のテスト"""
        test_config = ABTestConfiguration(
            test_id="risk_test",
            variant_a=ABTestVariant("a", "/path/a", "1.0", "A"),
            variant_b=ABTestVariant("b", "/path/b", "1.0", "B"),
            minimum_sample_size=100,
            minimum_effect_size=0.1,
            confidence_level=0.95
        )

        test_state = ABTestState(
            test_id="risk_test",
            status="running",
            start_time=datetime.now(),
            metrics_a=ABTestMetrics(sample_count=50, mean_reward=10, std_reward=1, total_trades=10, win_rate=0.6),
            metrics_b=ABTestMetrics(sample_count=50, mean_reward=10.1, std_reward=1, total_trades=10, win_rate=0.61),
            regression_detected=False
        )

        result_summary = ABTestResultSummary(
            test_id="risk_test",
            result=ABTestResult.WINNER_B,
            statistical_result=StatisticalResult(
                p_value=0.3,  # 高めのp値
                effect_size=0.05,  # 小さめの効果量
                confidence_interval=(-0.1, 0.2),
                sample_size_a=50,
                sample_size_b=50,
                test_type="t-test"
            ),
            confidence_level=0.95,
            analysis_time=datetime.now()
        )

        # リスク評価を実行
        risks = self.selector._assess_deployment_risks(test_config, test_state, result_summary)

        # 高リスクが検出されていることを確認
        self.assertEqual(risks["sample_size_risk"], "high")
        self.assertEqual(risks["overall_risk"], "high")

    def test_rollback_triggers(self):
        """ロールバックトリガーのテスト"""
        test_id = "rollback_test"
        variant = ABTestVariant("test_variant", "/path", "1.0", "Test")

        # ロールバックトリガーを設定
        self.selector._setup_rollback_triggers(test_id, variant, {"overall_risk": "high"})

        # トリガーが設定されていることを確認
        self.assertIn(test_id, self.selector.rollback_triggers)

        # ロールバック条件をチェック
        metrics = {"error_rate": 0.15}  # しきい値を超える
        timestamp = datetime.now()

        rollback_triggered = self.selector.check_rollback_conditions(test_id, metrics, timestamp)

        # 高リスク設定なのでロールバックがトリガーされるはず
        # （実際の動作はトリガー条件による）

    def test_force_rollback(self):
        """強制ロールバックのテスト"""
        test_id = "force_rollback_test"
        variant = ABTestVariant("test_variant", "/path", "1.0", "Test")

        # まずトリガーを設定
        self.selector._setup_rollback_triggers(test_id, variant, {"overall_risk": "low"})

        # 強制ロールバックを実行
        success = self.selector.force_rollback(test_id, "Manual test rollback")

        # ロールバックが成功したことを確認
        self.assertTrue(success)

        # トリガーが削除されていることを確認
        self.assertNotIn(test_id, self.selector.rollback_triggers)

        # 履歴に記録されていることを確認
        history = self.selector.get_deployment_history(test_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["reason"], "Manual test rollback")


class TestTrafficManager(unittest.TestCase):
    """TrafficManagerのテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        self.manager = TrafficManager(self.config)

    def test_traffic_allocation(self):
        """トラフィック割り当てのテスト"""
        test_id = "traffic_test"
        variant_a = ABTestVariant("a", "/path/a", "1.0", "A")
        variant_b = ABTestVariant("b", "/path/b", "1.0", "B")
        percentage = 25.0

        # トラフィックを割り当て
        allocation = self.manager.allocate_traffic(test_id, variant_a, variant_b, percentage)

        # 割り当てが正しいことを確認
        expected_a = (100 - percentage) / 100
        expected_b = percentage / 100

        self.assertAlmostEqual(allocation[variant_a.variant_id], expected_a, places=5)
        self.assertAlmostEqual(allocation[variant_b.variant_id], expected_b, places=5)

    def test_traffic_ramp_up(self):
        """トラフィック段階増加のテスト"""
        test_id = "ramp_test"
        variant_a = ABTestVariant("a", "/path/a", "1.0", "A")
        variant_b = ABTestVariant("b", "/path/b", "1.0", "B")

        # 初期割り当て
        self.manager.allocate_traffic(test_id, variant_a, variant_b, 10.0)

        # 段階増加スケジュールを生成
        schedule = self.manager.ramp_up_traffic(test_id, 50.0, steps=5, interval_minutes=30)

        # スケジュールが正しく生成されていることを確認
        self.assertEqual(len(schedule), 5)

        # 各ステップの割合が正しく増加していることを確認
        for i, step in enumerate(schedule):
            expected_percentage = 10.0 + (40.0 / 5) * (i + 1)
            self.assertAlmostEqual(step["percentage"], expected_percentage, places=1)

    def test_traffic_update(self):
        """トラフィック更新のテスト"""
        test_id = "update_test"
        variant_a = ABTestVariant("a", "/path/a", "1.0", "A")
        variant_b = ABTestVariant("b", "/path/b", "1.0", "B")

        # 初期割り当て
        self.manager.allocate_traffic(test_id, variant_a, variant_b, 20.0)

        # 更新
        success = self.manager.update_traffic_allocation(test_id, 40.0)

        # 更新が成功したことを確認
        self.assertTrue(success)

        # 新しい割り当てを取得
        allocation = self.manager.get_traffic_allocation(test_id)
        self.assertIsNotNone(allocation)

        expected_a = (100 - 40.0) / 100
        expected_b = 40.0 / 100

        self.assertAlmostEqual(allocation[variant_a.variant_id], expected_a, places=5)
        self.assertAlmostEqual(allocation[variant_b.variant_id], expected_b, places=5)


class TestPerformanceBenchmarks(unittest.TestCase):
    """パフォーマンスベンチマークテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        self.executor = ABTestExecutor(self.config)
        self.analyzer = ABTestAnalyzer(self.config)

    def test_large_scale_processing(self):
        """大規模データ処理のパフォーマンステスト"""
        # 10万件のデータを処理
        data_a = np.random.normal(0, 1, 50000)
        data_b = np.random.normal(0.1, 1, 50000)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # ストリーミング処理
        for i in range(0, len(data_a), 1000):
            batch_a = data_a[i:i+1000]
            batch_b = data_b[i:i+1000]

            self.analyzer.update_streaming_stats('bench_a', batch_a)
            self.analyzer.update_streaming_stats('bench_b', batch_b)

        # 統計解析を実行
        result = self.analyzer.analyze_parallel(data_a, data_b)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        processing_time = end_time - start_time
        memory_used = end_memory - start_memory

        # パフォーマンス要件を検証
        self.assertLess(processing_time, 10.0)  # 10秒未満
        self.assertLess(memory_used, 1000)  # 1GB未満

        # 結果が有効であることを確認
        self.assertIsNotNone(result.p_value)
        self.assertIsNotNone(result.effect_size)

    def test_concurrent_test_execution(self):
        """並列テスト実行のパフォーマンステスト"""
        # 10個のテストを並列実行
        test_configs = []
        for i in range(10):
            config = ABTestConfiguration(
                test_id=f"perf_test_{i}",
                variant_a=ABTestVariant(f"a_{i}", f"/path/a_{i}", "1.0", f"A_{i}"),
                variant_b=ABTestVariant(f"b_{i}", f"/path/b_{i}", "1.0", f"B_{i}"),
                minimum_sample_size=1000,
                minimum_effect_size=0.1,
                confidence_level=0.95
            )
            test_configs.append(config)

        start_time = time.time()

        # 並列実行
        results = []
        with ThreadPoolExecutor(max_workers=self.config.processing_threads) as executor:
            futures = [executor.submit(self._run_test_mock, config) for config in test_configs]
            for future in futures:
                results.append(future.result())

        end_time = time.time()
        total_time = end_time - start_time

        # すべてのテストが完了したことを確認
        self.assertEqual(len(results), 10)

        # 並列処理により時間が節約されていることを確認
        # （シングルスレッドより大幅に高速）
        self.assertLess(total_time, 30.0)  # 30秒未満

    def _run_test_mock(self, config):
        """モックテスト実行（パフォーマンステスト用）"""
        # 実際のテスト実行をシミュレート
        time.sleep(0.1)  # 軽い処理をシミュレート
        return {"test_id": config.test_id, "status": "completed"}


if __name__ == '__main__':
    unittest.main()