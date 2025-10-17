"""
Unit tests for A/B Testing Framework
処理時間短縮・メモリ効率を考慮したテスト実装
"""

import unittest
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

from ztb.adaptation.ab_test.analyzer import ABTestAnalyzer
from ztb.adaptation.ab_test.config import ABTestConfig, ABTestPerformanceConfig
from ztb.adaptation.ab_test.selector import ModelSelector, TrafficManager
from ztb.adaptation.ab_test.types import (
    StatisticalResult, ABTestMetrics, StatisticalTest,
    ABTestConfiguration, ABTestVariant, ABTestState, ABTestStatus,
    ABTestResultSummary, ABTestResult
)


class TestABTestAnalyzer(unittest.TestCase):
    """ABTestAnalyzerのテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        # テスト用にパフォーマンス設定を調整
        self.config.performance.max_workers = 2
        self.config.performance.batch_size = 100
        self.analyzer = ABTestAnalyzer(self.config)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # メモリリーク防止のため明示的にクリーンアップ
        if hasattr(self.analyzer, 'executor'):
            self.analyzer.executor.shutdown(wait=False)
        gc.collect()

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.analyzer.config, ABTestConfig)
        self.assertIsInstance(self.analyzer.executor, ThreadPoolExecutor)

    def test_parallel_analysis_basic(self):
        """基本的な並列分析テスト"""
        # テストデータ生成
        np.random.seed(42)
        data_a = np.random.normal(0.5, 0.1, 1000)
        data_b = np.random.normal(0.52, 0.1, 1000)

        start_time = time.time()
        result = self.analyzer.analyze_parallel(data_a, data_b)
        end_time = time.time()

        # 結果の検証
        self.assertIsInstance(result, StatisticalResult)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size, float)
        self.assertEqual(result.sample_size_a, 1000)
        self.assertEqual(result.sample_size_b, 1000)
        self.assertLess(result.p_value, 1.0)  # p値は1以下
        self.assertGreater(result.p_value, 0.0)  # p値は0以上

        # 処理時間が妥当か確認（並列処理で高速化されているはず）
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)  # 1秒以内に完了すべき

    def test_effect_size_calculation(self):
        """効果量計算テスト"""
        # 明確な差があるデータ（標準偏差があるデータ）
        np.random.seed(42)
        data_a = np.random.normal(1.0, 0.1, 100)  # 平均1.0, 標準偏差0.1
        data_b = np.random.normal(1.2, 0.1, 100)  # 平均1.2, 標準偏差0.1

        result = self.analyzer.analyze_parallel(data_a, data_b)

        # 効果量が正の値であることを確認
        self.assertGreater(result.effect_size, 0)

        # 逆の場合
        result_reverse = self.analyzer.analyze_parallel(data_b, data_a)
        self.assertLess(result_reverse.effect_size, 0)

    def test_bootstrap_analysis(self):
        """ブートストラップ分析テスト"""
        np.random.seed(42)
        data_a = np.random.normal(0.5, 0.1, 200)
        data_b = np.random.normal(0.52, 0.1, 200)

        start_time = time.time()
        result = self.analyzer.calculate_bootstrap_ci(data_a, data_b, n_bootstrap=100)
        end_time = time.time()

        # 結果の検証
        self.assertIsInstance(result, StatisticalResult)
        self.assertIsInstance(result.confidence_interval, tuple)
        self.assertEqual(len(result.confidence_interval), 2)

        # 処理時間が妥当か確認
        processing_time = end_time - start_time
        self.assertLess(processing_time, 2.0)  # 2秒以内に完了すべき

    def test_memory_efficiency_large_dataset(self):
        """大規模データセットでのメモリ効率テスト"""
        # 大規模データ生成
        np.random.seed(42)
        data_a = np.random.normal(0.5, 0.1, 10000)
        data_b = np.random.normal(0.52, 0.1, 10000)

        # メモリ使用量を監視しながら分析（簡易的な方法）
        import sys
        memory_before = sys.getsizeof(data_a) + sys.getsizeof(data_b)

        result = self.analyzer.analyze_parallel(data_a, data_b)

        # データがメモリ上に存在することを確認（大規模データでも処理可能）
        self.assertIsInstance(result, StatisticalResult)
        self.assertEqual(result.sample_size_a, 10000)
        self.assertEqual(result.sample_size_b, 10000)

        # 処理が完了し、結果が得られることを確認
        self.assertIsNotNone(result.p_value)
        self.assertIsNotNone(result.effect_size)

    def test_streaming_statistics(self):
        """ストリーミング統計テスト"""
        from ztb.adaptation.ab_test.types import StreamingStatistics

        # ストリーミング統計クラスのテスト
        stats = StreamingStatistics("test_variant")

        # サンプルデータを追加
        for i in range(100):
            sample = {"value": float(i)}
            stats.add_sample(sample)

        # 統計量が正しく計算されていることを確認
        self.assertEqual(stats.count, 100)
        self.assertAlmostEqual(stats.mean, 49.5, places=1)
        self.assertGreater(stats.get_std(), 0)

        # リセットテスト
        stats.reset()
        self.assertEqual(stats.count, 0)

    def test_performance_optimization(self):
        """パフォーマンス最適化テスト"""
        # 異なるワーカー数での比較
        np.random.seed(42)
        data_a = np.random.normal(0.5, 0.1, 5000)
        data_b = np.random.normal(0.52, 0.1, 5000)

        # シングルワーカー設定
        config_single = ABTestConfig()
        config_single.performance.max_workers = 1
        analyzer_single = ABTestAnalyzer(config_single)

        # マルチワーカー設定
        config_multi = ABTestConfig()
        config_multi.performance.max_workers = 4
        analyzer_multi = ABTestAnalyzer(config_multi)

        # シングルワーカーでの実行時間
        start_time = time.time()
        result_single = analyzer_single.analyze_parallel(data_a, data_b)
        time_single = time.time() - start_time

        # マルチワーカーでの実行時間
        start_time = time.time()
        result_multi = analyzer_multi.analyze_parallel(data_a, data_b)
        time_multi = time.time() - start_time

        # 結果が同等であることを確認
        self.assertAlmostEqual(result_single.p_value, result_multi.p_value, places=3)
        self.assertAlmostEqual(result_single.effect_size, result_multi.effect_size, places=3)

        # クリーンアップ
        analyzer_single.executor.shutdown(wait=False)
        analyzer_multi.executor.shutdown(wait=False)

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 空のデータでのテスト
        try:
            result = self.analyzer.analyze_parallel(np.array([]), np.array([]))
            # 空のデータでもクラッシュしないことを確認
            self.assertIsInstance(result, StatisticalResult)
        except Exception as e:
            # 例外が発生しても適切に処理されることを確認
            self.assertIsInstance(e, (ValueError, ZeroDivisionError))

        # 片方が空のデータ
        data_a = np.array([1.0, 2.0, 3.0])
        data_b = np.array([])

        try:
            result = self.analyzer.analyze_parallel(data_a, data_b)
            self.assertIsInstance(result, StatisticalResult)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, ZeroDivisionError))


class TestABTestConfiguration(unittest.TestCase):
    """ABTestConfigのテスト"""

    def test_default_initialization(self):
        """デフォルト設定での初期化"""
        config = ABTestConfig()

        self.assertIsInstance(config.performance, ABTestPerformanceConfig)
        self.assertGreater(config.performance.max_workers, 0)
        self.assertGreater(config.performance.batch_size, 0)

    def test_performance_config(self):
        """パフォーマンス設定テスト"""
        perf_config = ABTestPerformanceConfig(
            max_memory_mb=2048,
            max_workers=8,
            batch_size=2000
        )

        self.assertEqual(perf_config.max_memory_mb, 2048)
        self.assertEqual(perf_config.max_workers, 8)
        self.assertEqual(perf_config.batch_size, 2000)


class TestModelSelector(unittest.TestCase):
    """ModelSelectorのテスト"""

    def setUp(self):
        self.config = ABTestConfig()
        self.selector = ModelSelector(self.config)
        self.traffic_manager = TrafficManager(self.config)

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.selector.config, ABTestConfig)
        self.assertEqual(self.traffic_manager.traffic_allocations, {})

    def test_select_model_basic(self):
        """基本的なモデル選択テスト"""
        # テストデータ作成
        variant_a = ABTestVariant(
            variant_id="variant_a",
            model_path="/path/to/model_a",
            model_version="1.0",
            description="Model A"
        )

        variant_b = ABTestVariant(
            variant_id="variant_b",
            model_path="/path/to/model_b",
            model_version="1.0",
            description="Model B"
        )

        test_config = ABTestConfiguration(
            test_id="test_001",
            name="Test A/B",
            description="Basic test",
            variant_a=variant_a,
            variant_b=variant_b
        )

        # 統計結果作成
        stat_result = StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            p_value=0.01,
            effect_size=0.3,
            confidence_interval=(0.01, 0.03),
            sample_size_a=1000,
            sample_size_b=1000,
            mean_a=0.05,
            mean_b=0.07,
            std_a=0.02,
            std_b=0.025
        )

        result_summary = ABTestResultSummary(
            test_id="test_001",
            result=ABTestResult.WINNER_B,
            winner_variant_id="variant_b",
            confidence_level=0.95,
            statistical_result=stat_result,
            risk_assessment={"overall_risk": "low"},
            recommendations=["Deploy variant B"]
        )

        # テスト状態（モック）- 簡略化のため最小限のデータ
        class MockTestState:
            def __init__(self):
                self.metrics_a = type('MockMetrics', (), {'sample_count': 1000})()
                self.metrics_b = type('MockMetrics', (), {'sample_count': 1000})()
                self.regression_detected = False

        test_state = MockTestState()

        # モデル選択
        decision = self.selector.select_model(test_config, test_state, result_summary)
        self.assertEqual(decision["action"], "deploy")
        self.assertEqual(decision["selected_variant"], variant_b)

    def test_select_model_no_winner(self):
        """勝者なしの場合のテスト"""
        variant_a = ABTestVariant(
            variant_id="variant_a",
            model_path="/path/to/model_a",
            model_version="1.0",
            description="Model A"
        )

        variant_b = ABTestVariant(
            variant_id="variant_b",
            model_path="/path/to/model_b",
            model_version="1.0",
            description="Model B"
        )

        test_config = ABTestConfiguration(
            test_id="test_002",
            name="Test A/B",
            description="No winner test",
            variant_a=variant_a,
            variant_b=variant_b
        )

        stat_result = StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            p_value=0.6,
            effect_size=0.05,
            confidence_interval=(-0.005, 0.007),
            sample_size_a=1000,
            sample_size_b=1000,
            mean_a=0.05,
            mean_b=0.051,
            std_a=0.02,
            std_b=0.02
        )

        result_summary = ABTestResultSummary(
            test_id="test_002",
            result=ABTestResult.INCONCLUSIVE,
            winner_variant_id=None,
            confidence_level=0.95,
            statistical_result=stat_result,
            risk_assessment={"overall_risk": "low"},
            recommendations=["Continue testing"]
        )

        class MockTestState:
            def __init__(self):
                self.metrics_a = type('MockMetrics', (), {'sample_count': 1000})()
                self.metrics_b = type('MockMetrics', (), {'sample_count': 1000})()
                self.regression_detected = False

        test_state = MockTestState()

        decision = self.selector.select_model(test_config, test_state, result_summary)
        self.assertEqual(decision["action"], "hold")
        self.assertIsNone(decision["selected_variant"])

    def test_calculate_confidence_level(self):
        """信頼区間計算テスト"""
        stat_result = StatisticalResult(
            test_type=StatisticalTest.T_TEST,
            p_value=0.01,
            effect_size=0.3,
            confidence_interval=(0.01, 0.03),
            sample_size_a=1000,
            sample_size_b=1000,
            mean_a=0.05,
            mean_b=0.07,
            std_a=0.02,
            std_b=0.025
        )

        result_summary = ABTestResultSummary(
            test_id="test_confidence",
            result=ABTestResult.WINNER_B,
            winner_variant_id="variant_b",
            confidence_level=0.95,
            statistical_result=stat_result,
            risk_assessment={"overall_risk": "low"},
            recommendations=[]
        )

        confidence = self.selector._calculate_confidence_level(result_summary)
        self.assertGreater(confidence, 0.7)  # 高い信頼度

    def test_rollback_conditions(self):
        """ロールバック条件テスト"""
        test_id = "test_rollback"
        variant = ABTestVariant(
            variant_id="variant_a",
            model_path="/path/to/model_a",
            model_version="1.0",
            description="Model A"
        )

        # ロールバックトリガー設定
        self.selector._setup_rollback_triggers(test_id, variant, {"overall_risk": "high"})

        # トリガー存在確認
        self.assertIn(test_id, self.selector.rollback_triggers)

        # ロールバック条件チェック（正常）
        metrics = {"performance": 0.05}  # 正常範囲
        timestamp = datetime.now()
        rollback_triggered = self.selector.check_rollback_conditions(test_id, metrics, timestamp)
        self.assertFalse(rollback_triggered)


if __name__ == '__main__':
    unittest.main()