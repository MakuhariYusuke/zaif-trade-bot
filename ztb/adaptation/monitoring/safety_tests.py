"""
Unit tests for Safety Mechanisms and Fallback System
安全メカニズムとフォールバックシステムのテスト
"""

import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from ztb.adaptation.monitoring.safety import (
    AnomalyDetector,
    FallbackHandler,
    RecoveryManager,
    SafetyChecker,
    SafetyConfig,
    SafetyManager,
)
from ztb.adaptation.monitoring.types import (
    AnomalyType,
    FallbackType,
    MetricType,
    MetricValue,
    SafetyLevel,
    SafetyStatus,
)


class TestAnomalyDetector(unittest.TestCase):
    """AnomalyDetectorのテスト"""

    def setUp(self):
        self.config = SafetyConfig()
        self.detector = AnomalyDetector(self.config)

    def test_anomaly_detection_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.detector.metric_history, dict)
        self.assertIsInstance(self.detector.baseline_stats, dict)

    def test_statistical_anomaly_detection(self):
        """統計的異常検知テスト"""
        metric_name = "test_metric"

        # 正常なデータを追加
        for i in range(100):
            value = 50.0 + np.random.normal(0, 2)  # 平均50、標準偏差2
            metric = MetricValue(
                name=metric_name,
                value=value,
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
            )
            self.detector.metric_history[metric_name].append(value)

        # ベースライン統計更新
        self.detector._update_baseline_stats(metric_name)

        # 正常値のテスト
        normal_metric = MetricValue(
            name=metric_name,
            value=52.0,
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
        )
        anomalies = self.detector.detect_anomalies({metric_name: normal_metric})
        self.assertEqual(len(anomalies), 0)

        # 異常値のテスト（3σ以上）
        anomaly_metric = MetricValue(
            name=metric_name,
            value=70.0,
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,  # 平均+10σ
        )
        anomalies = self.detector.detect_anomalies({metric_name: anomaly_metric})
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].anomaly_type, AnomalyType.STATISTICAL)
        self.assertEqual(anomalies[0].metric_name, metric_name)


class TestSafetyChecker(unittest.TestCase):
    """SafetyCheckerのテスト"""

    def setUp(self):
        self.config = SafetyConfig()
        self.checker = SafetyChecker(self.config)

    def test_safety_checks_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.checker.check_functions, dict)
        self.assertGreater(len(self.checker.check_functions), 0)

    def test_system_health_check(self):
        """システムヘルスチェックテスト"""
        # 正常なメトリクス
        normal_metrics = {
            "cpu_usage_percent": MetricValue(
                name="cpu_usage_percent",
                value=50.0,
                timestamp=datetime.now(),
                metric_type=MetricType.SYSTEM,
            ),
            "memory_usage_percent": MetricValue(
                name="memory_usage_percent",
                value=60.0,
                timestamp=datetime.now(),
                metric_type=MetricType.SYSTEM,
            ),
        }

        check = self.checker._check_system_health(normal_metrics, [])
        self.assertTrue(check.passed)
        self.assertEqual(check.safety_level, SafetyLevel.NORMAL)

        # 高CPU使用率
        high_cpu_metrics = normal_metrics.copy()
        high_cpu_metrics["cpu_usage_percent"] = MetricValue(
            name="cpu_usage_percent",
            value=95.0,
            timestamp=datetime.now(),
            metric_type=MetricType.SYSTEM,
        )

        check = self.checker._check_system_health(high_cpu_metrics, [])
        self.assertFalse(check.passed)
        self.assertEqual(check.safety_level, SafetyLevel.CRITICAL)

    def test_performance_stability_check(self):
        """パフォーマンス安定性チェックテスト"""
        # 正常なメトリクス
        normal_metrics = {
            "win_rate": MetricValue(
                name="win_rate",
                value=0.6,
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
            ),
            "max_drawdown": MetricValue(
                name="max_drawdown",
                value=0.1,
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
            ),
        }

        check = self.checker._check_performance_stability(normal_metrics, [])
        self.assertTrue(check.passed)
        self.assertEqual(check.safety_level, SafetyLevel.NORMAL)

        # 低勝率
        low_winrate_metrics = normal_metrics.copy()
        low_winrate_metrics["win_rate"] = MetricValue(
            name="win_rate",
            value=0.3,
            timestamp=datetime.now(),
            metric_type=MetricType.PERFORMANCE,
        )

        check = self.checker._check_performance_stability(low_winrate_metrics, [])
        self.assertFalse(check.passed)
        self.assertEqual(check.safety_level, SafetyLevel.CRITICAL)


class TestFallbackHandler(unittest.TestCase):
    """FallbackHandlerのテスト"""

    def setUp(self):
        self.config = SafetyConfig()
        self.handler = FallbackHandler(self.config)

    def test_fallback_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.handler.active_fallbacks, dict)
        self.assertIsInstance(self.handler.fallback_functions, dict)

    def test_gradual_fallback_initiation(self):
        """段階的フォールバック開始テスト"""
        reason = "Test gradual fallback"
        action_id = self.handler.initiate_fallback(FallbackType.GRADUAL, reason)

        self.assertIn(action_id, self.handler.active_fallbacks)
        action = self.handler.active_fallbacks[action_id]
        self.assertEqual(action.fallback_type, FallbackType.GRADUAL)
        self.assertIn(reason, action.description)
        self.assertGreater(len(action.rollback_steps), 0)

    def test_immediate_fallback_initiation(self):
        """即時フォールバック開始テスト"""
        reason = "Test immediate fallback"
        action_id = self.handler.initiate_fallback(FallbackType.IMMEDIATE, reason)

        self.assertIn(action_id, self.handler.active_fallbacks)
        action = self.handler.active_fallbacks[action_id]
        self.assertEqual(action.fallback_type, FallbackType.IMMEDIATE)
        self.assertEqual(len(action.rollback_steps), 4)  # 即時ロールバックのステップ数

    def test_fallback_cancellation(self):
        """フォールバックキャンセルテスト"""
        action_id = self.handler.initiate_fallback(FallbackType.CONSERVATIVE, "Test")

        # キャンセル
        result = self.handler.cancel_fallback(action_id)
        self.assertTrue(result)
        self.assertNotIn(action_id, self.handler.active_fallbacks)

        # 存在しないIDのキャンセル
        result = self.handler.cancel_fallback("nonexistent")
        self.assertFalse(result)


class TestRecoveryManager(unittest.TestCase):
    """RecoveryManagerのテスト"""

    def setUp(self):
        self.config = SafetyConfig()
        self.manager = RecoveryManager(self.config)

    def test_recovery_plan_creation(self):
        """回復計画作成テスト"""
        trigger_reason = "Test recovery"
        steps = ["Step 1", "Step 2", "Step 3"]
        success_criteria = ["Criteria 1", "Criteria 2"]

        plan_id = self.manager.create_recovery_plan(
            trigger_reason, steps, success_criteria
        )

        self.assertIn(plan_id, self.manager.recovery_plans)
        plan = self.manager.recovery_plans[plan_id]
        self.assertEqual(plan.triggered_by, trigger_reason)
        self.assertEqual(plan.steps, steps)
        self.assertEqual(plan.success_criteria, success_criteria)

    @patch("ztb.adaptation.monitoring.safety.time.sleep")
    def test_recovery_execution_success(self, mock_sleep):
        """回復実行成功テスト"""
        plan_id = self.manager.create_recovery_plan(
            "Test recovery", ["Step 1", "Step 2"], ["Success criteria"]
        )

        result = self.manager.execute_recovery(plan_id)
        self.assertTrue(result)

        # time.sleepが呼ばれたことを確認
        self.assertEqual(mock_sleep.call_count, 2)

    def test_recovery_attempt_limit(self):
        """回復試行回数制限テスト"""
        trigger_reason = "Test failure"
        plan_id = self.manager.create_recovery_plan(
            trigger_reason, ["Recovery step"], ["Success criteria"]
        )

        # 回復が成功するはず
        result = self.manager.execute_recovery(plan_id)
        self.assertTrue(result)

        # 試行回数がリセットされていることを確認
        self.assertEqual(self.manager.recovery_attempts[trigger_reason], 0)


class TestSafetyManager(unittest.TestCase):
    """SafetyManagerのテスト"""

    def setUp(self):
        self.config = SafetyConfig()
        self.manager = SafetyManager(self.config)

    def tearDown(self):
        if (
            hasattr(self.manager, "monitoring_thread")
            and self.manager.monitoring_thread
        ):
            self.manager.stop_safety_monitoring()

    def test_safety_manager_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.manager.anomaly_detector, AnomalyDetector)
        self.assertIsInstance(self.manager.safety_checker, SafetyChecker)
        self.assertIsInstance(self.manager.fallback_handler, FallbackHandler)
        self.assertIsInstance(self.manager.recovery_manager, RecoveryManager)
        self.assertEqual(
            self.manager.current_status.overall_safety_level, SafetyLevel.NORMAL
        )

    def test_manual_fallback_initiation(self):
        """手動フォールバック開始テスト"""
        action_id = self.manager.initiate_manual_fallback(
            FallbackType.CONSERVATIVE, "Manual test fallback"
        )

        self.assertIsInstance(action_id, str)
        self.assertIn(action_id, self.manager.fallback_handler.active_fallbacks)

    def test_recovery_plan_creation_and_execution(self):
        """回復計画作成と実行テスト"""
        plan_id = self.manager.create_recovery_plan(
            "Test recovery",
            ["Recovery step 1", "Recovery step 2"],
            ["Recovery criteria"],
        )

        result = self.manager.execute_recovery(plan_id)
        self.assertTrue(result)

    def test_safety_status_retrieval(self):
        """安全ステータス取得テスト"""
        status = self.manager.get_safety_status()

        self.assertIsInstance(status, SafetyStatus)
        self.assertIsInstance(status.overall_safety_level, SafetyLevel)
        self.assertIsInstance(status.last_updated, datetime)
        self.assertGreaterEqual(status.system_health_score, 0.0)
        self.assertLessEqual(status.system_health_score, 1.0)


if __name__ == "__main__":
    unittest.main()
