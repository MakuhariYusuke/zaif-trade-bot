"""
Test Continuous Evaluation Manager
継続的評価マネージャーのテスト
"""

import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from .evaluation_manager import ContinuousEvaluationManager
from .evaluation_types import (
    AlertLevel,
    AlertType,
    EvaluationMetrics,
    EvaluationResult,
    MonitoringAlert,
    SystemMetrics,
)


class TestContinuousEvaluationManager(unittest.TestCase):
    """継続的評価マネージャーのテスト"""

    def setUp(self):
        """テストセットアップ"""
        # モックの作成
        self.mock_monitor = Mock()
        self.mock_safety_manager = Mock()
        self.mock_drift_manager = Mock()

        # モックの設定
        self.mock_monitor.get_latest_metrics.return_value = {
            "win_rate": Mock(name="win_rate", value=0.55),
            "precision": Mock(name="precision", value=0.60),
            "recall": Mock(name="recall", value=0.50),
            "f1_score": Mock(name="f1_score", value=0.55),
            "sharpe_ratio": Mock(name="sharpe_ratio", value=1.2),
            "max_drawdown": Mock(name="max_drawdown", value=0.15),
            "total_return": Mock(name="total_return", value=0.08),
            "volatility": Mock(name="volatility", value=0.12),
        }

        self.mock_safety_manager.get_safety_status.return_value = Mock(
            overall_safety_level=Mock(value="HIGH"),
            active_anomalies=[],
            recent_checks=[],
            system_health_score=0.85,
        )

        # マネージャーの作成
        self.manager = ContinuousEvaluationManager(
            monitor=self.mock_monitor,
            safety_manager=self.mock_safety_manager,
            drift_manager=self.mock_drift_manager,
        )

    def tearDown(self):
        """テストクリーンアップ"""
        if self.manager.is_running:
            self.manager.stop_continuous_evaluation()

    def test_initialization(self):
        """初期化テスト"""
        self.assertFalse(self.manager.is_running)
        self.assertEqual(len(self.manager.evaluation_history), 0)
        self.assertEqual(len(self.manager.active_alerts), 0)
        self.assertEqual(len(self.manager.system_metrics_history), 0)

    def test_perform_evaluation_success(self):
        """評価実行成功テスト"""
        result = self.manager.perform_evaluation()

        self.assertIsInstance(result, EvaluationResult)
        self.assertIsNotNone(result.performance_metrics)
        self.assertIsNotNone(result.safety_metrics)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertGreater(result.processing_time_seconds, 0)
        self.assertIsNotNone(result.overall_score)
        self.assertIsInstance(result.recommendations, list)

    def test_perform_evaluation_with_drift(self):
        """ドリフト検知付き評価テスト"""
        # ドリフト検知のモック設定
        self.mock_drift_manager.detect_drift.return_value = [
            Mock(
                drift_detected=True,
                severity=Mock(value=4),
                drift_type=Mock(value="feature_drift"),
            )
        ]

        result = self.manager.perform_evaluation()

        self.assertTrue(result.drift_detected)
        self.assertEqual(result.drift_severity, 4)
        self.assertIn("drift_detected", [r.lower() for r in result.recommendations])

    def test_evaluate_performance(self):
        """パフォーマンス評価テスト"""
        metrics = self.manager._evaluate_performance()

        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertEqual(metrics.accuracy, 0.55)
        self.assertEqual(metrics.sharpe_ratio, 1.2)
        self.assertEqual(metrics.max_drawdown, 0.15)

    def test_evaluate_safety(self):
        """安全評価テスト"""
        safety_metrics = self.manager._evaluate_safety()

        self.assertIsInstance(safety_metrics, dict)
        self.assertIn("overall_safety_level", safety_metrics)
        self.assertIn("active_anomalies", safety_metrics)
        self.assertIn("safety_score", safety_metrics)

    def test_calculate_overall_score(self):
        """総合スコア計算テスト"""
        results = {
            "performance": EvaluationMetrics(
                accuracy=0.6,
                precision=0.65,
                recall=0.55,
                f1_score=0.6,
                sharpe_ratio=1.5,
                max_drawdown=0.1,
                total_return=0.1,
                volatility=0.1,
            ),
            "safety": {"safety_score": 0.9},
            "drift": {"drift_detected": False, "severity": 0},
        }

        score = self.manager._calculate_overall_score(results)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_generate_recommendations(self):
        """推奨事項生成テスト"""
        results = {
            "performance": EvaluationMetrics(
                accuracy=0.35,
                precision=0.4,
                recall=0.3,
                f1_score=0.35,
                sharpe_ratio=0.8,
                max_drawdown=0.3,
                total_return=-0.05,
                volatility=0.2,
            ),
            "safety": {"active_anomalies": 6, "safety_score": 0.5},
            "drift": {"drift_detected": True, "severity": 4},
        }

        recommendations = self.manager._generate_recommendations(results)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("再学習" in r for r in recommendations))

    def test_alert_generation_performance(self):
        """パフォーマンスアラート生成テスト"""
        # 低精度の評価結果を追加
        low_perf_result = EvaluationResult(
            timestamp=datetime.now(),
            performance_metrics=EvaluationMetrics(
                accuracy=0.35,
                precision=0.4,
                recall=0.3,
                f1_score=0.35,
                sharpe_ratio=0.8,
                max_drawdown=0.3,
                total_return=-0.05,
                volatility=0.2,
            ),
            overall_score=0.4,
        )
        self.manager.evaluation_history.append(low_perf_result)

        alerts = self.manager._check_performance_alerts()

        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].alert_type, AlertType.PERFORMANCE)
        self.assertEqual(alerts[0].alert_level, AlertLevel.CRITICAL)

    def test_alert_generation_safety(self):
        """安全アラート生成テスト"""
        # 異常の多い安全ステータスを設定
        self.mock_safety_manager.get_safety_status.return_value = Mock(
            overall_safety_level=Mock(value="HIGH"),
            active_anomalies=[Mock()] * 6,  # 6つの異常
            recent_checks=[],
            system_health_score=0.4,
        )

        alerts = self.manager._check_safety_alerts()

        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].alert_type, AlertType.SAFETY)

    def test_alert_generation_drift(self):
        """ドリフトアラート生成テスト"""
        # ドリフト検知結果を設定
        drift_result = EvaluationResult(
            timestamp=datetime.now(),
            drift_detected=True,
            drift_severity=4,
            overall_score=0.6,
        )
        self.manager.evaluation_history.append(drift_result)

        alerts = self.manager._check_drift_alerts()

        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].alert_type, AlertType.DRIFT)

    def test_alert_resolution(self):
        """アラート解決テスト"""
        # 解決されたアラートを作成
        resolved_alert = MonitoringAlert(
            alert_id="test_alert",
            alert_type=AlertType.PERFORMANCE,
            alert_level=AlertLevel.HIGH,
            message="Test alert",
            timestamp=datetime.now() - timedelta(minutes=10),
            details={"accuracy": 0.35},
        )

        # 現在のパフォーマンスが改善されていることをシミュレート
        improved_result = EvaluationResult(
            timestamp=datetime.now(),
            performance_metrics=EvaluationMetrics(
                accuracy=0.6,
                precision=0.65,
                recall=0.55,
                f1_score=0.6,
                sharpe_ratio=1.5,
                max_drawdown=0.1,
                total_return=0.1,
                volatility=0.1,
            ),
            overall_score=0.8,
        )
        self.manager.evaluation_history.append(improved_result)

        self.assertTrue(self.manager._is_alert_resolved(resolved_alert))

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_connections")
    @patch("threading.active_count")
    def test_collect_system_metrics(
        self, mock_threads, mock_net, mock_disk, mock_memory, mock_cpu
    ):
        """システムメトリクス収集テスト"""
        # モックの設定
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(percent=67.8)
        mock_disk.return_value = Mock(percent=23.4)
        mock_net.return_value = [Mock()] * 15
        mock_threads.return_value = 8

        self.manager._collect_system_metrics()

        self.assertEqual(len(self.manager.system_metrics_history), 1)
        metrics = self.manager.system_metrics_history[0]

        self.assertIsInstance(metrics, SystemMetrics)
        self.assertEqual(metrics.cpu_usage, 45.5)
        self.assertEqual(metrics.memory_usage, 67.8)
        self.assertEqual(metrics.disk_usage, 23.4)
        self.assertEqual(metrics.network_connections, 15)
        self.assertEqual(metrics.active_threads, 8)

    def test_get_evaluation_summary(self):
        """評価サマリーテスト"""
        # 評価履歴を追加
        for i in range(5):
            result = EvaluationResult(
                timestamp=datetime.now() - timedelta(hours=i),
                overall_score=0.7 + i * 0.05,
                drift_detected=i % 2 == 0,
            )
            self.manager.evaluation_history.append(result)

        summary = self.manager.get_evaluation_summary(hours=24)

        self.assertIn("total_evaluations", summary)
        self.assertIn("average_score", summary)
        self.assertIn("drift_rate", summary)
        self.assertEqual(summary["total_evaluations"], 5)

    def test_callback_system(self):
        """コールバックシステムテスト"""
        alert_callback = Mock()
        evaluation_callback = Mock()

        self.manager.add_alert_callback(alert_callback)
        self.manager.add_evaluation_callback(evaluation_callback)

        # アラートをトリガー
        alert = MonitoringAlert(
            alert_id="test",
            alert_type=AlertType.SYSTEM,
            alert_level=AlertLevel.MEDIUM,
            message="Test alert",
            timestamp=datetime.now(),
        )
        self.manager._trigger_alert_callbacks(alert)

        # 評価をトリガー
        result = EvaluationResult(timestamp=datetime.now(), overall_score=0.8)
        self.manager._trigger_evaluation_callbacks(result)

        alert_callback.assert_called_once_with(alert)
        evaluation_callback.assert_called_once_with(result)

    def test_continuous_evaluation_start_stop(self):
        """継続的評価の開始・停止テスト"""
        # 短い間隔でテストするために設定を変更
        self.manager.evaluation_interval_seconds = 0.1
        self.manager.alert_check_interval_seconds = 0.1

        success = self.manager.start_continuous_evaluation()
        self.assertTrue(success)
        self.assertTrue(self.manager.is_running)

        # 少し待ってから停止
        time.sleep(0.5)
        self.manager.stop_continuous_evaluation()

        self.assertFalse(self.manager.is_running)
        self.assertGreater(len(self.manager.evaluation_history), 0)


if __name__ == "__main__":
    unittest.main()
