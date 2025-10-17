"""
Integration Test for Continuous Evaluation Manager
継続的評価マネージャーの統合テスト
"""

import unittest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from ztb.adaptation import (
    ContinuousEvaluationManager,
    ContinuousMonitoringConfig,
    EvaluationResult,
    MonitoringAlert,
    AlertType,
    AlertLevel
)
from ztb.adaptation.monitoring.config import AlertThreshold


class TestContinuousEvaluationIntegration(unittest.TestCase):
    """継続的評価マネージャーの統合テスト"""

    def setUp(self):
        """テストセットアップ"""
        # モックの作成
        self.mock_monitor = Mock()
        self.mock_safety_manager = Mock()
        self.mock_drift_manager = Mock()

        # モックの設定
        self.mock_monitor.get_latest_metrics.return_value = {
            'win_rate': Mock(name='win_rate', value=0.55),
            'precision': Mock(name='precision', value=0.60),
            'sharpe_ratio': Mock(name='sharpe_ratio', value=1.2),
            'max_drawdown': Mock(name='max_drawdown', value=0.15),
        }

        self.mock_safety_manager.get_safety_status.return_value = Mock(
            overall_safety_level=Mock(value="HIGH"),
            active_anomalies=[],  # 空のリスト
            recent_checks=[],     # 空のリスト
            system_health_score=0.85
        )

        self.mock_drift_manager.detect_drift.return_value = []  # 空のリスト

    def test_full_evaluation_cycle(self):
        """完全な評価サイクルのテスト"""
        # マネージャーの作成
        manager = ContinuousEvaluationManager(
            monitor=self.mock_monitor,
            safety_manager=self.mock_safety_manager,
            drift_manager=self.mock_drift_manager
        )

        # 評価実行
        result = manager.perform_evaluation()

        # 結果の検証
        self.assertIsInstance(result, EvaluationResult)
        self.assertIsNotNone(result.performance_metrics)
        self.assertIsNotNone(result.safety_metrics)
        self.assertIsNotNone(result.overall_score)
        self.assertGreater(len(result.recommendations), 0)

        # 履歴に追加されていることを確認
        self.assertEqual(len(manager.evaluation_history), 1)

    def test_continuous_monitoring_workflow(self):
        """継続的監視ワークフローのテスト"""
        manager = ContinuousEvaluationManager(
            monitor=self.mock_monitor,
            safety_manager=self.mock_safety_manager,
            drift_manager=self.mock_drift_manager
        )

        # 短い間隔でテスト
        manager.evaluation_interval_seconds = 0.1
        manager.alert_check_interval_seconds = 0.1

        # 継続的評価開始
        success = manager.start_continuous_evaluation()
        self.assertTrue(success)
        self.assertTrue(manager.is_running)

        # 少し待って評価が実行されることを確認
        time.sleep(0.5)

        # 評価が実行されていることを確認
        self.assertGreater(len(manager.evaluation_history), 0)

        # 停止
        manager.stop_continuous_evaluation()
        self.assertFalse(manager.is_running)

    def test_alert_system_integration(self):
        """アラートシステム統合テスト"""
        manager = ContinuousEvaluationManager(
            monitor=self.mock_monitor,
            safety_manager=self.mock_safety_manager,
            drift_manager=self.mock_drift_manager
        )

        # アラートコールバックの設定
        alerts_received = []
        def alert_callback(alert):
            alerts_received.append(alert)

        manager.add_alert_callback(alert_callback)

        # 低パフォーマンスの評価結果を追加
        low_perf_result = EvaluationResult(
            timestamp=datetime.now(),
            performance_metrics=Mock(
                accuracy=0.35,
                sharpe_ratio=0.8,
                max_drawdown=0.3
            ),
            overall_score=0.4
        )
        manager.evaluation_history.append(low_perf_result)

        # アラートチェック
        manager._check_and_generate_alerts()

        # アラートが生成されていることを確認
        self.assertGreater(len(manager.active_alerts), 0)
        self.assertEqual(len(alerts_received), len(manager.active_alerts))

    def test_evaluation_summary_generation(self):
        """評価サマリー生成テスト"""
        manager = ContinuousEvaluationManager(
            monitor=self.mock_monitor,
            safety_manager=self.mock_safety_manager,
            drift_manager=self.mock_drift_manager
        )

        # 複数の評価結果を追加
        for i in range(5):
            result = EvaluationResult(
                timestamp=datetime.now() - timedelta(hours=i),
                overall_score=0.7 + i * 0.05,
                drift_detected=i % 2 == 0
            )
            manager.evaluation_history.append(result)

        # サマリー生成
        summary = manager.get_evaluation_summary(hours=24)

        # サマリーの検証
        self.assertIn('total_evaluations', summary)
        self.assertIn('average_score', summary)
        self.assertIn('drift_rate', summary)
        self.assertEqual(summary['total_evaluations'], 5)

    def test_system_health_monitoring(self):
        """システム健全性監視テスト"""
        manager = ContinuousEvaluationManager(
            monitor=self.mock_monitor,
            safety_manager=self.mock_safety_manager,
            drift_manager=self.mock_drift_manager
        )

        # システムメトリクス収集をシミュレート
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[Mock()] * 10), \
             patch('threading.active_count', return_value=8):

            mock_memory.return_value = Mock(percent=67.8)
            mock_disk.return_value = Mock(percent=23.4)

            manager._collect_system_metrics()

            # メトリクスが収集されていることを確認
            self.assertEqual(len(manager.system_metrics_history), 1)
            metrics = manager.system_metrics_history[0]
            self.assertEqual(metrics.cpu_usage, 45.5)
            self.assertEqual(metrics.memory_usage, 67.8)

    def test_adaptation_module_integration(self):
        """適応モジュール統合テスト"""
        # 適応モジュールからのインポートを確認
        try:
            from ztb.adaptation import ContinuousEvaluationManager as CEM
            from ztb.adaptation import ContinuousMonitoringConfig as CMC

            # クラスが正しくインポートされていることを確認
            self.assertEqual(CEM, ContinuousEvaluationManager)
            self.assertEqual(CMC, ContinuousMonitoringConfig)

        except ImportError as e:
            self.fail(f"Failed to import from adaptation module: {e}")

    def test_configuration_integration(self):
        """設定統合テスト"""
        # デフォルト設定の使用
        config = ContinuousMonitoringConfig()

        self.assertIsNotNone(config.evaluation)
        self.assertIsNotNone(config.alerts)
        self.assertTrue(config.enable_monitoring)

        # 設定値の検証
        self.assertEqual(config.evaluation.evaluation_interval_seconds, 60)
        self.assertEqual(config.alerts.alert_levels[AlertThreshold.LOW]['enabled'], True)


if __name__ == '__main__':
    unittest.main()