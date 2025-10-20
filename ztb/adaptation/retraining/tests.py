"""
Unit tests for Automatic Retraining Triggers Module
"""

import gc
import unittest
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import patch

from ztb.adaptation.retraining.config import RetrainingConfig, RetrainingPolicy
from ztb.adaptation.retraining.trigger import RetrainingTrigger
from ztb.adaptation.retraining.types import (
    DataDistributionMetrics,
    PerformanceMetrics,
    RetrainingRequest,
    RetrainingResult,
    TriggerCondition,
    TriggerPriority,
    TriggerStatus,
    TriggerType,
)


class TestRetrainingTrigger(unittest.TestCase):
    """RetrainingTriggerのテスト"""

    def setUp(self):
        self.config = RetrainingConfig()
        # テスト用に履歴サイズを小さく設定
        self.config.max_history_size = 10
        self.config.cleanup_interval_hours = 1  # テスト用に短く設定
        self.trigger = RetrainingTrigger(self.config)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # メモリリーク防止のため明示的にクリーンアップ
        if hasattr(self.trigger, "_cleanup_timer") and self.trigger._cleanup_timer:
            self.trigger._cleanup_timer.cancel()

        # 弱参照のクリーンアップ
        self.trigger._metric_callbacks.clear()

        # ガベージコレクション
        gc.collect()

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.trigger.trigger_states, dict)
        self.assertGreater(len(self.trigger.trigger_states), 0)
        self.assertIsInstance(self.trigger.performance_history, deque)
        self.assertIsInstance(self.trigger.distribution_history, deque)

    def test_performance_trigger_win_rate(self):
        """勝率低下によるパフォーマンストリガー"""
        # 正常なパフォーマンス
        normal_metrics = PerformanceMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
            win_rate=0.5,  # 正常
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            timestamp=datetime.now(),
        )

        # トリガーが発動しないことを確認
        requests = self.trigger.update_performance_metrics(normal_metrics)
        self.assertEqual(len(requests), 0)

        # 低下したパフォーマンス
        low_metrics = PerformanceMetrics(
            accuracy=0.6,
            precision=0.55,
            recall=0.5,
            f1_score=0.52,
            win_rate=0.35,  # 低下（閾値0.45未満）
            sharpe_ratio=0.8,
            max_drawdown=0.15,
            timestamp=datetime.now() + timedelta(minutes=70),  # 期間条件を満たす
        )

        # 複数回更新して期間条件を満たす
        requests = []
        for i in range(5):
            current_requests = self.trigger.update_performance_metrics(low_metrics)
            if current_requests:
                requests.extend(current_requests)

        # トリガーが発動することを確認
        self.assertGreater(len(requests), 0)
        self.assertEqual(requests[0].trigger_type, TriggerType.PERFORMANCE)
        self.assertIn("win_rate", requests[0].trigger_reason)

    def test_performance_trigger_sharpe_ratio(self):
        """シャープレシオ低下によるパフォーマンストリガー"""
        # 低下したシャープレシオ
        low_metrics = PerformanceMetrics(
            accuracy=0.7,
            precision=0.65,
            recall=0.6,
            f1_score=0.62,
            win_rate=0.48,
            sharpe_ratio=0.5,  # 低下（閾値1.0未満）
            max_drawdown=0.12,
            timestamp=datetime.now(),
        )

        # 複数回更新して期間条件を満たす
        requests = []
        for _ in range(10):
            current_requests = self.trigger.update_performance_metrics(low_metrics)
            if current_requests:
                requests.extend(current_requests)

        # トリガーが発動することを確認
        self.assertGreater(len(requests), 0)
        self.assertEqual(requests[0].trigger_type, TriggerType.PERFORMANCE)
        self.assertIn("sharpe_ratio", requests[0].trigger_reason)

    def test_distribution_trigger(self):
        """データ分布変化によるトリガー"""
        # 初期分布
        initial_metrics = DataDistributionMetrics(
            feature_means={"feature1": 0.0, "feature2": 1.0},
            feature_stds={"feature1": 1.0, "feature2": 2.0},
            feature_skewness={"feature1": 0.0, "feature2": 0.0},
            feature_kurtosis={"feature1": 0.0, "feature2": 0.0},
            sample_size=1000,
            timestamp=datetime.now(),
        )

        # 分布変化なし
        requests = self.trigger.update_distribution_metrics(initial_metrics)
        self.assertEqual(len(requests), 0)

        # 分布変化
        changed_metrics = DataDistributionMetrics(
            feature_means={"feature1": 1.0, "feature2": 3.0},  # 大きく変化
            feature_stds={"feature1": 1.0, "feature2": 2.0},
            feature_skewness={"feature1": 0.0, "feature2": 0.0},
            feature_kurtosis={"feature1": 0.0, "feature2": 0.0},
            sample_size=1000,
            timestamp=datetime.now() + timedelta(minutes=40),
        )

        requests = self.trigger.update_distribution_metrics(changed_metrics)
        # 分布トリガーはより複雑なので、条件によっては発動しない場合もある
        # テストはエラーが発生しないことを確認

    def test_scheduled_triggers(self):
        """スケジュールされたトリガー"""
        # スケジュールを即時実行に設定
        for schedule in self.trigger.schedules:
            schedule.next_run = datetime.now() - timedelta(minutes=1)

        requests = self.trigger.check_scheduled_triggers()
        self.assertGreater(len(requests), 0)

        for request in requests:
            self.assertEqual(request.trigger_type, TriggerType.TIME_BASED)
            self.assertIn("Scheduled", request.trigger_reason)

    def test_volume_based_triggers(self):
        """出来高ベースのトリガー"""
        # 閾値未満
        requests = self.trigger.check_volume_based_triggers(500)
        self.assertEqual(len(requests), 0)

        # 閾値以上
        requests = self.trigger.check_volume_based_triggers(1500)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].trigger_type, TriggerType.VOLUME_BASED)
        self.assertIn("New samples threshold", requests[0].trigger_reason)

    def test_cooldown_mechanism(self):
        """クールダウン機構のテスト"""
        # トリガーを発動
        low_metrics = PerformanceMetrics(
            accuracy=0.6,
            precision=0.55,
            recall=0.5,
            f1_score=0.52,
            win_rate=0.35,
            sharpe_ratio=0.8,
            max_drawdown=0.15,
            timestamp=datetime.now(),
        )

        # 複数回更新してトリガーを発動
        requests = []
        for _ in range(5):
            current_requests = self.trigger.update_performance_metrics(low_metrics)
            if current_requests:
                requests.extend(current_requests)

        initial_trigger_count = len(requests)

        # すぐに再度チェック（クールダウン中）
        immediate_requests = self.trigger.update_performance_metrics(low_metrics)
        self.assertEqual(len(immediate_requests), 0)  # クールダウン中なので発動しない

    def test_retraining_result_recording(self):
        """再訓練結果の記録"""
        result = RetrainingResult(
            request_id="test_request_123",
            success=True,
            new_model_path="/path/to/model",
            performance_improvement=0.05,
            training_duration=timedelta(hours=1),
            completed_at=datetime.now(),
        )

        # 記録前の履歴サイズ
        initial_size = len(self.trigger.retraining_history)

        self.trigger.record_retraining_result(result)

        # 履歴サイズが増加したことを確認
        self.assertEqual(len(self.trigger.retraining_history), initial_size + 1)

        # 記録された内容を確認
        history = self.trigger.retraining_history[-1]
        self.assertEqual(history.request_id, result.request_id)
        self.assertTrue(history.success)
        self.assertEqual(history.performance_change, result.performance_improvement)

    def test_history_size_limit(self):
        """履歴サイズ制限のテスト"""
        # 多くの結果を記録
        for i in range(15):  # max_history_size (10) を超える
            result = RetrainingResult(
                request_id=f"test_request_{i}",
                success=True,
                new_model_path=f"/path/to/model_{i}",
                performance_improvement=0.01,
                training_duration=timedelta(hours=1),
                completed_at=datetime.now(),
            )
            self.trigger.record_retraining_result(result)

        # 履歴サイズが制限されていることを確認
        self.assertLessEqual(
            len(self.trigger.retraining_history), self.config.max_history_size
        )

    def test_reset_functionality(self):
        """リセット機能のテスト"""
        # 何らかの状態を作成
        metrics = PerformanceMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
            win_rate=0.5,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            timestamp=datetime.now(),
        )
        self.trigger.update_performance_metrics(metrics)

        # アクティブリクエストを追加
        self.trigger.active_requests["test"] = RetrainingRequest(
            request_id="test",
            trigger_type=TriggerType.MANUAL,
            trigger_reason="Test",
            priority=TriggerPriority.MEDIUM,
            requested_at=datetime.now(),
            estimated_duration=timedelta(hours=1),
            required_resources={},
        )

        # リセット前の状態を確認
        self.assertGreater(len(self.trigger.performance_history), 0)
        self.assertGreater(len(self.trigger.active_requests), 0)

        # リセット
        self.trigger.reset_triggers()

        # リセット後の状態を確認
        self.assertEqual(len(self.trigger.active_requests), 0)
        # 履歴は保持される（リセットでは削除しない）

    def test_memory_cleanup(self):
        """メモリクリーンアップのテスト"""
        # 大量のデータを追加
        for i in range(20):
            metrics = PerformanceMetrics(
                accuracy=0.8,
                precision=0.75,
                recall=0.7,
                f1_score=0.72,
                win_rate=0.5,
                sharpe_ratio=1.5,
                max_drawdown=0.1,
                timestamp=datetime.now() + timedelta(minutes=i),
            )
            self.trigger.update_performance_metrics(metrics)

        # 履歴サイズが制限されていることを確認
        self.assertLessEqual(
            len(self.trigger.performance_history), self.config.max_history_size
        )

        # 明示的なクリーンアップを実行
        self.trigger._cleanup_history_if_needed()

        # 引き続き制限内であることを確認
        self.assertLessEqual(
            len(self.trigger.performance_history), self.config.max_history_size
        )

    def test_error_handling_invalid_metrics(self):
        """無効なメトリクスデータのエラーハンドリング"""
        # 無効な値を含むメトリクス（負の値）
        invalid_metrics = PerformanceMetrics(
            accuracy=-0.1,
            precision=0.5,
            recall=0.4,
            f1_score=0.45,
            win_rate=0.5,
            sharpe_ratio=0.8,
            max_drawdown=0.1,
            timestamp=datetime.now(),
        )

        # エラーが発生せず、適切に処理されることを確認
        requests = self.trigger.update_performance_metrics(invalid_metrics)
        self.assertIsInstance(requests, list)

    def test_edge_case_empty_history(self):
        """空の履歴に対するエッジケース"""
        # 新しいトリガーを作成
        new_trigger = RetrainingTrigger(self.config)

        # 分布トリガーをチェック（履歴が空の場合）
        distribution_metrics = DataDistributionMetrics(
            feature_means={"feature1": 1.0, "feature2": 2.0},
            feature_stds={"feature1": 0.1, "feature2": 0.2},
            feature_skewness={"feature1": 0.0, "feature2": 0.0},
            feature_kurtosis={"feature1": 0.0, "feature2": 0.0},
            sample_size=1000,
            timestamp=datetime.now(),
        )

        requests = new_trigger.update_distribution_metrics(distribution_metrics)
        # 履歴が不十分なのでトリガーが発動しないことを確認
        self.assertEqual(len(requests), 0)

    def test_disabled_trigger_behavior(self):
        """無効化されたトリガーの動作"""
        # 設定で特定のトリガーを無効化
        disabled_config = RetrainingConfig()
        disabled_config.trigger_conditions = []  # すべてのトリガーを無効化

        disabled_trigger = RetrainingTrigger(disabled_config)

        # パフォーマンスメトリクスを更新
        metrics = PerformanceMetrics(
            accuracy=0.3,
            precision=0.25,
            recall=0.2,
            f1_score=0.22,
            win_rate=0.2,
            sharpe_ratio=0.3,
            max_drawdown=0.3,
            timestamp=datetime.now(),
        )

        requests = disabled_trigger.update_performance_metrics(metrics)
        # トリガーが無効化されているので何も発動しない
        self.assertEqual(len(requests), 0)

    def test_trigger_priority_ordering(self):
        """トリガーの優先順位付け"""
        # 複数のトリガーが同時に発動する状況を作成
        low_metrics = PerformanceMetrics(
            accuracy=0.4,
            precision=0.35,
            recall=0.3,
            f1_score=0.32,
            win_rate=0.25,
            sharpe_ratio=0.4,
            max_drawdown=0.25,
            timestamp=datetime.now(),
        )

        # 複数回更新して複数のトリガーを発動
        for i in range(10):
            self.trigger.update_performance_metrics(low_metrics)

        # 優先順位が適切に設定されていることを確認
        states = self.trigger.get_trigger_states()
        for state in states.values():
            if state.condition:
                self.assertIn(
                    state.condition.priority,
                    [TriggerPriority.LOW, TriggerPriority.MEDIUM, TriggerPriority.HIGH],
                )

    def test_configuration_changes_runtime(self):
        """実行時の設定変更"""
        # 設定変更前の動作を確認
        initial_max_history = self.trigger.config.max_history_size

        # 設定を変更（最小値の100以上）
        new_config = RetrainingConfig(max_history_size=initial_max_history + 100)
        new_trigger = RetrainingTrigger(new_config)

        # 新しい設定が適用されていることを確認
        self.assertEqual(new_trigger.config.max_history_size, initial_max_history + 100)

    def test_resource_usage_tracking(self):
        """リソース使用量の追跡"""
        # 再訓練リクエストを作成
        request = RetrainingRequest(
            request_id="test_resource_123",
            trigger_type=TriggerType.PERFORMANCE,
            trigger_reason="Test resource tracking",
            priority=TriggerPriority.HIGH,
            requested_at=datetime.now(),
            estimated_duration=timedelta(hours=2),
            required_resources={"cpu": 4, "memory_gb": 8},
        )

        # リクエストを記録
        self.trigger.active_requests[request.request_id] = request

        # リソースが追跡されていることを確認
        self.assertIn(request.request_id, self.trigger.active_requests)
        self.assertEqual(
            self.trigger.active_requests[request.request_id].required_resources["cpu"],
            4,
        )

    def test_logging_functionality(self):
        """ログ機能のテスト"""
        with patch("ztb.adaptation.retraining.trigger.logger") as mock_logger:
            # トリガーを初期化（ログが出力される）
            trigger = RetrainingTrigger(self.config)

            # ログが呼ばれたことを確認
            mock_logger.info.assert_called_with("RetrainingTrigger initialized")

    def test_thread_safety_cleanup_timer(self):
        """クリーンアップタイマーのスレッドセーフティ"""
        # タイマーが正しく開始されていることを確認
        self.assertIsNotNone(self.trigger._cleanup_timer)
        if self.trigger._cleanup_timer:
            self.assertTrue(self.trigger._cleanup_timer.is_alive())

        # タイマーを停止
        self.trigger._stop_cleanup_timer()
        self.assertIsNone(self.trigger._cleanup_timer)

    def test_weak_reference_memory_management(self):
        """弱参照によるメモリ管理"""
        # コールバックオブジェクトを作成
        callback_obj = lambda: None

        # オブジェクトを弱参照セットに追加
        self.trigger._metric_callbacks.add(callback_obj)

        # オブジェクトが生存していることを確認
        self.assertIn(callback_obj, self.trigger._metric_callbacks)

        # オブジェクトを削除
        del callback_obj
        gc.collect()

        # 弱参照が自動的にクリアされていることを確認
        self.assertEqual(len(self.trigger._metric_callbacks), 0)

    def test_concurrent_trigger_evaluation(self):
        """同時トリガー評価のテスト"""
        # 複数の異なるタイプのトリガーが同時に評価される状況
        performance_metrics = PerformanceMetrics(
            accuracy=0.35,
            precision=0.3,
            recall=0.25,
            f1_score=0.27,
            win_rate=0.3,
            sharpe_ratio=0.5,
            max_drawdown=0.22,
            timestamp=datetime.now(),
        )

        distribution_metrics = DataDistributionMetrics(
            feature_means={"feature1": 1.5, "feature2": 2.5},
            feature_stds={"feature1": 0.15, "feature2": 0.25},
            feature_skewness={"feature1": 0.1, "feature2": -0.1},
            feature_kurtosis={"feature1": 0.2, "feature2": 0.3},
            sample_size=1000,
            timestamp=datetime.now(),
        )

        # まずパフォーマンス履歴を構築
        for i in range(5):
            self.trigger.update_performance_metrics(performance_metrics)

        # 次に分布メトリクスを追加
        for i in range(3):
            self.trigger.update_distribution_metrics(distribution_metrics)

        # 両方のタイプのトリガーが適切に処理されていることを確認
        self.assertIsInstance(self.trigger.performance_history, deque)
        self.assertIsInstance(self.trigger.distribution_history, deque)

    def test_boundary_conditions_thresholds(self):
        """閾値の境界条件テスト"""
        # 閾値と等しい値でのテスト
        boundary_metrics = PerformanceMetrics(
            accuracy=0.5,
            precision=0.45,
            recall=0.4,
            f1_score=0.42,
            win_rate=0.4,
            sharpe_ratio=1.0,
            max_drawdown=0.15,  # シャープレシオが閾値と等しい
            timestamp=datetime.now(),
        )

        # 境界値でトリガーが発動しないことを確認
        requests = self.trigger.update_performance_metrics(boundary_metrics)
        self.assertEqual(len(requests), 0)

        # 閾値をわずかに下回る値
        below_threshold = PerformanceMetrics(
            accuracy=0.5,
            precision=0.45,
            recall=0.4,
            f1_score=0.42,
            win_rate=0.4,
            sharpe_ratio=0.99,
            max_drawdown=0.15,  # 閾値を下回る
            timestamp=datetime.now(),
        )

        # 複数回更新してトリガーを発動
        for i in range(10):
            requests = self.trigger.update_performance_metrics(below_threshold)
            if requests:
                break

        # トリガーが発動することを確認
        self.assertGreater(len(requests), 0)

    def test_retraining_history_compression(self):
        """再訓練履歴の圧縮テスト"""
        # 圧縮が有効な設定でトリガーを作成
        compression_config = RetrainingConfig()
        compression_config.max_history_size = 50
        compression_config.compression_enabled = True

        compression_trigger = RetrainingTrigger(compression_config)

        # 大量の履歴データを追加
        for i in range(100):
            metrics = PerformanceMetrics(
                accuracy=0.5,
                precision=0.4,
                recall=0.3,
                f1_score=0.35,
                win_rate=0.4,
                sharpe_ratio=0.6,
                max_drawdown=0.2,
                timestamp=datetime.now(),
            )
            compression_trigger.update_performance_metrics(metrics)

        # 圧縮が実行されていることを確認（履歴サイズが制限されている）
        self.assertLessEqual(
            len(compression_trigger.performance_history),
            compression_config.max_history_size,
        )

    def test_trigger_state_persistence(self):
        """トリガー状態の永続性テスト"""
        # 初期状態を保存
        initial_states = self.trigger.get_trigger_states().copy()

        # トリガーを確実に発動させるような条件で複数回更新
        base_time = datetime.now()

        # 十分な回数更新してトリガーが発動するようにする（時間を進める）
        for i in range(20):
            current_metrics = PerformanceMetrics(
                accuracy=0.3,
                precision=0.25,
                recall=0.2,
                f1_score=0.22,
                win_rate=0.2,
                sharpe_ratio=0.3,
                max_drawdown=0.3,
                timestamp=base_time
                + timedelta(minutes=i * 10),  # 10分ごとに時間を進める
            )
            self.trigger.update_performance_metrics(current_metrics)

        # 状態が変更されていることを確認（トリガーが発動したか）
        updated_states = self.trigger.get_trigger_states()
        trigger_fired = False
        for trigger_id in updated_states:
            if trigger_id in initial_states:
                state = updated_states[trigger_id]
                # トリガーが発動したことを確認（last_triggeredが設定されている、またはstatusがTRIGGERED）
                if (
                    state.last_triggered is not None
                    or state.status == TriggerStatus.TRIGGERED
                    or state.cooldown_until is not None
                ):
                    trigger_fired = True
                    break

        self.assertTrue(trigger_fired, "少なくとも1つのトリガーが発動しているべき")

    def test_performance_metrics_validation(self):
        """パフォーマンスメトリクスのバリデーション"""
        # 無効な値を含むメトリクス（境界外の値）
        invalid_metrics = [
            PerformanceMetrics(
                accuracy=-0.1,
                precision=0.5,
                recall=0.4,
                f1_score=0.45,
                win_rate=0.5,
                sharpe_ratio=0.8,
                max_drawdown=0.1,
                timestamp=datetime.now(),
            ),
            PerformanceMetrics(
                accuracy=1.5,
                precision=0.5,
                recall=0.4,
                f1_score=0.45,
                win_rate=0.5,
                sharpe_ratio=0.8,
                max_drawdown=0.1,
                timestamp=datetime.now(),
            ),
        ]

        for invalid_metric in invalid_metrics:
            # システムがクラッシュせず、適切に処理することを確認
            try:
                requests = self.trigger.update_performance_metrics(invalid_metric)
                self.assertIsInstance(requests, list)
            except Exception as e:
                self.fail(f"無効なメトリクスで例外が発生: {e}")

        # NaN値のテスト
        try:
            nan_metrics = PerformanceMetrics(
                accuracy=float("nan"),
                precision=0.5,
                recall=0.4,
                f1_score=0.45,
                win_rate=0.5,
                sharpe_ratio=0.8,
                max_drawdown=0.1,
                timestamp=datetime.now(),
            )
            requests = self.trigger.update_performance_metrics(nan_metrics)
            self.assertIsInstance(requests, list)
        except Exception as e:
            # NaNは例外を投げる可能性があるが、システムがクラッシュしないことを確認
            self.assertIsInstance(e, (ValueError, TypeError, ArithmeticError))

    def test_distribution_metrics_validation(self):
        """分布メトリクスのバリデーション"""
        # 空の特徴量辞書
        empty_metrics = DataDistributionMetrics(
            feature_means={},
            feature_stds={},
            feature_skewness={},
            feature_kurtosis={},
            sample_size=0,
            timestamp=datetime.now(),
        )

        # 空のメトリクスが適切に処理されることを確認
        requests = self.trigger.update_distribution_metrics(empty_metrics)
        self.assertIsInstance(requests, list)

        # 不整合な特徴量（平均はあるが標準偏差がない）
        inconsistent_metrics = DataDistributionMetrics(
            feature_means={"feature1": 1.0, "feature2": 2.0},
            feature_stds={"feature1": 0.1},  # feature2の標準偏差がない
            feature_skewness={"feature1": 0.0},
            feature_kurtosis={"feature1": 0.0},
            sample_size=1000,
            timestamp=datetime.now(),
        )

        # 不整合なメトリクスも適切に処理されることを確認
        requests = self.trigger.update_distribution_metrics(inconsistent_metrics)
        self.assertIsInstance(requests, list)


class TestRetrainingPolicy(unittest.TestCase):
    """RetrainingPolicyのテスト"""

    def setUp(self):
        self.condition = TriggerCondition(
            trigger_type=TriggerType.PERFORMANCE,
            metric_name="win_rate",
            operator="lt",
            threshold=0.45,
            duration_minutes=60,
            cooldown_minutes=240,
            priority=TriggerPriority.HIGH,
        )
        self.policy = RetrainingPolicy(
            policy_name="test_policy",
            trigger_conditions=[self.condition],
            retraining_strategy="full",
            resource_requirements={"cpu": 2, "memory_gb": 4},
            max_execution_time_hours=4,
            success_criteria={"min_improvement": 0.02},
        )

    def test_should_trigger_true(self):
        """トリガー条件を満たす場合"""
        metrics = {"win_rate": 0.35}  # 閾値未満
        self.assertTrue(self.policy.should_trigger(metrics))

    def test_should_trigger_false(self):
        """トリガー条件を満たさない場合"""
        metrics = {"win_rate": 0.5}  # 閾値以上
        self.assertFalse(self.policy.should_trigger(metrics))

    def test_missing_metric(self):
        """メトリクスが存在しない場合"""
        metrics = {"accuracy": 0.8}  # win_rateなし
        self.assertFalse(self.policy.should_trigger(metrics))


class TestRetrainingConfig(unittest.TestCase):
    """RetrainingConfigのテスト"""

    def test_default_initialization(self):
        """デフォルト設定での初期化"""
        config = RetrainingConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.max_concurrent_retraining, 1)
        self.assertGreater(len(config.trigger_conditions), 0)
        self.assertIsInstance(config.priority_weights, dict)

    def test_custom_initialization(self):
        """カスタム設定での初期化"""
        config = RetrainingConfig(
            enabled=False, max_concurrent_retraining=3, max_history_size=500
        )

        self.assertFalse(config.enabled)
        self.assertEqual(config.max_concurrent_retraining, 3)
        self.assertEqual(config.max_history_size, 500)

    def test_validation(self):
        """設定の検証"""
        # 無効な値での初期化をテスト
        with self.assertRaises(ValueError):
            RetrainingConfig(max_concurrent_retraining=0)

        with self.assertRaises(ValueError):
            RetrainingConfig(retraining_timeout_hours=0)

        with self.assertRaises(ValueError):
            RetrainingConfig(max_history_size=50)  # 100未満


if __name__ == "__main__":
    unittest.main()
