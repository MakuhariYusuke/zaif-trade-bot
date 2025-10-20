"""
Scalability and Operations System Tests
スケーラビリティと運用システムのテスト
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from .config import ScalabilityConfig
from .scalability import (
    AutoScaler,
    DeploymentManager,
    LoadBalancer,
    OperationsManager,
    ResourceOptimizer,
    ScalabilityManager,
)
from .types import (
    DeploymentStatus,
    OperationsStatus,
    ResourceType,
    ResourceUsage,
    ScalabilityMetrics,
    ScalingAction,
    ScalingDecision,
    ScalingStrategy,
)


class TestLoadBalancer(unittest.TestCase):
    """負荷分散器テスト"""

    def setUp(self):
        self.config = ScalabilityConfig()
        self.lb = LoadBalancer(self.config)

    def test_register_instance(self):
        """インスタンス登録テスト"""
        self.lb.register_instance("instance_1")
        self.assertIn("instance_1", self.lb.instances)

        dist = self.lb.instances["instance_1"]
        self.assertEqual(dist.instance_id, "instance_1")
        self.assertEqual(dist.current_load, 0.0)
        self.assertEqual(dist.max_load, 100.0)

    def test_unregister_instance(self):
        """インスタンス登録解除テスト"""
        self.lb.register_instance("instance_1")
        self.assertIn("instance_1", self.lb.instances)

        self.lb.unregister_instance("instance_1")
        self.assertNotIn("instance_1", self.lb.instances)

    def test_round_robin_selection(self):
        """ラウンドロビン選択テスト"""
        self.config.load_distribution_algorithm = "round_robin"

        self.lb.register_instance("instance_1")
        self.lb.register_instance("instance_2")
        self.lb.register_instance("instance_3")

        # 最初の選択
        instance = self.lb.get_next_instance()
        self.assertEqual(instance, "instance_1")

        # 2番目の選択
        instance = self.lb.get_next_instance()
        self.assertEqual(instance, "instance_2")

        # 3番目の選択
        instance = self.lb.get_next_instance()
        self.assertEqual(instance, "instance_3")

        # 4番目の選択（循環）
        instance = self.lb.get_next_instance()
        self.assertEqual(instance, "instance_1")

    def test_least_connections_selection(self):
        """最小接続数選択テスト"""
        self.config.load_distribution_algorithm = "least_connections"

        self.lb.register_instance("instance_1")
        self.lb.register_instance("instance_2")

        # 接続数を設定
        self.lb.update_instance_load("instance_1", {"active_connections": 5})
        self.lb.update_instance_load("instance_2", {"active_connections": 3})

        # 最小接続数のインスタンスを選択
        instance = self.lb.get_next_instance()
        self.assertEqual(instance, "instance_2")

    def test_update_instance_load(self):
        """インスタンス負荷更新テスト"""
        self.lb.register_instance("instance_1")

        load_data = {
            "current_load": 75.0,
            "active_connections": 10,
            "queue_length": 2,
            "response_time_ms": 150.0,
        }

        self.lb.update_instance_load("instance_1", load_data)

        dist = self.lb.instances["instance_1"]
        self.assertEqual(dist.current_load, 75.0)
        self.assertEqual(dist.active_connections, 10)
        self.assertEqual(dist.queue_length, 2)
        self.assertEqual(dist.response_time_ms, 150.0)

    def test_get_load_distribution(self):
        """負荷分布取得テスト"""
        self.lb.register_instance("instance_1")
        self.lb.register_instance("instance_2")

        distribution = self.lb.get_load_distribution()
        self.assertEqual(len(distribution), 2)

        instance_ids = [d.instance_id for d in distribution]
        self.assertIn("instance_1", instance_ids)
        self.assertIn("instance_2", instance_ids)


class TestAutoScaler(unittest.TestCase):
    """自動スケーラーテスト"""

    def setUp(self):
        self.config = ScalabilityConfig(
            min_instances=1,
            max_instances=5,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            cooldown_period_seconds=60,
        )
        self.lb = LoadBalancer(self.config)
        self.scaler = AutoScaler(self.config, self.lb)

    def test_scale_up_decision(self):
        """スケールアップ決定テスト"""
        # 高負荷のCPU使用状況
        resource_usage = [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=85.0,
                max_capacity=100.0,
                utilization_percentage=0.85,
                timestamp=datetime.now(),
                instance_id="instance_1",
            )
        ]

        action = self.scaler.evaluate_scaling(resource_usage)
        self.assertIsNotNone(action)
        self.assertEqual(action.scaling_decision, ScalingDecision.SCALE_UP)
        self.assertEqual(action.target_instances, 2)

    def test_scale_down_decision(self):
        """スケールダウン決定テスト"""
        self.scaler.current_instances = 3

        # 低負荷のCPU使用状況
        resource_usage = [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=15.0,
                max_capacity=100.0,
                utilization_percentage=0.15,
                timestamp=datetime.now(),
                instance_id="instance_1",
            )
        ]

        action = self.scaler.evaluate_scaling(resource_usage)
        self.assertIsNotNone(action)
        self.assertEqual(action.scaling_decision, ScalingDecision.SCALE_DOWN)
        self.assertEqual(action.target_instances, 2)

    def test_no_scaling_decision(self):
        """スケーリングなし決定テスト"""
        # 中間負荷のCPU使用状況
        resource_usage = [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=50.0,
                max_capacity=100.0,
                utilization_percentage=0.50,
                timestamp=datetime.now(),
                instance_id="instance_1",
            )
        ]

        action = self.scaler.evaluate_scaling(resource_usage)
        self.assertIsNone(action)

    def test_cooldown_period(self):
        """クールダウン期間テスト"""
        # 最初のスケーリング
        resource_usage = [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=85.0,
                max_capacity=100.0,
                utilization_percentage=0.85,
                timestamp=datetime.now(),
                instance_id="instance_1",
            )
        ]

        action1 = self.scaler.evaluate_scaling(resource_usage)
        self.assertIsNotNone(action1)

        # すぐに2回目のスケーリングを試みる（クールダウン期間内）
        action2 = self.scaler.evaluate_scaling(resource_usage)
        self.assertIsNone(action2)

    def test_get_scaling_history(self):
        """スケーリング履歴取得テスト"""
        # スケーリングアクションを追加
        action = ScalingAction(
            action_id="test_action",
            scaling_decision=ScalingDecision.SCALE_UP,
            scaling_strategy=ScalingStrategy.AUTO,
            target_instances=2,
            current_instances=1,
            reason="Test scaling",
            estimated_cost_impact=0.1,
            timestamp=datetime.now(),
            executed_by="test",
        )
        self.scaler.scaling_history.append(action)

        history = self.scaler.get_scaling_history(hours=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].action_id, "test_action")


class TestResourceOptimizer(unittest.TestCase):
    """リソース最適化器テスト"""

    def setUp(self):
        self.config = ScalabilityConfig(cost_optimization_target=0.2)
        self.optimizer = ResourceOptimizer(self.config)

    def test_analyze_resource_usage(self):
        """リソース使用分析テスト"""
        resource_usage = [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=30.0,
                max_capacity=100.0,
                utilization_percentage=0.30,
                timestamp=datetime.now(),
                instance_id="instance_1",
            ),
            ResourceUsage(
                resource_type=ResourceType.MEMORY,
                current_usage=20.0,
                max_capacity=100.0,
                utilization_percentage=0.20,
                timestamp=datetime.now(),
                instance_id="instance_1",
            ),
        ]

        optimizations = self.optimizer.analyze_resource_usage(resource_usage)
        self.assertGreater(len(optimizations), 0)

        # CPU最適化を確認
        cpu_opt = next(
            (opt for opt in optimizations if opt.resource_type == ResourceType.CPU),
            None,
        )
        self.assertIsNotNone(cpu_opt)
        self.assertEqual(cpu_opt.savings_percentage, 0.2)
        self.assertIn("CPU allocation", cpu_opt.recommendations[0])

    def test_no_optimization_needed(self):
        """最適化不要テスト"""
        resource_usage = [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=70.0,
                max_capacity=100.0,
                utilization_percentage=0.70,
                timestamp=datetime.now(),
                instance_id="instance_1",
            )
        ]

        optimizations = self.optimizer.analyze_resource_usage(resource_usage)
        self.assertEqual(len(optimizations), 0)


class TestDeploymentManager(unittest.TestCase):
    """デプロイメントマネージャーテスト"""

    def setUp(self):
        self.config = ScalabilityConfig(rollback_on_failure=True)
        self.manager = DeploymentManager(self.config)

    def test_create_deployment_plan(self):
        """デプロイメント計画作成テスト"""
        plan_id = self.manager.create_deployment_plan("v1.2.0", 3, "rolling")

        self.assertIn("deploy_", plan_id)
        self.assertIn("v1.2.0", plan_id)

        # 計画が履歴に追加されていることを確認
        plan = next(
            (p for p in self.manager.deployment_history if p.plan_id == plan_id), None
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.version, "v1.2.0")
        self.assertEqual(plan.target_instances, 3)
        self.assertEqual(plan.rollout_strategy, "rolling")
        self.assertEqual(plan.status, DeploymentStatus.PENDING)

    def test_execute_deployment_success(self):
        """デプロイメント実行成功テスト"""
        plan_id = self.manager.create_deployment_plan("v1.2.0", 2)

        success = self.manager.execute_deployment(plan_id)
        self.assertTrue(success)

        # 計画のステータスを確認
        plan = next(
            (p for p in self.manager.deployment_history if p.plan_id == plan_id), None
        )
        self.assertEqual(plan.status, DeploymentStatus.SUCCESS)

    def test_execute_deployment_failure(self):
        """デプロイメント実行失敗テスト"""
        plan_id = self.manager.create_deployment_plan("v1.2.0", 2)

        # 失敗をシミュレート
        with patch.object(
            self.manager, "_execute_rolling_deployment", return_value=False
        ):
            success = self.manager.execute_deployment(plan_id)
            self.assertFalse(success)

            # 計画のステータスを確認
            plan = next(
                (p for p in self.manager.deployment_history if p.plan_id == plan_id),
                None,
            )
            self.assertEqual(plan.status, DeploymentStatus.FAILED)

    def test_rollback_plan_generation(self):
        """ロールバック計画生成テスト"""
        plan = self.manager.create_deployment_plan("v1.2.0", 2, "blue_green")

        # 計画を取得
        deployment_plan = next(
            (p for p in self.manager.deployment_history if p.plan_id == plan), None
        )
        self.assertIsNotNone(deployment_plan)

        rollback_plan = deployment_plan.rollback_plan
        self.assertIn("Switch traffic back to blue environment", rollback_plan)


class TestOperationsManager(unittest.TestCase):
    """運用マネージャーテスト"""

    def setUp(self):
        self.config = ScalabilityConfig(
            backup_enabled=True, backup_interval_hours=24, monitoring_retention_days=30
        )
        self.manager = OperationsManager(self.config)

    def test_perform_backup(self):
        """バックアップ実行テスト"""
        success = self.manager.perform_backup()
        self.assertTrue(success)

        # バックアップ履歴を確認
        self.assertGreater(len(self.manager.backup_history), 0)

    def test_schedule_maintenance(self):
        """メンテナンススケジューリングテスト"""
        start_time = datetime.now() + timedelta(hours=2)
        maintenance_id = self.manager.schedule_maintenance(start_time, 4)

        self.assertIn("maint_", maintenance_id)
        self.assertGreater(len(self.manager.maintenance_schedule), 0)

    def test_get_operations_status(self):
        """運用ステータス取得テスト"""
        status = self.manager.get_operations_status()

        self.assertIsInstance(status, OperationsStatus)
        self.assertEqual(status.system_status, "operational")
        self.assertIn("cpu", status.resource_utilization)
        self.assertIn("memory", status.resource_utilization)
        self.assertIn("disk", status.resource_utilization)

    def test_cleanup_old_data(self):
        """古いデータクリーンアップテスト"""
        # 古いバックアップを追加
        old_date = datetime.now() - timedelta(days=40)
        self.manager.backup_history.append(old_date)

        # 新しいバックアップを追加
        new_date = datetime.now() - timedelta(days=10)
        self.manager.backup_history.append(new_date)

        self.manager.cleanup_old_data()

        # 古いバックアップが削除されていることを確認
        self.assertEqual(len(self.manager.backup_history), 1)
        self.assertEqual(self.manager.backup_history[0], new_date)


class TestScalabilityManager(unittest.TestCase):
    """スケーラビリティマネージャーテスト"""

    def setUp(self):
        self.config = ScalabilityConfig(
            min_instances=1,
            max_instances=3,
            auto_scaling_enabled=True,
            resource_optimization_enabled=True,
        )
        self.manager = ScalabilityManager(self.config)

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.manager.load_balancer)
        self.assertIsNotNone(self.manager.auto_scaler)
        self.assertIsNotNone(self.manager.resource_optimizer)
        self.assertIsNotNone(self.manager.deployment_manager)
        self.assertIsNotNone(self.manager.operations_manager)

        # 初期インスタンスが登録されていることを確認
        distribution = self.manager.load_balancer.get_load_distribution()
        self.assertEqual(len(distribution), self.config.min_instances)

    def test_start_stop_monitoring(self):
        """監視開始・停止テスト"""
        self.manager.start_scalability_monitoring()
        self.assertTrue(self.manager.is_monitoring)

        self.manager.stop_scalability_monitoring()
        self.assertFalse(self.manager.is_monitoring)

    def test_get_scalability_metrics(self):
        """スケーラビリティメトリクス取得テスト"""
        metrics = self.manager.get_scalability_metrics()

        self.assertIsInstance(metrics, ScalabilityMetrics)
        self.assertEqual(metrics.total_instances, self.config.min_instances)
        self.assertGreaterEqual(metrics.uptime_percentage, 0.0)
        self.assertLessEqual(metrics.uptime_percentage, 100.0)

    def test_create_and_execute_deployment(self):
        """デプロイメント作成・実行テスト"""
        plan_id = self.manager.create_deployment("v1.3.0", 2)
        self.assertIsNotNone(plan_id)

        success = self.manager.execute_deployment(plan_id)
        self.assertTrue(success)

    def test_get_operations_status(self):
        """運用ステータス取得テスト"""
        status = self.manager.get_operations_status()

        self.assertIsInstance(status, OperationsStatus)
        self.assertEqual(status.system_status, "operational")


if __name__ == "__main__":
    unittest.main()
