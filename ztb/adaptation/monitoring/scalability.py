"""
Scalability and Operations System
スケーラビリティと運用システム
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ScalabilityConfig
from .types import (
    CostOptimization,
    DeploymentPlan,
    DeploymentStatus,
    LoadDistribution,
    OperationsStatus,
    ResourceType,
    ResourceUsage,
    ScalabilityMetrics,
    ScalingAction,
    ScalingDecision,
    ScalingStrategy,
)

logger = logging.getLogger(__name__)


class LoadBalancer:
    """負荷分散器"""

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.instances: Dict[str, LoadDistribution] = {}
        self.current_index = 0

    def register_instance(self, instance_id: str, max_load: float = 100.0) -> None:
        """インスタンス登録"""
        self.instances[instance_id] = LoadDistribution(
            instance_id=instance_id,
            current_load=0.0,
            max_load=max_load,
            active_connections=0,
            queue_length=0,
            response_time_ms=0.0,
            timestamp=datetime.now(),
        )
        logger.info(f"Instance {instance_id} registered with load balancer")

    def unregister_instance(self, instance_id: str) -> None:
        """インスタンス登録解除"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info(f"Instance {instance_id} unregistered from load balancer")

    def get_next_instance(self) -> Optional[str]:
        """次のインスタンスを取得（負荷分散アルゴリズムに基づく）"""
        if not self.instances:
            return None

        if self.config.load_distribution_algorithm == "round_robin":
            return self._round_robin_selection()
        elif self.config.load_distribution_algorithm == "least_connections":
            return self._least_connections_selection()
        elif self.config.load_distribution_algorithm == "weighted":
            return self._weighted_selection()
        else:
            return self._round_robin_selection()

    def _round_robin_selection(self) -> Optional[str]:
        """ラウンドロビン選択"""
        active_instances = [
            iid
            for iid, dist in self.instances.items()
            if dist.current_load < dist.max_load
        ]

        if not active_instances:
            return None

        instance_id = active_instances[self.current_index % len(active_instances)]
        self.current_index += 1
        return instance_id

    def _least_connections_selection(self) -> Optional[str]:
        """最小接続数選択"""
        active_instances = [
            (iid, dist)
            for iid, dist in self.instances.items()
            if dist.current_load < dist.max_load
        ]

        if not active_instances:
            return None

        # 接続数が最も少ないインスタンスを選択
        active_instances.sort(key=lambda x: x[1].active_connections)
        return active_instances[0][0]

    def _weighted_selection(self) -> Optional[str]:
        """重み付け選択（現在はラウンドロビンと同じ）"""
        return self._round_robin_selection()

    def update_instance_load(self, instance_id: str, load_data: Dict[str, Any]) -> None:
        """インスタンス負荷更新"""
        if instance_id not in self.instances:
            return

        dist = self.instances[instance_id]
        dist.current_load = load_data.get("current_load", dist.current_load)
        dist.active_connections = load_data.get(
            "active_connections", dist.active_connections
        )
        dist.queue_length = load_data.get("queue_length", dist.queue_length)
        dist.response_time_ms = load_data.get("response_time_ms", dist.response_time_ms)
        dist.timestamp = datetime.now()

    def get_load_distribution(self) -> List[LoadDistribution]:
        """負荷分布取得"""
        return list(self.instances.values())


class AutoScaler:
    """自動スケーラー"""

    def __init__(self, config: ScalabilityConfig, load_balancer: LoadBalancer):
        self.config = config
        self.load_balancer = load_balancer
        self.scaling_history: List[ScalingAction] = []
        self.last_scaling_time = datetime.min
        self.current_instances = config.min_instances

    def evaluate_scaling(
        self, resource_usage: List[ResourceUsage]
    ) -> Optional[ScalingAction]:
        """スケーリング評価"""
        if not self.config.auto_scaling_enabled:
            return None

        # クールダウン期間チェック
        if (
            datetime.now() - self.last_scaling_time
        ).total_seconds() < self.config.cooldown_period_seconds:
            return None

        # 平均CPU使用率計算
        cpu_usage = [
            r.utilization_percentage
            for r in resource_usage
            if r.resource_type == ResourceType.CPU
        ]

        if not cpu_usage:
            return None

        avg_cpu_usage = np.mean(cpu_usage)

        # スケーリング決定
        decision = ScalingDecision.NO_CHANGE
        target_instances = self.current_instances

        if avg_cpu_usage >= self.config.scale_up_threshold:
            if self.current_instances < self.config.max_instances:
                decision = ScalingDecision.SCALE_UP
                target_instances = min(
                    self.current_instances + 1, self.config.max_instances
                )
        elif avg_cpu_usage <= self.config.scale_down_threshold:
            if self.current_instances > self.config.min_instances:
                decision = ScalingDecision.SCALE_DOWN
                target_instances = max(
                    self.current_instances - 1, self.config.min_instances
                )

        if decision != ScalingDecision.NO_CHANGE:
            action = ScalingAction(
                action_id=f"scale_{int(time.time())}_{decision.value}",
                scaling_decision=decision,
                scaling_strategy=ScalingStrategy.AUTO,
                target_instances=target_instances,
                current_instances=self.current_instances,
                reason=f"Average CPU usage: {avg_cpu_usage:.2%}",
                estimated_cost_impact=self._estimate_cost_impact(
                    decision, target_instances
                ),
                timestamp=datetime.now(),
                executed_by="auto_scaler",
            )

            self.scaling_history.append(action)
            self.last_scaling_time = datetime.now()
            self.current_instances = target_instances

            return action

        return None

    def _estimate_cost_impact(
        self, decision: ScalingDecision, target_instances: int
    ) -> float:
        """コスト影響見積もり"""
        # 簡易的なコスト計算（実際の実装では詳細な料金モデルを使用）
        base_cost_per_instance = 0.1  # 時間あたりのコスト（仮定）
        hours_affected = 1  # 影響を受ける時間数

        if decision == ScalingDecision.SCALE_UP:
            return base_cost_per_instance * hours_affected
        elif decision == ScalingDecision.SCALE_DOWN:
            return -base_cost_per_instance * hours_affected
        else:
            return 0.0

    def get_scaling_history(self, hours: int = 24) -> List[ScalingAction]:
        """スケーリング履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            action for action in self.scaling_history if action.timestamp >= cutoff_time
        ]


class ResourceOptimizer:
    """リソース最適化器"""

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.optimization_history: List[CostOptimization] = []

    def analyze_resource_usage(
        self, resource_usage: List[ResourceUsage]
    ) -> List[CostOptimization]:
        """リソース使用分析"""
        optimizations = []

        # リソースタイプごとに分析
        for resource_type in ResourceType:
            type_usage = [r for r in resource_usage if r.resource_type == resource_type]

            if not type_usage:
                continue

            # 平均使用率計算
            avg_utilization = np.mean([r.utilization_percentage for r in type_usage])

            # 最適化の可能性を評価
            if avg_utilization < 0.5:  # 50%未満の使用率
                optimization = CostOptimization(
                    optimization_id=f"opt_{resource_type.value}_{int(time.time())}",
                    resource_type=resource_type,
                    current_cost=self._estimate_current_cost(type_usage),
                    optimized_cost=self._estimate_optimized_cost(type_usage),
                    savings_percentage=self.config.cost_optimization_target,
                    recommendations=self._generate_recommendations(
                        resource_type, avg_utilization
                    ),
                    implementation_status="pending",
                    timestamp=datetime.now(),
                )
                optimizations.append(optimization)
                self.optimization_history.append(optimization)

        return optimizations

    def _estimate_current_cost(self, usage: List[ResourceUsage]) -> float:
        """現在のコスト見積もり"""
        # 簡易的なコスト計算
        return len(usage) * 0.05  # 仮定のコスト

    def _estimate_optimized_cost(self, usage: List[ResourceUsage]) -> float:
        """最適化後のコスト見積もり"""
        current_cost = self._estimate_current_cost(usage)
        return current_cost * (1 - self.config.cost_optimization_target)

    def _generate_recommendations(
        self, resource_type: ResourceType, utilization: float
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        if resource_type == ResourceType.CPU:
            if utilization < 0.3:
                recommendations.append("Consider reducing CPU allocation by 50%")
                recommendations.append("Evaluate if instance size can be downgraded")
            elif utilization < 0.5:
                recommendations.append("Consider reducing CPU allocation by 25%")
        elif resource_type == ResourceType.MEMORY:
            if utilization < 0.4:
                recommendations.append("Consider reducing memory allocation")
                recommendations.append("Monitor for memory leaks")

        recommendations.append("Schedule regular resource usage reviews")
        return recommendations


class DeploymentManager:
    """デプロイメントマネージャー"""

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.deployment_history: List[DeploymentPlan] = []
        self.active_deployments: Dict[str, DeploymentPlan] = {}

    def create_deployment_plan(
        self, version: str, target_instances: int, rollout_strategy: str = "rolling"
    ) -> str:
        """デプロイメント計画作成"""
        plan_id = f"deploy_{int(time.time())}_{version}"

        plan = DeploymentPlan(
            plan_id=plan_id,
            version=version,
            target_instances=target_instances,
            rollout_strategy=rollout_strategy,
            rollback_plan=self._generate_rollback_plan(rollout_strategy),
            estimated_duration_minutes=self._estimate_duration(
                target_instances, rollout_strategy
            ),
            created_at=datetime.now(),
            status=DeploymentStatus.PENDING,
        )

        self.deployment_history.append(plan)
        return plan_id

    def execute_deployment(self, plan_id: str) -> bool:
        """デプロイメント実行"""
        if plan_id not in [p.plan_id for p in self.deployment_history]:
            logger.error(f"Deployment plan {plan_id} not found")
            return False

        plan = next(p for p in self.deployment_history if p.plan_id == plan_id)
        plan.status = DeploymentStatus.IN_PROGRESS
        self.active_deployments[plan_id] = plan

        try:
            logger.info(
                f"Executing deployment {plan_id} with strategy {plan.rollout_strategy}"
            )

            # デプロイメント実行（シミュレーション）
            if plan.rollout_strategy == "rolling":
                success = self._execute_rolling_deployment(plan)
            elif plan.rollout_strategy == "blue_green":
                success = self._execute_blue_green_deployment(plan)
            else:
                success = self._execute_canary_deployment(plan)

            if success:
                plan.status = DeploymentStatus.SUCCESS
                logger.info(f"Deployment {plan_id} completed successfully")
            else:
                plan.status = DeploymentStatus.FAILED
                logger.error(f"Deployment {plan_id} failed")
                if self.config.rollback_on_failure:
                    self._execute_rollback(plan)

            return success

        except Exception as e:
            logger.error(f"Deployment {plan_id} execution failed: {e}")
            plan.status = DeploymentStatus.FAILED
            return False
        finally:
            if plan_id in self.active_deployments:
                del self.active_deployments[plan_id]

    def _generate_rollback_plan(self, strategy: str) -> List[str]:
        """ロールバック計画生成"""
        if strategy == "rolling":
            return [
                "Stop new version rollout",
                "Gradually rollback to previous version",
            ]
        elif strategy == "blue_green":
            return [
                "Switch traffic back to blue environment",
                "Terminate green environment",
            ]
        else:  # canary
            return ["Stop canary rollout", "Redirect all traffic to stable version"]

    def _estimate_duration(self, target_instances: int, strategy: str) -> int:
        """所要時間見積もり"""
        base_time = 10  # 分
        if strategy == "rolling":
            return base_time * target_instances
        elif strategy == "blue_green":
            return base_time * 2
        else:  # canary
            return base_time

    def _execute_rolling_deployment(self, plan: DeploymentPlan) -> bool:
        """ローリングデプロイメント実行"""
        # シミュレーション
        time.sleep(2)
        return True

    def _execute_blue_green_deployment(self, plan: DeploymentPlan) -> bool:
        """ブルーグリーンデプロイメント実行"""
        # シミュレーション
        time.sleep(3)
        return True

    def _execute_canary_deployment(self, plan: DeploymentPlan) -> bool:
        """カナリーデプロイメント実行"""
        # シミュレーション
        time.sleep(1)
        return True

    def _execute_rollback(self, plan: DeploymentPlan) -> None:
        """ロールバック実行"""
        logger.warning(f"Executing rollback for deployment {plan.plan_id}")
        plan.status = DeploymentStatus.ROLLED_BACK
        # ロールバック実行（シミュレーション）
        time.sleep(1)


class OperationsManager:
    """運用マネージャー"""

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.backup_history: List[datetime] = []
        self.maintenance_schedule: List[datetime] = []

    def perform_backup(self) -> bool:
        """バックアップ実行"""
        if not self.config.backup_enabled:
            return True

        try:
            logger.info("Starting system backup...")
            # バックアップ実行（シミュレーション）
            time.sleep(5)  # バックアップ時間

            self.backup_history.append(datetime.now())
            logger.info("System backup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def schedule_maintenance(self, start_time: datetime, duration_hours: int) -> str:
        """メンテナンススケジューリング"""
        maintenance_id = f"maint_{int(time.time())}"

        end_time = start_time + timedelta(hours=duration_hours)
        self.maintenance_schedule.append(end_time)

        logger.info(f"Maintenance scheduled: {start_time} to {end_time}")
        return maintenance_id

    def get_operations_status(self) -> OperationsStatus:
        """運用ステータス取得"""
        return OperationsStatus(
            system_status="operational",
            last_backup=max(self.backup_history)
            if self.backup_history
            else datetime.min,
            next_maintenance=min(self.maintenance_schedule)
            if self.maintenance_schedule
            else datetime.max,
            active_alerts=0,  # 実際の実装では監視システムから取得
            pending_updates=0,  # 実際の実装では更新管理システムから取得
            resource_utilization={"cpu": 0.45, "memory": 0.60, "disk": 0.30},
            performance_score=0.85,
            timestamp=datetime.now(),
        )

    def cleanup_old_data(self) -> None:
        """古いデータクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(
            days=self.config.monitoring_retention_days
        )

        # バックアップ履歴クリーンアップ
        self.backup_history = [b for b in self.backup_history if b >= cutoff_date]

        logger.info("Old data cleanup completed")


class ScalabilityManager:
    """スケーラビリティマネージャー"""

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.auto_scaler = AutoScaler(config, self.load_balancer)
        self.resource_optimizer = ResourceOptimizer(config)
        self.deployment_manager = DeploymentManager(config)
        self.operations_manager = OperationsManager(config)

        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False

        # 初期インスタンス登録
        for i in range(config.min_instances):
            self.load_balancer.register_instance(f"instance_{i+1}")

    def start_scalability_monitoring(self) -> None:
        """スケーラビリティ監視開始"""
        if self.is_monitoring:
            logger.warning("Scalability monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._scalability_monitoring_worker, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Scalability monitoring started")

    def stop_scalability_monitoring(self) -> None:
        """スケーラビリティ監視停止"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Scalability monitoring stopped")

    def _scalability_monitoring_worker(self) -> None:
        """スケーラビリティ監視ワーカー"""
        while self.is_monitoring:
            try:
                # リソース使用状況取得（実際の実装では監視システムから）
                resource_usage = self._get_resource_usage()

                # 自動スケーリング評価
                scaling_action = self.auto_scaler.evaluate_scaling(resource_usage)
                if scaling_action:
                    self._execute_scaling_action(scaling_action)

                # リソース最適化分析
                if self.config.resource_optimization_enabled:
                    optimizations = self.resource_optimizer.analyze_resource_usage(
                        resource_usage
                    )
                    self._implement_optimizations(optimizations)

                # 定期的な運用タスク
                self._perform_operations_tasks()

                time.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"Scalability monitoring error: {e}")
                time.sleep(5)

    def _get_resource_usage(self) -> List[ResourceUsage]:
        """リソース使用状況取得"""
        # モックデータ（実際の実装ではシステムメトリクスから取得）
        return [
            ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=45.0 + np.random.normal(0, 10),
                max_capacity=100.0,
                utilization_percentage=0.45 + np.random.normal(0, 0.1),
                timestamp=datetime.now(),
                instance_id="instance_1",
            ),
            ResourceUsage(
                resource_type=ResourceType.MEMORY,
                current_usage=60.0 + np.random.normal(0, 5),
                max_capacity=100.0,
                utilization_percentage=0.60 + np.random.normal(0, 0.05),
                timestamp=datetime.now(),
                instance_id="instance_1",
            ),
        ]

    def _execute_scaling_action(self, action: ScalingAction) -> None:
        """スケーリングアクション実行"""
        logger.info(
            f"Executing scaling action: {action.scaling_decision.value} to {action.target_instances} instances"
        )

        if action.scaling_decision == ScalingDecision.SCALE_UP:
            # 新しいインスタンス追加
            for i in range(action.current_instances, action.target_instances):
                instance_id = f"instance_{i+1}"
                self.load_balancer.register_instance(instance_id)
        elif action.scaling_decision == ScalingDecision.SCALE_DOWN:
            # インスタンス削除
            for i in range(action.target_instances, action.current_instances):
                instance_id = f"instance_{i+1}"
                self.load_balancer.unregister_instance(instance_id)

    def _implement_optimizations(self, optimizations: List[CostOptimization]) -> None:
        """最適化実装"""
        for optimization in optimizations:
            if optimization.implementation_status == "pending":
                logger.info(
                    f"Implementing optimization: {optimization.optimization_id}"
                )
                # 実際の実装では最適化を実行
                optimization.implementation_status = "implemented"

    def _perform_operations_tasks(self) -> None:
        """運用タスク実行"""
        # バックアップチェック
        if self.config.backup_enabled:
            last_backup = self.operations_manager.get_operations_status().last_backup
            hours_since_backup = (datetime.now() - last_backup).total_seconds() / 3600

            if hours_since_backup >= self.config.backup_interval_hours:
                self.operations_manager.perform_backup()

        # データクリーンアップ
        self.operations_manager.cleanup_old_data()

    def get_scalability_metrics(self) -> ScalabilityMetrics:
        """スケーラビリティメトリクス取得"""
        load_distribution = self.load_balancer.get_load_distribution()
        scaling_history = self.auto_scaler.get_scaling_history(hours=24)

        return ScalabilityMetrics(
            total_instances=len(load_distribution),
            active_instances=len(
                [d for d in load_distribution if d.current_load < d.max_load]
            ),
            average_load=np.mean([d.current_load for d in load_distribution])
            if load_distribution
            else 0.0,
            peak_load=max([d.current_load for d in load_distribution])
            if load_distribution
            else 0.0,
            scaling_events_count=len(scaling_history),
            average_response_time_ms=np.mean(
                [d.response_time_ms for d in load_distribution]
            )
            if load_distribution
            else 0.0,
            cost_per_hour=self._estimate_hourly_cost(),
            uptime_percentage=99.9,  # 仮定値
            timestamp=datetime.now(),
        )

    def _estimate_hourly_cost(self) -> float:
        """時間あたりのコスト見積もり"""
        instance_count = len(self.load_balancer.instances)
        cost_per_instance = 0.1  # 仮定値
        return instance_count * cost_per_instance

    def create_deployment(self, version: str, target_instances: int) -> str:
        """デプロイメント作成"""
        return self.deployment_manager.create_deployment_plan(version, target_instances)

    def execute_deployment(self, plan_id: str) -> bool:
        """デプロイメント実行"""
        return self.deployment_manager.execute_deployment(plan_id)

    def get_operations_status(self) -> OperationsStatus:
        """運用ステータス取得"""
        return self.operations_manager.get_operations_status()
