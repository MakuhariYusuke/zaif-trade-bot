"""
V433 Phase 5: Parallel Running Layer - Traffic Distributor

取引シグナルの割合ベース分散と動的調整を行う。
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional


# Mock classes for testing
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.MARKET
    timestamp: Optional[datetime] = None


class DistributionMode(Enum):
    """分散モード"""

    FIXED = "fixed"  # 固定割合
    DYNAMIC = "dynamic"  # 動的調整
    PERFORMANCE_BASED = "performance_based"  # パフォーマンスベース
    GRADUAL = "gradual"  # 段階的移行


class DistributionRule(Enum):
    """分散ルール"""

    ROUND_ROBIN = "round_robin"  # ラウンドロビン
    WEIGHTED_RANDOM = "weighted_random"  # 加重ランダム
    PERFORMANCE_WEIGHTED = "performance_weighted"  # パフォーマンス加重
    VOLUME_BASED = "volume_based"  # 出来高ベース


@dataclass
class SystemEndpoint:
    """システムエンドポイント"""

    system_id: str
    name: str
    capacity: int  # 1秒あたりの最大注文数
    current_load: int = 0
    is_active: bool = True
    last_health_check: datetime = field(default_factory=datetime.now)
    performance_score: float = 1.0  # パフォーマンススコア（0-1）


@dataclass
class DistributionConfig:
    """分散設定"""

    mode: DistributionMode = DistributionMode.FIXED
    rule: DistributionRule = DistributionRule.WEIGHTED_RANDOM
    total_weight: int = 100  # 総重量
    rebalance_interval_seconds: int = 300  # リバランス間隔
    max_single_system_weight: int = 80  # 単一システム最大重量
    min_single_system_weight: int = 5  # 単一システム最小重量
    emergency_switch_threshold: float = 0.8  # 緊急切り替え閾値


@dataclass
class TrafficAllocation:
    """トラフィック配分"""

    system_id: str
    weight: int  # 配分重量
    percentage: float  # 配分割合（%）
    current_orders: int = 0
    success_rate: float = 1.0
    average_latency_ms: float = 0


@dataclass
class DistributionEvent:
    """分散イベント"""

    event_id: str
    timestamp: datetime
    order_id: str
    assigned_system: str
    distribution_rule: str
    execution_time_ms: int
    success: bool
    reason: Optional[str] = None


class TrafficDistributor:
    """
    トラフィックディストリビューター

    取引シグナルの割合ベース分散と動的調整を行い、
    システム間の負荷分散とフェイルセーフを実現する。
    """

    def __init__(self, config: Optional[DistributionConfig] = None):
        """
        初期化

        Args:
            config: 分散設定
        """
        self.config = config or DistributionConfig()

        # システムエンドポイント
        self.endpoints: Dict[str, SystemEndpoint] = {}
        self.endpoint_weights: Dict[str, int] = {}

        # トラフィック配分
        self.allocations: Dict[str, TrafficAllocation] = {}
        self.total_weight = 0

        # 統計情報
        self.distribution_events: List[DistributionEvent] = []
        self.order_counter = 0

        # リバランス制御
        self.last_rebalance = datetime.now()
        self.rebalance_active = False

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # コールバック
        self.distribution_callbacks: List[
            Callable[[DistributionEvent], Awaitable[None]]
        ] = []
        self.rebalance_callbacks: List[
            Callable[[Dict[str, TrafficAllocation]], Awaitable[None]]
        ] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Traffic Distributor initialized")

    def add_endpoint(self, endpoint: SystemEndpoint) -> None:
        """
        エンドポイント追加

        Args:
            endpoint: システムエンドポイント
        """
        self.endpoints[endpoint.system_id] = endpoint

        # 初期配分（等分）
        if self.endpoints:
            equal_weight = max(1, self.config.total_weight // len(self.endpoints))
            self.endpoint_weights[endpoint.system_id] = equal_weight

        self._update_allocations()
        self.logger.info(f"Endpoint added: {endpoint.name} ({endpoint.system_id})")

    def remove_endpoint(self, system_id: str) -> bool:
        """
        エンドポイント削除

        Args:
            system_id: システムID

        Returns:
            bool: 削除成功フラグ
        """
        if system_id in self.endpoints:
            del self.endpoints[system_id]
            if system_id in self.endpoint_weights:
                del self.endpoint_weights[system_id]

            self._update_allocations()
            self.logger.info(f"Endpoint removed: {system_id}")
            return True

        return False

    def update_endpoint_weight(self, system_id: str, weight: int) -> bool:
        """
        エンドポイント重量更新

        Args:
            system_id: システムID
            weight: 新しい重量

        Returns:
            bool: 更新成功フラグ
        """
        if system_id not in self.endpoints:
            return False

        # 重量制約チェック
        if (
            weight < self.config.min_single_system_weight
            or weight > self.config.max_single_system_weight
        ):
            self.logger.warning(f"Weight {weight} out of range for {system_id}")
            return False

        self.endpoint_weights[system_id] = weight
        self._update_allocations()

        self.logger.info(f"Endpoint weight updated: {system_id} = {weight}")
        return True

    def update_endpoint_performance(
        self, system_id: str, performance_score: float, latency_ms: int
    ) -> None:
        """
        エンドポイントパフォーマンス更新

        Args:
            system_id: システムID
            performance_score: パフォーマンススコア（0-1）
            latency_ms: 平均遅延時間
        """
        if system_id in self.endpoints:
            self.endpoints[system_id].performance_score = max(
                0.0, min(1.0, performance_score)
            )

            # アロケーション更新
            if system_id in self.allocations:
                self.allocations[system_id].average_latency_ms = latency_ms

            # パフォーマンスベースモードの場合、リバランス
            if self.config.mode == DistributionMode.PERFORMANCE_BASED:
                self._trigger_rebalance()

    async def distribute_order(self, order: Order) -> Optional[str]:
        """
        注文分散

        Args:
            order: 注文オブジェクト

        Returns:
            Optional[str]: 割り当てられたシステムID
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:08d}"

        start_time = time.time()

        try:
            # 有効なエンドポイント取得
            active_endpoints = {
                sid: ep for sid, ep in self.endpoints.items() if ep.is_active
            }

            if not active_endpoints:
                self.logger.error("No active endpoints available")
                await self._record_event(
                    order_id, None, "No active endpoints", False, "No active endpoints"
                )
                return None

            # 分散アルゴリズム選択
            assigned_system = await self._select_system(order, active_endpoints)

            if assigned_system:
                # 負荷更新
                if assigned_system in self.endpoints:
                    self.endpoints[assigned_system].current_load += 1

                # アロケーション統計更新
                if assigned_system in self.allocations:
                    self.allocations[assigned_system].current_orders += 1

                execution_time = int((time.time() - start_time) * 1000)
                await self._record_event(
                    order_id,
                    assigned_system,
                    self.config.rule.value,
                    True,
                    None,
                    execution_time,
                )

                return assigned_system
            else:
                execution_time = int((time.time() - start_time) * 1000)
                await self._record_event(
                    order_id,
                    None,
                    self.config.rule.value,
                    False,
                    "Distribution failed",
                    execution_time,
                )
                return None

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            await self._record_event(
                order_id, None, self.config.rule.value, False, str(e), execution_time
            )
            self.logger.error(f"Order distribution failed: {e}")
            return None

    async def _select_system(
        self, order: Order, active_endpoints: Dict[str, SystemEndpoint]
    ) -> Optional[str]:
        """
        システム選択

        Args:
            order: 注文オブジェクト
            active_endpoints: 有効なエンドポイント

        Returns:
            Optional[str]: 選択されたシステムID
        """
        if self.config.rule == DistributionRule.ROUND_ROBIN:
            return self._round_robin_selection(active_endpoints)

        elif self.config.rule == DistributionRule.WEIGHTED_RANDOM:
            return self._weighted_random_selection(active_endpoints)

        elif self.config.rule == DistributionRule.PERFORMANCE_WEIGHTED:
            return self._performance_weighted_selection(active_endpoints)

        elif self.config.rule == DistributionRule.VOLUME_BASED:
            return self._volume_based_selection(order, active_endpoints)

        else:
            # デフォルトは加重ランダム
            return self._weighted_random_selection(active_endpoints)

    def _round_robin_selection(
        self, active_endpoints: Dict[str, SystemEndpoint]
    ) -> Optional[str]:
        """
        ラウンドロビン選択

        Args:
            active_endpoints: 有効なエンドポイント

        Returns:
            Optional[str]: 選択されたシステムID
        """
        endpoint_ids = list(active_endpoints.keys())
        if not endpoint_ids:
            return None

        # シンプルなラウンドロビン（実際にはもっと洗練された実装が必要）
        selected_index = self.order_counter % len(endpoint_ids)
        return endpoint_ids[selected_index]

    def _weighted_random_selection(
        self, active_endpoints: Dict[str, SystemEndpoint]
    ) -> Optional[str]:
        """
        加重ランダム選択

        Args:
            active_endpoints: 有効なエンドポイント

        Returns:
            Optional[str]: 選択されたシステムID
        """
        # 重量に基づく選択
        total_weight = sum(
            self.endpoint_weights.get(sid, 1) for sid in active_endpoints.keys()
        )
        if total_weight == 0:
            return None

        pick = random.uniform(0, total_weight)
        current_weight = 0

        for system_id in active_endpoints.keys():
            weight = self.endpoint_weights.get(system_id, 1)
            current_weight += weight
            if pick <= current_weight:
                return system_id

        return None

    def _performance_weighted_selection(
        self, active_endpoints: Dict[str, SystemEndpoint]
    ) -> Optional[str]:
        """
        パフォーマンス加重選択

        Args:
            active_endpoints: 有効なエンドポイント

        Returns:
            Optional[str]: 選択されたシステムID
        """
        # パフォーマンススコアに基づく選択
        total_score = sum(ep.performance_score for ep in active_endpoints.values())
        if total_score == 0:
            return None

        pick = random.uniform(0, total_score)
        current_score = 0

        for system_id, endpoint in active_endpoints.items():
            current_score += endpoint.performance_score
            if pick <= current_score:
                return system_id

        return None

    def _volume_based_selection(
        self, order: Order, active_endpoints: Dict[str, SystemEndpoint]
    ) -> Optional[str]:
        """
        出来高ベース選択

        Args:
            order: 注文オブジェクト
            active_endpoints: 有効なエンドポイント

        Returns:
            Optional[str]: 選択されたシステムID
        """
        # 注文サイズに基づく選択（大口注文は特定システムに）
        order_value = order.quantity * order.price

        # 大口注文の場合、容量の大きいシステムを選択
        if order_value > Decimal("10000"):  # 閾値は設定可能にすべき
            candidates = sorted(
                active_endpoints.items(),
                key=lambda x: (x[1].capacity - x[1].current_load)
                * x[1].performance_score,
                reverse=True,
            )
            return candidates[0][0] if candidates else None
        else:
            # 小口注文は通常の加重ランダム
            return self._weighted_random_selection(active_endpoints)

    def _update_allocations(self) -> None:
        """アロケーション更新"""
        self.total_weight = sum(self.endpoint_weights.values())

        for system_id, endpoint in self.endpoints.items():
            weight = self.endpoint_weights.get(system_id, 0)
            percentage = (
                (weight / self.total_weight * 100) if self.total_weight > 0 else 0
            )

            if system_id not in self.allocations:
                self.allocations[system_id] = TrafficAllocation(
                    system_id=system_id, weight=weight, percentage=percentage
                )
            else:
                self.allocations[system_id].weight = weight
                self.allocations[system_id].percentage = percentage

    async def _record_event(
        self,
        order_id: str,
        assigned_system: Optional[str],
        rule: str,
        success: bool,
        reason: Optional[str] = None,
        execution_time_ms: int = 0,
    ) -> None:
        """
        イベント記録

        Args:
            order_id: 注文ID
            assigned_system: 割り当てシステム
            rule: 分散ルール
            success: 成功フラグ
            reason: 理由
            execution_time_ms: 実行時間
        """
        event = DistributionEvent(
            event_id=f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            order_id=order_id,
            assigned_system=assigned_system or "",
            distribution_rule=rule,
            execution_time_ms=execution_time_ms,
            success=success,
            reason=reason,
        )

        self.distribution_events.append(event)

        # 履歴サイズ制限（最新1000件）
        if len(self.distribution_events) > 1000:
            self.distribution_events = self.distribution_events[-1000:]

        # コールバック実行
        for callback in self.distribution_callbacks:
            try:
                await callback(event)
            except Exception as e:
                self.logger.error(f"Distribution callback error: {e}")

    def _trigger_rebalance(self) -> None:
        """リバランストリガー"""
        now = datetime.now()
        if (
            now - self.last_rebalance
        ).total_seconds() >= self.config.rebalance_interval_seconds:
            asyncio.create_task(self._perform_rebalance())

    async def _perform_rebalance(self) -> None:
        """リバランス実行"""
        if self.rebalance_active:
            return

        self.rebalance_active = True

        try:
            self.logger.info("Starting traffic rebalance")

            # パフォーマンスベースのリバランス
            if self.config.mode == DistributionMode.PERFORMANCE_BASED:
                await self._performance_based_rebalance()
            elif self.config.mode == DistributionMode.GRADUAL:
                await self._gradual_rebalance()

            self.last_rebalance = datetime.now()

            # コールバック実行
            for callback in self.rebalance_callbacks:
                try:
                    await callback(self.allocations.copy())
                except Exception as e:
                    self.logger.error(f"Rebalance callback error: {e}")

            self.logger.info("Traffic rebalance completed")

        except Exception as e:
            self.logger.error(f"Rebalance failed: {e}")
        finally:
            self.rebalance_active = False

    async def _performance_based_rebalance(self) -> None:
        """パフォーマンスベースリバランス"""
        # パフォーマンススコアに基づいて重量を調整
        total_score = sum(
            ep.performance_score for ep in self.endpoints.values() if ep.is_active
        )

        if total_score == 0:
            return

        for system_id, endpoint in self.endpoints.items():
            if not endpoint.is_active:
                continue

            # パフォーマンス比率に基づく新しい重量
            performance_ratio = endpoint.performance_score / total_score
            new_weight = int(performance_ratio * self.config.total_weight)

            # 制約適用
            new_weight = max(
                self.config.min_single_system_weight,
                min(self.config.max_single_system_weight, new_weight),
            )

            self.endpoint_weights[system_id] = new_weight

        self._update_allocations()

    async def _gradual_rebalance(self) -> None:
        """段階的リバランス"""
        # 段階的に重量を調整（例: 既存システムから新システムへ移行）
        # 実装はユースケースによる
        pass

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Traffic monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Traffic monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # 定期リバランスチェック
                if self.config.mode in [
                    DistributionMode.DYNAMIC,
                    DistributionMode.PERFORMANCE_BASED,
                ]:
                    if (
                        datetime.now() - self.last_rebalance
                    ).total_seconds() >= self.config.rebalance_interval_seconds:
                        asyncio.run(self._perform_rebalance())

                # 負荷リセット（1秒ごと）
                for endpoint in self.endpoints.values():
                    endpoint.current_load = max(
                        0, endpoint.current_load - endpoint.capacity // 60
                    )  # 1分平均

                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)

    def get_allocations(self) -> Dict[str, TrafficAllocation]:
        """
        アロケーション取得

        Returns:
            Dict[str, TrafficAllocation]: トラフィック配分
        """
        return self.allocations.copy()

    def get_distribution_stats(self) -> Dict[str, Any]:
        """
        分散統計取得

        Returns:
            Dict[str, Any]: 分散統計
        """
        total_events = len(self.distribution_events)
        if total_events == 0:
            return {}

        recent_events = self.distribution_events[-100:]  # 最新100件

        success_rate = sum(1 for e in recent_events if e.success) / len(recent_events)
        avg_latency = sum(e.execution_time_ms for e in recent_events) / len(
            recent_events
        )

        system_stats = {}
        for system_id in self.endpoints.keys():
            system_events = [e for e in recent_events if e.assigned_system == system_id]
            if system_events:
                system_stats[system_id] = {
                    "orders": len(system_events),
                    "success_rate": sum(1 for e in system_events if e.success)
                    / len(system_events),
                    "avg_latency": sum(e.execution_time_ms for e in system_events)
                    / len(system_events),
                }

        return {
            "total_orders": self.order_counter,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "system_stats": system_stats,
            "active_endpoints": len(
                [ep for ep in self.endpoints.values() if ep.is_active]
            ),
        }

    def add_distribution_callback(
        self, callback: Callable[[DistributionEvent], Awaitable[None]]
    ) -> None:
        """
        分散コールバック追加

        Args:
            callback: コールバック関数
        """
        self.distribution_callbacks.append(callback)

    def add_rebalance_callback(
        self, callback: Callable[[Dict[str, TrafficAllocation]], Awaitable[None]]
    ) -> None:
        """
        リバランスコールバック追加

        Args:
            callback: コールバック関数
        """
        self.rebalance_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "config": {
                "mode": self.config.mode.value,
                "rule": self.config.rule.value,
                "total_weight": self.config.total_weight,
                "rebalance_interval_seconds": self.config.rebalance_interval_seconds,
                "max_single_system_weight": self.config.max_single_system_weight,
                "min_single_system_weight": self.config.min_single_system_weight,
                "emergency_switch_threshold": self.config.emergency_switch_threshold,
            },
            "endpoints": [
                {
                    "system_id": ep.system_id,
                    "name": ep.name,
                    "capacity": ep.capacity,
                    "current_load": ep.current_load,
                    "is_active": ep.is_active,
                    "last_health_check": ep.last_health_check.isoformat(),
                    "performance_score": ep.performance_score,
                }
                for ep in self.endpoints.values()
            ],
            "endpoint_weights": self.endpoint_weights,
            "last_rebalance": self.last_rebalance.isoformat(),
            "order_counter": self.order_counter,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Distributor state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 設定復元
            config_data = state["config"]
            self.config = DistributionConfig(
                mode=DistributionMode(config_data["mode"]),
                rule=DistributionRule(config_data["rule"]),
                total_weight=config_data["total_weight"],
                rebalance_interval_seconds=config_data["rebalance_interval_seconds"],
                max_single_system_weight=config_data["max_single_system_weight"],
                min_single_system_weight=config_data["min_single_system_weight"],
                emergency_switch_threshold=config_data["emergency_switch_threshold"],
            )

            # エンドポイント復元
            self.endpoints = {}
            for ep_data in state.get("endpoints", []):
                endpoint = SystemEndpoint(
                    system_id=ep_data["system_id"],
                    name=ep_data["name"],
                    capacity=ep_data["capacity"],
                    current_load=ep_data.get("current_load", 0),
                    is_active=ep_data.get("is_active", True),
                    performance_score=ep_data.get("performance_score", 1.0),
                )
                endpoint.last_health_check = datetime.fromisoformat(
                    ep_data["last_health_check"]
                )
                self.endpoints[endpoint.system_id] = endpoint

            # 重量復元
            self.endpoint_weights = state.get("endpoint_weights", {})

            # その他
            self.last_rebalance = datetime.fromisoformat(state["last_rebalance"])
            self.order_counter = state.get("order_counter", 0)

            self._update_allocations()

            self.logger.info(f"Distributor state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load distributor state: {e}")
            return False
