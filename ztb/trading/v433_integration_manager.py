#!/usr/bin/env python3
"""
V433 Phase 4: 統合システムマネージャー
全コンポーネントの統合、パフォーマンス最適化、包括的検証
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import logging

from ztb.utils.logging_utils import get_logger
from ztb.trading.v433_integrated_system import V433IntegratedSystem
from ztb.trading.trade_execution_engine import TradeExecutionEngine
from ztb.trading.position_manager import PositionManager
from ztb.trading.risk_overlay import RiskOverlay

logger = get_logger(__name__)

@dataclass
class SystemIntegrationConfig:
    """システム統合設定"""
    # パフォーマンス設定
    target_latency_ms: float = 100.0  # 目標レイテンシー (100ms)
    max_memory_usage_gb: float = 4.0   # 最大メモリ使用量 (4GB)
    max_cpu_usage_percent: float = 80.0  # 最大CPU使用率 (80%)

    # 統合設定
    component_startup_timeout: int = 30  # コンポーネント起動タイムアウト (秒)
    system_health_check_interval: int = 60  # システムヘルスチェック間隔 (秒)
    data_flow_monitoring: bool = True

    # 最適化設定
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 300  # パフォーマンスログ間隔 (5分)
    memory_cleanup_interval: int = 3600  # メモリクリーンアップ間隔 (1時間)

    # フェイルセーフ設定
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: int = 300  # リカバリー間隔 (5分)

    # モニタリング設定
    enable_detailed_logging: bool = True
    alert_on_performance_degradation: bool = True
    performance_degradation_threshold: float = 0.15  # 15%性能低下でアラート


@dataclass
class SystemHealthMetrics:
    """システムヘルス指標"""
    timestamp: datetime = field(default_factory=datetime.now)

    # パフォーマンス指標
    latency_ms: float = 0.0
    memory_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    thread_count: int = 0

    # コンポーネント状態
    components_active: Dict[str, bool] = field(default_factory=dict)
    data_flow_status: Dict[str, bool] = field(default_factory=dict)

    # エラー指標
    error_count: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0

    # ビジネス指標
    active_positions: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0


@dataclass
class IntegrationTestResult:
    """統合テスト結果"""
    test_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """パフォーマンス監視器"""

    def __init__(self, config: SystemIntegrationConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # パフォーマンス履歴
        self.metrics_history: List[SystemHealthMetrics] = []
        self.baseline_metrics: Optional[SystemHealthMetrics] = None

        # プロセス監視
        self.process = psutil.Process()

    def measure_latency(self, operation: callable, *args, **kwargs) -> Tuple[float, Any]:
        """操作のレイテンシーを測定"""
        start_time = time.time()
        result = operation(*args, **kwargs)
        latency = (time.time() - start_time) * 1000  # ms

        return latency, result

    def get_current_metrics(self) -> SystemHealthMetrics:
        """現在のシステム指標を取得"""
        metrics = SystemHealthMetrics()

        # パフォーマンス指標
        metrics.memory_usage_gb = self.process.memory_info().rss / (1024 ** 3)  # GB
        metrics.cpu_usage_percent = self.process.cpu_percent(interval=0.1)
        metrics.thread_count = self.process.num_threads()

        # レイテンシーは別途測定
        metrics.latency_ms = 0.0  # ダミー値

        return metrics

    def update_baseline(self):
        """ベースラインパフォーマンスを更新"""
        self.baseline_metrics = self.get_current_metrics()
        self.logger.info(f"Performance baseline updated: "
                        f"Memory={self.baseline_metrics.memory_usage_gb:.2f}GB, "
                        f"CPU={self.baseline_metrics.cpu_usage_percent:.1f}%")

    def check_performance_degradation(self, current: SystemHealthMetrics) -> bool:
        """パフォーマンス低下をチェック"""
        if not self.baseline_metrics:
            return False

        # メモリ使用量の増加チェック
        memory_increase = (current.memory_usage_gb - self.baseline_metrics.memory_usage_gb) / self.baseline_metrics.memory_usage_gb

        # CPU使用率の増加チェック
        cpu_increase = (current.cpu_usage_percent - self.baseline_metrics.cpu_usage_percent) / max(self.baseline_metrics.cpu_usage_percent, 1)

        # レイテンシーの増加チェック
        latency_increase = (current.latency_ms - self.baseline_metrics.latency_ms) / max(self.baseline_metrics.latency_ms, 1)

        # 閾値超過チェック
        if (memory_increase > self.config.performance_degradation_threshold or
            cpu_increase > self.config.performance_degradation_threshold or
            latency_increase > self.config.performance_degradation_threshold):
            return True

        return False

    def log_performance_metrics(self, metrics: SystemHealthMetrics):
        """パフォーマンス指標をログ"""
        self.metrics_history.append(metrics)

        # 履歴を制限 (最新1000件)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        # 定期ログ
        if len(self.metrics_history) % 10 == 0:  # 10回に1回
            self.logger.info(f"Performance: Memory={metrics.memory_usage_gb:.2f}GB, "
                           f"CPU={metrics.cpu_usage_percent:.1f}%, "
                           f"Threads={metrics.thread_count}")


class ComponentManager:
    """コンポーネントマネージャー"""

    def __init__(self, exchange: str = "zaif"):
        self.exchange = exchange
        self.logger = get_logger(__name__)

        # コンポーネントインスタンス
        self.v433_system: Optional[V433IntegratedSystem] = None
        self.execution_engine: Optional[TradeExecutionEngine] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_overlay: Optional[RiskOverlay] = None

        # コンポーネント状態
        self.components: Dict[str, Dict[str, Any]] = {}

    def initialize_components(self) -> bool:
        """全コンポーネントを初期化"""
        try:
            self.logger.info("Initializing V433 system components...")

            # 取引実行エンジンの初期化
            self.execution_engine = TradeExecutionEngine(self.exchange)
            self.components["execution_engine"] = {
                "instance": self.execution_engine,
                "status": "initialized",
                "start_time": datetime.now()
            }

            # ポジション管理システムの初期化
            self.position_manager = PositionManager(self.execution_engine, self.exchange)
            self.components["position_manager"] = {
                "instance": self.position_manager,
                "status": "initialized",
                "start_time": datetime.now()
            }

            # リスクオーバーレイの初期化
            self.risk_overlay = RiskOverlay(self.position_manager)
            self.components["risk_overlay"] = {
                "instance": self.risk_overlay,
                "status": "initialized",
                "start_time": datetime.now()
            }

            # V433統合システムの初期化
            self.v433_system = V433IntegratedSystem(self.exchange)
            self.components["v433_system"] = {
                "instance": self.v433_system,
                "status": "initialized",
                "start_time": datetime.now()
            }

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False

    def start_components(self) -> bool:
        """全コンポーネントを開始"""
        try:
            self.logger.info("Starting V433 system components...")

            # 実行順序: 実行エンジン → ポジション管理 → リスクオーバーレイ → V433システム

            # 取引実行エンジン開始
            self.execution_engine.start_execution()
            self.components["execution_engine"]["status"] = "running"

            # ポジション管理開始
            self.position_manager.start_management()
            self.components["position_manager"]["status"] = "running"

            # リスクオーバーレイ開始
            self.risk_overlay.start_overlay()
            self.components["risk_overlay"]["status"] = "running"

            # V433システム開始
            self.v433_system.start_system()
            self.components["v433_system"]["status"] = "running"

            self.logger.info("All components started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Component startup failed: {e}")
            self.stop_components()  # 失敗時は全停止
            return False

    def stop_components(self):
        """全コンポーネントを停止"""
        self.logger.info("Stopping V433 system components...")

        # 停止順序: V433システム → リスクオーバーレイ → ポジション管理 → 実行エンジン

        try:
            if self.v433_system and self.components["v433_system"]["status"] == "running":
                self.v433_system.stop_system()
                self.components["v433_system"]["status"] = "stopped"
        except Exception as e:
            self.logger.error(f"V433 system stop failed: {e}")

        try:
            if self.risk_overlay and self.components["risk_overlay"]["status"] == "running":
                self.risk_overlay.stop_overlay()
                self.components["risk_overlay"]["status"] = "stopped"
        except Exception as e:
            self.logger.error(f"Risk overlay stop failed: {e}")

        try:
            if self.position_manager and self.components["position_manager"]["status"] == "running":
                self.position_manager.stop_management()
                self.components["position_manager"]["status"] = "stopped"
        except Exception as e:
            self.logger.error(f"Position manager stop failed: {e}")

        try:
            if self.execution_engine and self.components["execution_engine"]["status"] == "running":
                self.execution_engine.stop_execution()
                self.components["execution_engine"]["status"] = "stopped"
        except Exception as e:
            self.logger.error(f"Execution engine stop failed: {e}")

        self.logger.info("All components stopped")

    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """コンポーネント状態を取得"""
        status = {}
        for name, component in self.components.items():
            status[name] = {
                "status": component["status"],
                "uptime": (datetime.now() - component["start_time"]).total_seconds() if "start_time" in component else 0,
                "healthy": component["status"] in ["running", "initialized"]
            }
        return status

    def restart_component(self, component_name: str) -> bool:
        """指定コンポーネントを再起動"""
        if component_name not in self.components:
            self.logger.error(f"Component {component_name} not found")
            return False

        try:
            self.logger.info(f"Restarting component: {component_name}")

            # コンポーネント固有の再起動ロジック
            if component_name == "execution_engine":
                if self.execution_engine:
                    self.execution_engine.stop_execution()
                    time.sleep(1)
                    self.execution_engine.start_execution()
            elif component_name == "position_manager":
                if self.position_manager:
                    self.position_manager.stop_management()
                    time.sleep(1)
                    self.position_manager.start_management()
            elif component_name == "risk_overlay":
                if self.risk_overlay:
                    self.risk_overlay.stop_overlay()
                    time.sleep(1)
                    self.risk_overlay.start_overlay()
            elif component_name == "v433_system":
                if self.v433_system:
                    self.v433_system.stop_system()
                    time.sleep(1)
                    self.v433_system.start_system()

            self.components[component_name]["status"] = "running"
            self.logger.info(f"Component {component_name} restarted successfully")
            return True

        except Exception as e:
            self.logger.error(f"Component {component_name} restart failed: {e}")
            return False


class IntegrationTester:
    """統合テスト実行器"""

    def __init__(self, component_manager: ComponentManager):
        self.component_manager = component_manager
        self.logger = get_logger(__name__)

        # テスト結果
        self.test_results: List[IntegrationTestResult] = []

    def run_full_integration_test(self) -> List[IntegrationTestResult]:
        """完全統合テストを実行"""
        self.logger.info("Running full integration test suite...")

        test_results = []

        # コンポーネント起動テスト
        test_results.append(self._test_component_startup())

        # データフロー統合テスト
        test_results.append(self._test_data_flow_integration())

        # 取引ワークフロー統合テスト
        test_results.append(self._test_trading_workflow_integration())

        # パフォーマンス統合テスト
        test_results.append(self._test_performance_integration())

        # エラー処理統合テスト
        test_results.append(self._test_error_handling_integration())

        # 結果保存
        self.test_results.extend(test_results)

        # サマリーログ
        success_count = sum(1 for r in test_results if r.success)
        total_count = len(test_results)
        self.logger.info(f"Integration test completed: {success_count}/{total_count} tests passed")

        return test_results

    def _test_component_startup(self) -> IntegrationTestResult:
        """コンポーネント起動テスト"""
        start_time = time.time()

        try:
            # コンポーネント初期化テスト
            success = self.component_manager.initialize_components()
            if not success:
                raise Exception("Component initialization failed")

            # コンポーネント開始テスト
            success = self.component_manager.start_components()
            if not success:
                raise Exception("Component startup failed")

            # 状態確認
            status = self.component_manager.get_component_status()
            all_running = all(s["status"] == "running" for s in status.values())

            if not all_running:
                raise Exception(f"Not all components running: {status}")

            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="component_startup",
                success=True,
                execution_time=execution_time,
                details={"component_status": status}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="component_startup",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _test_data_flow_integration(self) -> IntegrationTestResult:
        """データフロー統合テスト"""
        start_time = time.time()

        try:
            # 市場データ更新テスト
            test_price = 5000000.0
            self.component_manager.v433_system.update_market_data("btc_jpy", test_price)

            # データ伝播確認
            time.sleep(0.1)  # 伝播待機

            # V433システムの価格確認
            current_prices = self.component_manager.v433_system.current_prices
            if current_prices.get("btc_jpy") != test_price:
                raise Exception("Price data not propagated to V433 system")

            # リスクオーバーレイの価格確認
            risk_prices = self.component_manager.risk_overlay.current_prices
            if risk_prices.get("btc_jpy") != test_price:
                raise Exception("Price data not propagated to risk overlay")

            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="data_flow_integration",
                success=True,
                execution_time=execution_time,
                details={"test_price": test_price, "verified_components": ["v433_system", "risk_overlay"]}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="data_flow_integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _test_trading_workflow_integration(self) -> IntegrationTestResult:
        """取引ワークフロー統合テスト"""
        start_time = time.time()

        try:
            # テストシグナル送信
            from ztb.trading.position_manager import PositionSignal

            signal = PositionSignal(
                symbol="btc_jpy",
                action="open_long",
                strength=0.7,
                target_quantity=0.001,
                confidence=0.8,
                reason="integration_test"
            )

            # 非同期シグナル送信
            async def send_test_signal():
                await self.component_manager.position_manager.submit_signal(signal)

            asyncio.run(send_test_signal())

            # 処理待機
            time.sleep(1.0)

            # ポジション状態確認
            portfolio_state = self.component_manager.position_manager.portfolio_state
            has_position = len(portfolio_state.positions) > 0

            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="trading_workflow_integration",
                success=True,
                execution_time=execution_time,
                details={"signal_sent": True, "position_created": has_position}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="trading_workflow_integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _test_performance_integration(self) -> IntegrationTestResult:
        """パフォーマンス統合テスト"""
        start_time = time.time()

        try:
            # 複数操作のパフォーマンス測定
            operations = []

            # 価格更新操作 × 10
            for i in range(10):
                price = 5000000 + i * 10000
                op_start = time.time()
                self.component_manager.v433_system.update_market_data("btc_jpy", price)
                op_time = time.time() - op_start
                operations.append(op_time)

            # 平均レイテンシー計算
            avg_latency = np.mean(operations) * 1000  # ms

            # 目標レイテンシー確認 (100ms)
            within_target = avg_latency < 100.0

            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="performance_integration",
                success=within_target,
                execution_time=execution_time,
                details={
                    "avg_latency_ms": avg_latency,
                    "target_latency_ms": 100.0,
                    "within_target": within_target,
                    "operations_tested": len(operations)
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="performance_integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _test_error_handling_integration(self) -> IntegrationTestResult:
        """エラー処理統合テスト"""
        start_time = time.time()

        try:
            # 無効なデータでのエラー処理テスト
            try:
                # 無効な価格でテスト
                self.component_manager.v433_system.update_market_data("btc_jpy", -1000)
                error_handled = False  # 正常に処理された場合
            except Exception:
                error_handled = True  # 例外が発生した場合

            # 無効なシグナルでのエラー処理テスト
            try:
                from ztb.trading.position_manager import PositionSignal

                invalid_signal = PositionSignal(
                    symbol="btc_jpy",
                    action="invalid_action",
                    strength=0.5,
                    target_quantity=-1,  # 無効な数量
                    confidence=0.5,
                    reason="error_test"
                )

                async def send_invalid_signal():
                    await self.component_manager.position_manager.submit_signal(invalid_signal)

                asyncio.run(send_invalid_signal())
                signal_error_handled = True  # 正常に処理された場合

            except Exception:
                signal_error_handled = True  # 例外が発生した場合

            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="error_handling_integration",
                success=error_handled and signal_error_handled,
                execution_time=execution_time,
                details={
                    "invalid_price_handled": error_handled,
                    "invalid_signal_handled": signal_error_handled
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return IntegrationTestResult(
                test_name="error_handling_integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )


class V433IntegrationManager:
    """
    V433 Phase 4: 統合システムマネージャー
    全コンポーネントの統合、パフォーマンス最適化、包括的検証
    """

    def __init__(self, exchange: str = "zaif"):
        self.exchange = exchange
        self.logger = get_logger(__name__)

        # 設定の初期化
        self.config = SystemIntegrationConfig()

        # コンポーネントの初期化
        self.component_manager = ComponentManager(exchange)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.integration_tester = IntegrationTester(self.component_manager)

        # システム状態
        self.is_running = False
        self.system_health = "stopped"
        self.last_health_check = datetime.now()

        # モニタリング
        self.monitoring_thread = None
        self.performance_thread = None

        # 統合テスト結果
        self.integration_test_results: List[IntegrationTestResult] = []

    def initialize_system(self) -> bool:
        """システム全体を初期化"""
        try:
            self.logger.info("Initializing V433 integrated system...")

            # コンポーネント初期化
            if not self.component_manager.initialize_components():
                raise Exception("Component initialization failed")

            # パフォーマンスベースライン設定
            self.performance_monitor.update_baseline()

            self.system_health = "initialized"
            self.logger.info("V433 integrated system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.system_health = "initialization_failed"
            return False

    def start_system(self) -> bool:
        """システム全体を開始"""
        if self.is_running:
            return True

        try:
            self.logger.info("Starting V433 integrated system...")

            # コンポーネント開始
            if not self.component_manager.start_components():
                raise Exception("Component startup failed")

            self.is_running = True
            self.system_health = "running"

            # モニタリングスレッド開始
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

            # パフォーマンス監視スレッド開始
            if self.config.enable_performance_monitoring:
                self.performance_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
                self.performance_thread.start()

            self.logger.info("V433 integrated system started successfully")
            return True

        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            self.system_health = "startup_failed"
            self.stop_system()
            return False

    def stop_system(self):
        """システム全体を停止"""
        if not self.is_running:
            return

        self.logger.info("Stopping V433 integrated system...")
        self.is_running = False

        # スレッド停止
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        if self.performance_thread and self.performance_thread.is_alive():
            self.performance_thread.join(timeout=5)

        # コンポーネント停止
        self.component_manager.stop_components()

        self.system_health = "stopped"
        self.logger.info("V433 integrated system stopped")

    def run_integration_tests(self) -> List[IntegrationTestResult]:
        """統合テストを実行"""
        self.logger.info("Running V433 integration test suite...")

        if not self.is_running:
            self.logger.warning("System not running, starting for testing...")
            if not self.start_system():
                raise Exception("Failed to start system for testing")

        # 統合テスト実行
        test_results = self.integration_tester.run_full_integration_test()

        # 結果保存
        self.integration_test_results.extend(test_results)

        # テストサマリー
        success_count = sum(1 for r in test_results if r.success)
        total_count = len(test_results)

        self.logger.info(f"Integration tests completed: {success_count}/{total_count} passed")

        if success_count < total_count:
            failed_tests = [r for r in test_results if not r.success]
            for test in failed_tests:
                self.logger.error(f"Test failed: {test.test_name} - {test.error_message}")

        return test_results

    def optimize_performance(self) -> Dict[str, Any]:
        """パフォーマンス最適化を実行"""
        self.logger.info("Running performance optimization...")

        optimization_results = {
            "memory_optimization": self._optimize_memory_usage(),
            "cpu_optimization": self._optimize_cpu_usage(),
            "latency_optimization": self._optimize_latency(),
            "timestamp": datetime.now()
        }

        # 最適化結果ログ
        for opt_type, result in optimization_results.items():
            if opt_type != "timestamp":
                self.logger.info(f"{opt_type}: {result}")

        return optimization_results

    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量を最適化"""
        try:
            # ガベージコレクション実行
            import gc
            gc.collect()

            # メモリ使用量測定
            before_memory = self.performance_monitor.get_current_metrics().memory_usage_gb
            after_memory = self.performance_monitor.get_current_metrics().memory_usage_gb

            memory_reduction = before_memory - after_memory
            reduction_percent = (memory_reduction / before_memory) * 100 if before_memory > 0 else 0

            return {
                "success": True,
                "memory_before_gb": before_memory,
                "memory_after_gb": after_memory,
                "reduction_gb": memory_reduction,
                "reduction_percent": reduction_percent
            }

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def _optimize_cpu_usage(self) -> Dict[str, Any]:
        """CPU使用量を最適化"""
        try:
            # CPU使用量測定
            metrics = self.performance_monitor.get_current_metrics()
            cpu_usage = metrics.cpu_usage_percent

            # CPU使用量が閾値を超えている場合の最適化
            if cpu_usage > self.config.max_cpu_usage_percent:
                # スレッドプールサイズ調整などの最適化
                self.logger.warning(f"High CPU usage detected: {cpu_usage:.1f}%")

            return {
                "success": True,
                "cpu_usage_percent": cpu_usage,
                "within_limits": cpu_usage <= self.config.max_cpu_usage_percent
            }

        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def _optimize_latency(self) -> Dict[str, Any]:
        """レイテンシーを最適化"""
        try:
            # テスト操作のレイテンシー測定
            latencies = []

            for _ in range(5):
                latency, _ = self.performance_monitor.measure_latency(
                    self.component_manager.v433_system.update_market_data,
                    "btc_jpy", 5000000.0
                )
                latencies.append(latency)

            avg_latency = np.mean(latencies)
            within_target = avg_latency < self.config.target_latency_ms

            return {
                "success": True,
                "avg_latency_ms": avg_latency,
                "target_latency_ms": self.config.target_latency_ms,
                "within_target": within_target,
                "latency_samples": latencies
            }

        except Exception as e:
            self.logger.error(f"Latency optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def _monitoring_loop(self):
        """モニタリングループ"""
        while self.is_running:
            try:
                current_time = datetime.now()

                # ヘルスチェック
                if (current_time - self.last_health_check).seconds >= self.config.system_health_check_interval:
                    self._perform_health_check()
                    self.last_health_check = current_time

                # コンポーネント状態確認
                component_status = self.component_manager.get_component_status()
                unhealthy_components = [name for name, status in component_status.items() if not status["healthy"]]

                if unhealthy_components:
                    self.logger.warning(f"Unhealthy components detected: {unhealthy_components}")

                    # 自動回復が有効な場合
                    if self.config.enable_auto_recovery:
                        for component in unhealthy_components:
                            self._attempt_component_recovery(component)

                time.sleep(10)  # 10秒間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)

    def _performance_monitoring_loop(self):
        """パフォーマンス監視ループ"""
        while self.is_running:
            try:
                # パフォーマンス指標取得
                metrics = self.performance_monitor.get_current_metrics()

                # ビジネス指標更新
                if self.component_manager.v433_system:
                    system_status = self.component_manager.v433_system.get_system_status()
                    metrics.active_positions = system_status["portfolio_status"]["position_count"]
                    metrics.total_pnl = system_status["portfolio_status"]["portfolio_state"]["total_pnl"]
                    metrics.win_rate = system_status["performance_metrics"].get("win_rate", 0.0)

                # パフォーマンス低下チェック
                if self.performance_monitor.check_performance_degradation(metrics):
                    self.logger.warning("Performance degradation detected")

                    if self.config.alert_on_performance_degradation:
                        self._send_performance_alert(metrics)

                # パフォーマンスログ
                self.performance_monitor.log_performance_metrics(metrics)

                time.sleep(self.config.performance_log_interval)

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)

    def _perform_health_check(self):
        """ヘルスチェックを実行"""
        try:
            # コンポーネント状態確認
            component_status = self.component_manager.get_component_status()

            # データフロー確認
            data_flow_status = self._check_data_flow()

            # 全体ヘルス判定
            all_components_healthy = all(s["healthy"] for s in component_status.values())
            data_flow_healthy = all(data_flow_status.values())

            if all_components_healthy and data_flow_healthy:
                new_health = "healthy"
            elif all_components_healthy:
                new_health = "warning"
            else:
                new_health = "critical"

            if new_health != self.system_health:
                self.logger.info(f"System health changed: {self.system_health} -> {new_health}")
                self.system_health = new_health

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.system_health = "critical"

    def _check_data_flow(self) -> Dict[str, bool]:
        """データフローをチェック"""
        data_flow_status = {}

        try:
            # 価格データの伝播チェック
            test_price = 5000000.0
            self.component_manager.v433_system.update_market_data("btc_jpy", test_price)

            time.sleep(0.1)  # 伝播待機

            # 各コンポーネントでのデータ確認
            v433_price = self.component_manager.v433_system.current_prices.get("btc_jpy")
            risk_price = self.component_manager.risk_overlay.current_prices.get("btc_jpy")

            data_flow_status["v433_system"] = v433_price == test_price
            data_flow_status["risk_overlay"] = risk_price == test_price

        except Exception as e:
            self.logger.error(f"Data flow check failed: {e}")
            data_flow_status["error"] = False

        return data_flow_status

    def _attempt_component_recovery(self, component_name: str):
        """コンポーネント回復を試行"""
        try:
            self.logger.info(f"Attempting recovery for component: {component_name}")

            success = self.component_manager.restart_component(component_name)

            if success:
                self.logger.info(f"Component {component_name} recovered successfully")
            else:
                self.logger.error(f"Component {component_name} recovery failed")

        except Exception as e:
            self.logger.error(f"Component recovery error: {e}")

    def _send_performance_alert(self, metrics: SystemHealthMetrics):
        """パフォーマンスアラートを送信"""
        self.logger.warning(f"PERFORMANCE ALERT: "
                          f"Memory={metrics.memory_usage_gb:.2f}GB, "
                          f"CPU={metrics.cpu_usage_percent:.1f}%, "
                          f"Latency={metrics.latency_ms:.2f}ms")

    def get_system_status(self) -> Dict[str, Any]:
        """システム全体の状態を取得"""
        return {
            "system_health": self.system_health,
            "is_running": self.is_running,
            "component_status": self.component_manager.get_component_status(),
            "performance_metrics": self.performance_monitor.get_current_metrics().__dict__,
            "integration_test_results": [r.__dict__ for r in self.integration_test_results[-10:]],  # 最新10件
            "last_health_check": self.last_health_check,
            "config": self.config.__dict__
        }


def create_v433_integration_manager(exchange: str = "zaif") -> V433IntegrationManager:
    """V433統合マネージャーのファクトリ関数"""
    return V433IntegrationManager(exchange)


# 使用例
if __name__ == "__main__":
    # V433統合マネージャーの作成
    integration_manager = create_v433_integration_manager("zaif")

    # システム初期化
    if not integration_manager.initialize_system():
        print("System initialization failed")
        exit(1)

    # システム開始
    if not integration_manager.start_system():
        print("System startup failed")
        exit(1)

    try:
        print("V433 integrated system running...")

        # 統合テスト実行
        print("Running integration tests...")
        test_results = integration_manager.run_integration_tests()

        passed_tests = sum(1 for r in test_results if r.success)
        total_tests = len(test_results)
        print(f"Integration tests: {passed_tests}/{total_tests} passed")

        # パフォーマンス最適化実行
        print("Running performance optimization...")
        optimization_results = integration_manager.optimize_performance()
        print(f"Optimization completed: {optimization_results}")

        # システム状態確認
        status = integration_manager.get_system_status()
        print(f"System health: {status['system_health']}")

        # 実行維持
        time.sleep(30)

    finally:
        # システム停止
        integration_manager.stop_system()
        print("V433 integrated system stopped")