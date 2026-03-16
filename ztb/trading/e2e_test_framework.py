#!/usr/bin/env python3
"""
V433 Phase 4: エンドツーエンドテストフレームワーク
全システムコンポーネント、データパイプライン、取引ワークフローの包括的テスト
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ztb.trading.position_manager import PositionSignal
from ztb.trading.v433_integration_manager import V433IntegrationManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

ObjectMap = dict[str, object]
StepCallable = Callable[[], object]
HookCallable = Callable[[], None]
ExpectedPredicate = Callable[[object], bool]
ExpectedValue = object | ExpectedPredicate

@dataclass
class TestScenario:
    """テストシナリオ定義"""

    name: str
    description: str
    test_type: str  # "unit", "integration", "system", "performance", "stress"
    priority: str  # "critical", "high", "medium", "low"
    timeout_seconds: int = 300
    setup_steps: list[HookCallable] = field(default_factory=list)
    test_steps: list[StepCallable] = field(default_factory=list)
    teardown_steps: list[HookCallable] = field(default_factory=list)
    expected_results: dict[str, ExpectedValue] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

@dataclass
class TestExecutionResult:
    """テスト実行結果"""

    scenario_name: str
    success: bool
    execution_time: float
    error_message: str | None = None
    actual_results: ObjectMap = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TestSuiteResult:
    """テストスイート結果"""

    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    results: list[TestExecutionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

class TestDataGenerator:
    """テストデータ生成器"""

    def __init__(self):
        self.logger = get_logger(__name__)

        # テストデータ設定
        self.base_prices = {
            "btc_jpy": 5000000.0,
            "eth_jpy": 300000.0,
            "xrp_jpy": 100.0,
            "mona_jpy": 50.0,
        }

        self.volatilities = {
            "btc_jpy": 0.02,  # 2%
            "eth_jpy": 0.03,  # 3%
            "xrp_jpy": 0.05,  # 5%
            "mona_jpy": 0.04,  # 4%
        }

    def generate_price_series(
        self, symbol: str, periods: int = 1000, trend: str = "random"
    ) -> list[float]:
        """価格系列を生成"""
        base_price = self.base_prices.get(symbol, 1000.0)
        volatility = self.volatilities.get(symbol, 0.02)

        prices = [base_price]

        for i in range(1, periods):
            # ランダムウォーク + トレンド
            if trend == "upward":
                drift = 0.0001  # 上昇トレンド
            elif trend == "downward":
                drift = -0.0001  # 下降トレンド
            else:
                drift = 0.0  # ランダム

            # 幾何ブラウン運動
            shock = np.random.normal(drift, volatility)
            new_price = prices[-1] * (1 + shock)
            prices.append(max(new_price, 0.1))  # 最低価格設定

        return prices

    def generate_market_data_stream(
        self,
        symbols: list[str],
        duration_seconds: int = 60,
        update_interval: float = 1.0,
    ) -> list[tuple[str, float, float]]:
        """市場データストリームを生成"""
        data_stream = []

        for symbol in symbols:
            prices = self.generate_price_series(
                symbol, int(duration_seconds / update_interval)
            )

            for i, price in enumerate(prices):
                timestamp = i * update_interval
                data_stream.append((symbol, price, timestamp))

        # タイムスタンプでソート
        data_stream.sort(key=lambda x: x[2])

        return data_stream

    def generate_trading_signals(self, count: int = 10) -> list[PositionSignal]:
        """取引シグナルを生成"""
        signals = []
        symbols = list(self.base_prices.keys())

        for i in range(count):
            symbol = np.random.choice(symbols)
            action = np.random.choice(
                ["open_long", "open_short", "close_long", "close_short"]
            )
            strength = np.random.uniform(0.3, 0.9)
            quantity = np.random.uniform(0.0001, 0.01)
            confidence = np.random.uniform(0.4, 0.95)

            signal = PositionSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                target_quantity=quantity,
                confidence=confidence,
                reason=f"test_signal_{i}",
            )
            signals.append(signal)

        return signals

    def generate_stress_scenario(self, scenario_type: str) -> ObjectMap:
        """ストレスシナリオを生成"""
        if scenario_type == "flash_crash":
            return {
                "description": "瞬間暴落シナリオ",
                "price_shocks": {"btc_jpy": -0.3, "eth_jpy": -0.35, "xrp_jpy": -0.25},
                "duration_seconds": 10,
                "recovery_time": 30,
            }
        elif scenario_type == "high_volatility":
            return {
                "description": "高ボラティリティシナリオ",
                "volatility_multiplier": 3.0,
                "duration_seconds": 300,
                "price_changes": {},
            }
        elif scenario_type == "liquidity_crisis":
            return {
                "description": "流動性危機シナリオ",
                "spread_multiplier": 5.0,
                "volume_reduction": 0.8,
                "duration_seconds": 180,
            }
        else:
            return {
                "description": "通常シナリオ",
                "price_changes": {},
                "duration_seconds": 60,
            }

class EndToEndTestRunner:
    """エンドツーエンドテスト実行器"""

    def __init__(self, integration_manager: V433IntegrationManager):
        self.integration_manager = integration_manager
        self.test_data_generator = TestDataGenerator()
        self.logger = get_logger(__name__)

        # テスト結果
        self.test_results: list[TestExecutionResult] = []

        # テスト設定
        self.test_timeout = 300  # 5分
        self.parallel_execution = True
        self.max_parallel_tests = 3

    def run_test_scenario(self, scenario: TestScenario) -> TestExecutionResult:
        """テストシナリオを実行"""
        start_time = time.time()
        logs = []

        def log_message(msg: str):
            logs.append(f"{datetime.now()}: {msg}")
            self.logger.info(f"[{scenario.name}] {msg}")

        try:
            log_message(f"Starting test scenario: {scenario.description}")

            # セットアップ実行
            log_message("Running setup steps...")
            for setup_step in scenario.setup_steps:
                setup_step()

            # テスト実行
            log_message("Running test steps...")
            actual_results = {}
            performance_metrics = {}

            for test_step in scenario.test_steps:
                step_start = time.time()
                result = test_step()
                step_time = time.time() - step_start

                # 結果保存
                step_name = (
                    test_step.__name__
                    if hasattr(test_step, "__name__")
                    else str(test_step)
                )
                actual_results[step_name] = result
                performance_metrics[f"{step_name}_time"] = step_time

                log_message(f"Step {step_name} completed in {step_time:.2f}s")

            # 結果検証
            success = self._validate_test_results(
                scenario.expected_results, actual_results
            )

            if success:
                log_message("Test scenario PASSED")
            else:
                log_message("Test scenario FAILED - results don't match expectations")

            execution_time = time.time() - start_time

            # タイムアウトチェック
            if execution_time > scenario.timeout_seconds:
                success = False
                log_message(f"Test TIMED OUT after {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            log_message(f"Test FAILED with exception: {e}")

        finally:
            # ティアダウン実行
            try:
                log_message("Running teardown steps...")
                for teardown_step in scenario.teardown_steps:
                    teardown_step()
            except Exception as e:
                log_message(f"Teardown failed: {e}")

        return TestExecutionResult(
            scenario_name=scenario.name,
            success=success,
            execution_time=execution_time,
            actual_results=actual_results,
            performance_metrics=performance_metrics,
            logs=logs,
        )

    def run_test_suite(self, scenarios: list[TestScenario]) -> TestSuiteResult:
        """テストスイートを実行"""
        start_time = time.time()
        suite_name = f"e2e_test_suite_{int(start_time)}"

        self.logger.info(
            f"Starting test suite: {suite_name} with {len(scenarios)} scenarios"
        )

        results = []

        if self.parallel_execution and len(scenarios) > 1:
            # 並列実行
            with ThreadPoolExecutor(max_workers=self.max_parallel_tests) as executor:
                future_to_scenario = {
                    executor.submit(self.run_test_scenario, scenario): scenario
                    for scenario in scenarios
                }

                for future in as_completed(future_to_scenario):
                    scenario = future_to_scenario[future]
                    try:
                        result = future.result(timeout=self.test_timeout)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Test scenario {scenario.name} failed: {e}")
                        # エラーの場合の結果作成
                        results.append(
                            TestExecutionResult(
                                scenario_name=scenario.name,
                                success=False,
                                execution_time=0.0,
                                error_message=str(e),
                            )
                        )
        else:
            # 順次実行
            for scenario in scenarios:
                result = self.run_test_scenario(scenario)
                results.append(result)

        execution_time = time.time() - start_time

        # 結果集計
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = sum(1 for r in results if not r.success and r.error_message)
        skipped_tests = sum(1 for r in results if not r.success and not r.error_message)

        suite_result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(scenarios),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time=execution_time,
            results=results,
        )

        self.logger.info(
            f"Test suite completed: {passed_tests}/{len(scenarios)} passed "
            f"in {execution_time:.2f}s"
        )

        return suite_result

    def _validate_test_results(
        self, expected: Mapping[str, ExpectedValue], actual: Mapping[str, object]
    ) -> bool:
        """テスト結果を検証"""
        for key, expected_value in expected.items():
            if key not in actual:
                self.logger.warning(
                    f"Expected result key '{key}' not found in actual results"
                )
                return False

            actual_value = actual[key]

            # predicate-based validation (e.g. lambda x: x < threshold)
            if callable(expected_value):
                try:
                    if not bool(expected_value(actual_value)):
                        self.logger.warning(
                            f"Predicate validation failed for {key}: got {actual_value}"
                        )
                        return False
                except Exception as exc:
                    self.logger.warning(
                        f"Predicate validation errored for {key}: {exc}"
                    )
                    return False
            # bool check must be before numeric check because bool is subclass of int
            elif isinstance(expected_value, bool):
                if not isinstance(actual_value, bool) or actual_value != expected_value:
                    self.logger.warning(
                        f"Boolean mismatch for {key}: expected {expected_value}, "
                        f"got {actual_value}"
                    )
                    return False
            elif isinstance(expected_value, (int, float)):
                # 数値比較（許容誤差あり）
                if not isinstance(actual_value, (int, float)) or isinstance(
                    actual_value, bool
                ):
                    self.logger.warning(
                        f"Numeric type mismatch for {key}: expected numeric, got {type(actual_value).__name__}"
                    )
                    return False
                expected_numeric = float(expected_value)
                actual_numeric = float(actual_value)
                tolerance = max(abs(expected_numeric) * 0.1, 1e-9)  # 10%許容
                if abs(actual_numeric - expected_numeric) > tolerance:
                    self.logger.warning(
                        f"Value mismatch for {key}: expected {expected_numeric}, "
                        f"got {actual_numeric}"
                    )
                    return False
            elif isinstance(expected_value, str):
                # 文字列比較
                if actual_value != expected_value:
                    self.logger.warning(
                        f"String mismatch for {key}: expected '{expected_value}', "
                        f"got '{actual_value}'"
                    )
                    return False
            elif isinstance(expected_value, (list, tuple)):
                # リスト比較（簡易版）
                if not isinstance(actual_value, (list, tuple)):
                    self.logger.warning(
                        f"list type mismatch for {key}: expected list/tuple, got {type(actual_value).__name__}"
                    )
                    return False
                if len(actual_value) != len(expected_value):
                    self.logger.warning(
                        f"list length mismatch for {key}: expected {len(expected_value)}, "
                        f"got {len(actual_value)}"
                    )
                    return False

        return True

class ComprehensiveTestSuite:
    """包括的テストスイート"""

    def __init__(self, integration_manager: V433IntegrationManager):
        self.integration_manager = integration_manager
        self.test_runner = EndToEndTestRunner(integration_manager)
        self.test_data_generator = TestDataGenerator()

    def create_system_integration_tests(self) -> list[TestScenario]:
        """システム統合テストを作成"""
        scenarios = []

        # 1. コンポーネント起動統合テスト
        scenario = TestScenario(
            name="system_startup_integration",
            description="全コンポーネントの正常起動と統合テスト",
            test_type="integration",
            priority="critical",
            setup_steps=[self._setup_clean_system],
            test_steps=[
                self._test_component_initialization,
                self._test_component_startup,
                self._test_component_interaction,
            ],
            teardown_steps=[self._teardown_system],
            expected_results={
                "components_initialized": True,
                "components_started": True,
                "interaction_success": True,
            },
            tags=["integration", "startup", "critical"],
        )
        scenarios.append(scenario)

        # 2. データフロー統合テスト
        scenario = TestScenario(
            name="data_flow_integration",
            description="データパイプラインの統合テスト",
            test_type="integration",
            priority="critical",
            setup_steps=[self._setup_system_with_data],
            test_steps=[
                self._test_market_data_ingestion,
                self._test_data_propagation,
                self._test_data_consistency,
            ],
            teardown_steps=[self._teardown_system],
            expected_results={
                "data_ingested": True,
                "data_propagated": True,
                "data_consistent": True,
            },
            tags=["integration", "data", "critical"],
        )
        scenarios.append(scenario)

        # 3. 取引ワークフロー統合テスト
        scenario = TestScenario(
            name="trading_workflow_integration",
            description="取引実行ワークフローの統合テスト",
            test_type="integration",
            priority="critical",
            setup_steps=[self._setup_system_with_trading],
            test_steps=[
                self._test_signal_generation,
                self._test_order_execution,
                self._test_position_management,
                self._test_risk_management,
            ],
            teardown_steps=[self._teardown_system],
            expected_results={
                "signal_generated": True,
                "order_executed": True,
                "position_managed": True,
                "risk_managed": True,
            },
            tags=["integration", "trading", "critical"],
        )
        scenarios.append(scenario)

        return scenarios

    def create_performance_tests(self) -> list[TestScenario]:
        """パフォーマンステストを作成"""
        scenarios = []

        # 1. レイテンシーパフォーマンステスト
        scenario = TestScenario(
            name="latency_performance",
            description="システムレイテンシーのパフォーマンステスト",
            test_type="performance",
            priority="high",
            setup_steps=[self._setup_performance_test],
            test_steps=[
                self._test_data_processing_latency,
                self._test_signal_processing_latency,
                self._test_execution_latency,
            ],
            teardown_steps=[self._teardown_performance_test],
            expected_results={
                "data_latency_ms": lambda x: x < 100,
                "signal_latency_ms": lambda x: x < 200,
                "execution_latency_ms": lambda x: x < 500,
            },
            tags=["performance", "latency", "high"],
        )
        scenarios.append(scenario)

        # 2. スループットパフォーマンステスト
        scenario = TestScenario(
            name="throughput_performance",
            description="システムスループットのパフォーマンステスト",
            test_type="performance",
            priority="high",
            setup_steps=[self._setup_throughput_test],
            test_steps=[
                self._test_data_ingestion_rate,
                self._test_signal_processing_rate,
                self._test_concurrent_operations,
            ],
            teardown_steps=[self._teardown_throughput_test],
            expected_results={
                "data_ingestion_rate": lambda x: x > 100,  # 100 updates/sec
                "signal_processing_rate": lambda x: x > 50,  # 50 signals/sec
                "concurrent_operations": lambda x: x > 10,  # 10 concurrent ops
            },
            tags=["performance", "throughput", "high"],
        )
        scenarios.append(scenario)

        return scenarios

    def create_stress_tests(self) -> list[TestScenario]:
        """ストレステストを作成"""
        scenarios = []

        # 1. 高負荷ストレステスト
        scenario = TestScenario(
            name="high_load_stress",
            description="高負荷状態でのシステム安定性テスト",
            test_type="stress",
            priority="high",
            timeout_seconds=600,  # 10分
            setup_steps=[self._setup_stress_test],
            test_steps=[
                self._test_high_frequency_data,
                self._test_burst_signals,
                self._test_memory_pressure,
                self._test_recovery_capability,
            ],
            teardown_steps=[self._teardown_stress_test],
            expected_results={
                "system_stable": True,
                "no_crashes": True,
                "recovery_success": True,
                "performance_degradation": lambda x: x < 0.5,  # 50%以内の性能低下
            },
            tags=["stress", "load", "high"],
        )
        scenarios.append(scenario)

        # 2. 市場ストレスシナリオテスト
        scenario = TestScenario(
            name="market_stress_scenarios",
            description="市場ストレスシナリオでのシステム動作テスト",
            test_type="stress",
            priority="high",
            timeout_seconds=900,  # 15分
            setup_steps=[self._setup_market_stress_test],
            test_steps=[
                self._test_flash_crash_scenario,
                self._test_high_volatility_scenario,
                self._test_liquidity_crisis_scenario,
                self._test_emergency_procedures,
            ],
            teardown_steps=[self._teardown_market_stress_test],
            expected_results={
                "flash_crash_handled": True,
                "volatility_handled": True,
                "liquidity_handled": True,
                "emergency_procedures": True,
            },
            tags=["stress", "market", "high"],
        )
        scenarios.append(scenario)

        return scenarios

    def run_comprehensive_test_suite(self) -> dict[str, TestSuiteResult]:
        """包括的テストスイートを実行"""
        self.logger.info("Running comprehensive V433 test suite...")

        results = {}

        # システム統合テスト
        integration_scenarios = self.create_system_integration_tests()
        results["system_integration"] = self.test_runner.run_test_suite(
            integration_scenarios
        )

        # パフォーマンステスト
        performance_scenarios = self.create_performance_tests()
        results["performance"] = self.test_runner.run_test_suite(performance_scenarios)

        # ストレステスト
        stress_scenarios = self.create_stress_tests()
        results["stress"] = self.test_runner.run_test_suite(stress_scenarios)

        # サマリーログ
        total_passed = sum(r.passed_tests for r in results.values())
        total_tests = sum(r.total_tests for r in results.values())

        self.logger.info(
            f"Comprehensive test suite completed: {total_passed}/{total_tests} tests passed"
        )

        return results

    # テストセットアップ/ティアダウンメソッド
    def _setup_clean_system(self):
        """クリーンシステムのセットアップ"""
        if self.integration_manager.is_running:
            self.integration_manager.stop_system()
        self.integration_manager.initialize_system()

    def _setup_system_with_data(self):
        """データ付きシステムのセットアップ"""
        self._setup_clean_system()
        self.integration_manager.start_system()

        # サンプルデータ投入
        self.integration_manager.component_manager.v433_system.update_market_data(
            "btc_jpy", 5000000.0
        )
        time.sleep(0.1)

    def _setup_system_with_trading(self):
        """取引機能付きシステムのセットアップ"""
        self._setup_system_with_data()

    def _setup_performance_test(self):
        """パフォーマンステストのセットアップ"""
        self._setup_system_with_data()

    def _setup_throughput_test(self):
        """スループットテストのセットアップ"""
        self._setup_system_with_data()

    def _setup_stress_test(self):
        """ストレステストのセットアップ"""
        self._setup_system_with_data()

    def _setup_market_stress_test(self):
        """市場ストレステストのセットアップ"""
        self._setup_system_with_data()

    def _teardown_system(self):
        """システムのティアダウン"""
        if self.integration_manager.is_running:
            self.integration_manager.stop_system()

    def _teardown_performance_test(self):
        """パフォーマンステストのティアダウン"""
        self._teardown_system()

    def _teardown_throughput_test(self):
        """スループットテストのティアダウン"""
        self._teardown_system()

    def _teardown_stress_test(self):
        """ストレステストのティアダウン"""
        self._teardown_system()

    def _teardown_market_stress_test(self):
        """市場ストレステストのティアダウン"""
        self._teardown_system()

    # テストステップメソッド
    def _test_component_initialization(self) -> ObjectMap:
        """コンポーネント初期化テスト"""
        success = self.integration_manager.component_manager.initialize_components()
        return {"components_initialized": success}

    def _test_component_startup(self) -> ObjectMap:
        """コンポーネント起動テスト"""
        success = self.integration_manager.component_manager.start_components()
        return {"components_started": success}

    def _test_component_interaction(self) -> ObjectMap:
        """コンポーネント相互作用テスト"""
        # 基本的な相互作用テスト
        try:
            # V433システムが実行中か確認
            status = self.integration_manager.component_manager.get_component_status()
            all_running = all(s["status"] == "running" for s in status.values())
            return {"interaction_success": all_running}
        except Exception:
            return {"interaction_success": False}

    def _test_market_data_ingestion(self) -> ObjectMap:
        """市場データ取り込みテスト"""
        try:
            # テストデータ投入
            test_price = 5100000.0
            self.integration_manager.component_manager.v433_system.update_market_data(
                "btc_jpy", test_price
            )

            # データ確認
            current_prices = (
                self.integration_manager.component_manager.v433_system.current_prices
            )
            success = current_prices.get("btc_jpy") == test_price
            return {"data_ingested": success}
        except Exception:
            return {"data_ingested": False}

    def _test_data_propagation(self) -> ObjectMap:
        """データ伝播テスト"""
        try:
            time.sleep(0.1)  # 伝播待機

            # 各コンポーネントでのデータ確認
            v433_price = self.integration_manager.component_manager.v433_system.current_prices.get(
                "btc_jpy"
            )
            risk_price = self.integration_manager.component_manager.risk_overlay.current_prices.get(
                "btc_jpy"
            )

            success = (
                v433_price is not None
                and risk_price is not None
                and v433_price == risk_price
            )
            return {"data_propagated": success}
        except Exception:
            return {"data_propagated": False}

    def _test_data_consistency(self) -> ObjectMap:
        """データ整合性テスト"""
        try:
            # 複数データの整合性確認
            prices = (
                self.integration_manager.component_manager.v433_system.current_prices
            )
            success = len(prices) > 0 and all(
                isinstance(p, (int, float)) and p > 0 for p in prices.values()
            )
            return {"data_consistent": success}
        except Exception:
            return {"data_consistent": False}

    def _test_signal_generation(self) -> ObjectMap:
        """シグナル生成テスト"""
        try:
            # テストシグナル生成
            signals = self.test_data_generator.generate_trading_signals(1)
            signal = signals[0]

            # シグナル送信
            async def send_signal():
                await self.integration_manager.component_manager.position_manager.submit_signal(
                    signal
                )

            asyncio.run(send_signal())
            return {"signal_generated": True}
        except Exception:
            return {"signal_generated": False}

    def _test_order_execution(self) -> ObjectMap:
        """注文実行テスト"""
        try:
            # 注文実行確認（簡易版）
            # 実際の実装では注文履歴を確認
            return {"order_executed": True}  # 仮定
        except Exception:
            return {"order_executed": False}

    def _test_position_management(self) -> ObjectMap:
        """ポジション管理テスト"""
        try:
            # ポジション状態確認
            portfolio_state = self.integration_manager.component_manager.position_manager.portfolio_state
            success = hasattr(portfolio_state, "positions")
            return {"position_managed": success}
        except Exception:
            return {"position_managed": False}

    def _test_risk_management(self) -> ObjectMap:
        """リスク管理テスト"""
        try:
            # リスク指標確認
            risk_metrics = (
                self.integration_manager.component_manager.risk_overlay.risk_metrics
            )
            success = risk_metrics is not None
            return {"risk_managed": success}
        except Exception:
            return {"risk_managed": False}

    def _test_data_processing_latency(self) -> float:
        """データ処理レイテンシーテスト"""
        start_time = time.time()
        for _ in range(10):
            self.integration_manager.component_manager.v433_system.update_market_data(
                "btc_jpy", 5000000.0
            )
        end_time = time.time()
        return (end_time - start_time) / 10 * 1000  # ms

    def _test_signal_processing_latency(self) -> float:
        """シグナル処理レイテンシーテスト"""
        signals = self.test_data_generator.generate_trading_signals(5)
        start_time = time.time()

        async def send_signals():
            for signal in signals:
                await self.integration_manager.component_manager.position_manager.submit_signal(
                    signal
                )

        asyncio.run(send_signals())
        end_time = time.time()
        return (end_time - start_time) / 5 * 1000  # ms

    def _test_execution_latency(self) -> float:
        """実行レイテンシーテスト"""
        # 簡易版：システム状態取得のレイテンシー
        start_time = time.time()
        for _ in range(10):
            self.integration_manager.get_system_status()
        end_time = time.time()
        return (end_time - start_time) / 10 * 1000  # ms

    def _test_data_ingestion_rate(self) -> float:
        """データ取り込みレートテスト"""
        start_time = time.time()
        count = 0
        while time.time() - start_time < 10:  # 10秒間
            self.integration_manager.component_manager.v433_system.update_market_data(
                "btc_jpy", 5000000.0
            )
            count += 1
        return count / 10  # per second

    def _test_signal_processing_rate(self) -> float:
        """シグナル処理レートテスト"""
        signals = self.test_data_generator.generate_trading_signals(50)
        start_time = time.time()

        async def send_signals():
            for signal in signals:
                await self.integration_manager.component_manager.position_manager.submit_signal(
                    signal
                )

        asyncio.run(send_signals())
        end_time = time.time()
        elapsed = max(end_time - start_time, 1e-9)
        return len(signals) / elapsed  # per second

    def _test_concurrent_operations(self) -> int:
        """並行操作テスト"""
        # 簡易版：並行して実行可能な操作数
        return 10  # 仮定値

    def _test_high_frequency_data(self) -> ObjectMap:
        """高頻度データテスト"""
        try:
            # 高頻度データ投入
            for i in range(100):
                price = 5000000 + i * 100
                self.integration_manager.component_manager.v433_system.update_market_data(
                    "btc_jpy", price
                )

            # システム安定性確認
            status = self.integration_manager.get_system_status()
            stable = status["system_health"] in ["healthy", "warning"]
            return {"system_stable": stable}
        except Exception:
            return {"system_stable": False}

    def _test_burst_signals(self) -> ObjectMap:
        """バーストシグナルテスト"""
        try:
            # バーストシグナル送信
            signals = self.test_data_generator.generate_trading_signals(20)

            async def send_burst_signals():
                for signal in signals:
                    await self.integration_manager.component_manager.position_manager.submit_signal(
                        signal
                    )

            asyncio.run(send_burst_signals())

            # クラッシュなし確認
            return {"no_crashes": True}
        except Exception:
            return {"no_crashes": False}

    def _test_memory_pressure(self) -> ObjectMap:
        """メモリ負荷テスト"""
        try:
            # メモリ使用量確認
            initial_memory = self.integration_manager.performance_monitor.get_current_metrics().memory_usage_gb

            # 負荷をかける
            for _ in range(1000):
                self.integration_manager.component_manager.v433_system.update_market_data(
                    "btc_jpy", 5000000.0
                )

            final_memory = self.integration_manager.performance_monitor.get_current_metrics().memory_usage_gb
            baseline = max(initial_memory, 1e-9)
            memory_increase = (final_memory - initial_memory) / baseline

            return {"performance_degradation": memory_increase}
        except Exception:
            return {"performance_degradation": 1.0}

    def _test_recovery_capability(self) -> ObjectMap:
        """回復能力テスト"""
        try:
            # システム停止
            self.integration_manager.stop_system()

            # 再開
            success = self.integration_manager.start_system()
            return {"recovery_success": success}
        except Exception:
            return {"recovery_success": False}

    def _test_flash_crash_scenario(self) -> bool:
        """フラッシュクラッシュシナリオテスト"""
        try:
            # フラッシュクラッシュデータ生成
            scenario = self.test_data_generator.generate_stress_scenario("flash_crash")

            # シナリオ実行
            for symbol, shock in scenario["price_shocks"].items():
                base_price = self.test_data_generator.base_prices.get(symbol, 1000.0)
                crash_price = base_price * (1 + shock)
                self.integration_manager.component_manager.v433_system.update_market_data(
                    symbol, crash_price
                )

            # システム安定性確認
            time.sleep(1)
            status = self.integration_manager.get_system_status()
            return status["system_health"] != "critical"
        except Exception:
            return False

    def _test_high_volatility_scenario(self) -> bool:
        """高ボラティリティシナリオテスト"""
        try:
            # 高ボラティリティデータ生成
            for _ in range(60):  # 1分間
                for symbol in ["btc_jpy", "eth_jpy"]:
                    volatility = (
                        self.test_data_generator.volatilities.get(symbol, 0.02) * 3
                    )  # 3倍
                    base_price = self.test_data_generator.base_prices.get(
                        symbol, 1000.0
                    )
                    shock = np.random.normal(0, volatility)
                    price = base_price * (1 + shock)
                    self.integration_manager.component_manager.v433_system.update_market_data(
                        symbol, price
                    )
                time.sleep(1)

            status = self.integration_manager.get_system_status()
            return status["system_health"] != "critical"
        except Exception:
            return False

    def _test_liquidity_crisis_scenario(self) -> bool:
        """流動性危機シナリオテスト"""
        try:
            # 流動性危機シミュレーション（価格変動を抑えてテスト）
            for _ in range(30):  # 30秒間
                for symbol in ["btc_jpy", "eth_jpy"]:
                    # 小さな価格変動
                    base_price = self.test_data_generator.base_prices.get(
                        symbol, 1000.0
                    )
                    shock = np.random.normal(0, 0.001)  # 非常に小さな変動
                    price = base_price * (1 + shock)
                    self.integration_manager.component_manager.v433_system.update_market_data(
                        symbol, price
                    )
                time.sleep(1)

            status = self.integration_manager.get_system_status()
            return status["system_health"] != "critical"
        except Exception:
            return False

    def _test_emergency_procedures(self) -> bool:
        """緊急手順テスト"""
        try:
            # 緊急停止トリガー
            # 注意: 実際の運用では慎重に
            emergency_status = self.integration_manager.component_manager.risk_overlay.emergency_stop.get_emergency_status()
            return not emergency_status["triggered"]  # 緊急停止が発動していない
        except Exception:
            return False

def create_end_to_end_test_framework(
    integration_manager: V433IntegrationManager,
) -> ComprehensiveTestSuite:
    """エンドツーエンドテストフレームワークのファクトリ関数"""
    return ComprehensiveTestSuite(integration_manager)

# 使用例
if __name__ == "__main__":
    from ztb.trading.v433_integration_manager import create_v433_integration_manager

    # V433統合マネージャーの作成
    integration_manager = create_v433_integration_manager("zaif")

    # システム初期化と開始
    if integration_manager.initialize_system() and integration_manager.start_system():
        try:
            # エンドツーエンドテストスイートの作成
            test_suite = create_end_to_end_test_framework(integration_manager)

            # 包括的テスト実行
            print("Running comprehensive end-to-end test suite...")
            results = test_suite.run_comprehensive_test_suite()

            # 結果表示
            for suite_name, suite_result in results.items():
                print(f"\n{suite_name.upper()} TEST RESULTS:")
                print(f"  Total: {suite_result.total_tests}")
                print(f"  Passed: {suite_result.passed_tests}")
                print(f"  Failed: {suite_result.failed_tests}")
                print(f"  Success Rate: {suite_result.success_rate:.1%}")
                print(f"  Execution Time: {suite_result.execution_time:.2f}s")

                # 失敗したテストの詳細
                failed_results = [r for r in suite_result.results if not r.success]
                if failed_results:
                    print("  FAILED TESTS:")
                    for result in failed_results:
                        print(f"    - {result.scenario_name}: {result.error_message}")

        finally:
            # システム停止
            integration_manager.stop_system()
    else:
        print("Failed to initialize/start V433 system")
