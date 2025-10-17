"""
A/B Test Execution Engine
処理時間短縮・メモリ効率を考慮したストリーミング処理実装
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import gc
import psutil
from .types import (
    ABTestConfiguration, ABTestState, ABTestStatus, ABTestResult,
    SampleData, ABTestResultSummary, StatisticalResult,
    SampleProcessorCallback, TestCompletionCallback, RiskAlertCallback
)
from .analyzer import ABTestAnalyzer
from .config import ABTestConfig

logger = logging.getLogger(__name__)


class ABTestExecutor:
    """A/Bテスト実行エンジン（メモリ効率・処理時間最適化）"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.active_tests: Dict[str, ABTestState] = {}
        self.test_callbacks: Dict[str, List[Callable]] = {}

        # パフォーマンス最適化
        self.executor = ThreadPoolExecutor(max_workers=config.performance.max_workers)
        self.sample_queues: Dict[str, queue.Queue] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}

        # メモリ管理
        self.memory_monitor_thread: Optional[threading.Thread] = None
        self._start_memory_monitor()

        # 統計分析器
        self.analyzer = ABTestAnalyzer(config)

        logger.info("ABTestExecutor initialized with performance optimizations")

    def start_test(self, test_config: ABTestConfiguration) -> str:
        """A/Bテストを開始（メモリ効率的なストリーミング処理）"""
        if len(self.active_tests) >= self.config.max_concurrent_tests:
            raise RuntimeError(f"Maximum concurrent tests ({self.config.max_concurrent_tests}) reached")

        # テスト状態の初期化
        test_state = ABTestState(
            test_id=test_config.test_id,
            status=ABTestStatus.CREATED
        )

        self.active_tests[test_config.test_id] = test_state
        self.sample_queues[test_config.test_id] = queue.Queue(
            maxsize=self.config.performance.stream_buffer_size
        )

        # 処理スレッドの開始
        processing_thread = threading.Thread(
            target=self._process_test_stream,
            args=(test_config, test_state),
            daemon=True
        )
        processing_thread.start()
        self.processing_threads[test_config.test_id] = processing_thread

        # テスト開始
        test_state.status = ABTestStatus.RUNNING
        test_state.start_time = datetime.now()

        logger.info(f"Started A/B test: {test_config.test_id}")
        return test_config.test_id

    def add_sample(self, test_id: str, sample: SampleData) -> bool:
        """サンプルを追加（非同期・メモリ効率的なキューイング）"""
        if test_id not in self.sample_queues:
            logger.warning(f"Test {test_id} not found")
            return False

        try:
            # タイムアウト付きでキューに追加（ブロックしない）
            self.sample_queues[test_id].put(sample, timeout=0.1)
            return True
        except queue.Full:
            logger.warning(f"Sample queue full for test {test_id}")
            return False

    def add_samples_batch(self, test_id: str, samples: List[SampleData]) -> int:
        """サンプルをバッチで追加（処理時間短縮）"""
        if test_id not in self.sample_queues:
            logger.warning(f"Test {test_id} not found")
            return 0

        added_count = 0
        for sample in samples:
            if self.add_sample(test_id, sample):
                added_count += 1
            else:
                break  # キューが満杯になったら停止

        return added_count

    def stop_test(self, test_id: str) -> Optional[ABTestResultSummary]:
        """テストを停止"""
        if test_id not in self.active_tests:
            return None

        test_state = self.active_tests[test_id]
        test_state.status = ABTestStatus.COMPLETED
        test_state.end_time = datetime.now()

        # 結果の生成
        result_summary = self._generate_test_result(test_id)

        # クリーンアップ
        self._cleanup_test(test_id)

        logger.info(f"Stopped A/B test: {test_id}")
        return result_summary

    def get_test_status(self, test_id: str) -> Optional[ABTestState]:
        """テスト状態を取得"""
        return self.active_tests.get(test_id)

    def list_active_tests(self) -> List[str]:
        """アクティブなテストIDを取得"""
        return list(self.active_tests.keys())

    def _process_test_stream(self, test_config: ABTestConfiguration, test_state: ABTestState):
        """テストストリームを処理（メモリ効率的なメインループ）"""
        sample_queue = self.sample_queues[test_config.test_id]
        batch_buffer = []
        last_analysis_time = datetime.now()
        last_risk_check_time = datetime.now()

        try:
            while test_state.status == ABTestStatus.RUNNING:
                # サンプルをバッチで収集（処理時間短縮）
                batch_buffer = self._collect_sample_batch(sample_queue, batch_buffer)

                if batch_buffer:
                    # バッチ処理を実行
                    self._process_sample_batch(test_config, test_state, batch_buffer)
                    batch_buffer.clear()

                    # 定期的な分析実行
                    if (datetime.now() - last_analysis_time).seconds >= test_config.check_interval_minutes * 60:
                        self._perform_periodic_analysis(test_config, test_state)
                        last_analysis_time = datetime.now()

                    # リスクチェック
                    if (datetime.now() - last_risk_check_time).seconds >= 60:  # 1分ごと
                        self._check_risks(test_config, test_state)
                        last_risk_check_time = datetime.now()

                    # 完了条件チェック
                    if self._check_completion_conditions(test_config, test_state):
                        test_state.status = ABTestStatus.COMPLETED
                        break

                else:
                    # サンプルがない場合は少し待機
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Test processing failed for {test_config.test_id}: {e}")
            test_state.status = ABTestStatus.FAILED

        finally:
            # 最終分析を実行
            if test_state.status == ABTestStatus.COMPLETED:
                self._perform_final_analysis(test_config, test_state)

    def _collect_sample_batch(
        self,
        sample_queue: queue.Queue,
        buffer: List[SampleData]
    ) -> List[SampleData]:
        """サンプルをバッチで収集（メモリ効率的）"""
        batch_size = self.config.performance.batch_size

        # バッファが十分なサイズになるまで収集
        while len(buffer) < batch_size:
            try:
                sample = sample_queue.get(timeout=0.01)
                buffer.append(sample)
            except queue.Empty:
                break

        return buffer

    def _process_sample_batch(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState,
        batch: List[SampleData]
    ):
        """サンプルバッチを処理（並列処理対応）"""
        start_time = time.time()

        # バリアントごとにグループ化
        variant_samples = {"A": [], "B": []}
        for sample in batch:
            if sample.variant_id in variant_samples:
                variant_samples[sample.variant_id].append(sample)

        # 並列処理でメトリクス更新
        futures = []
        for variant_id, samples in variant_samples.items():
            if samples:
                future = self.executor.submit(
                    self._update_metrics_batch,
                    test_state,
                    variant_id,
                    samples
                )
                futures.append(future)

        # 結果を待機
        for future in as_completed(futures):
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

        # パフォーマンス監視
        processing_time = (time.time() - start_time) * 1000
        test_state.processing_time_ms += processing_time

        # サンプルカウント更新
        test_state.current_sample_count += len(batch)

    def _update_metrics_batch(
        self,
        test_state: ABTestState,
        variant_id: str,
        samples: List[SampleData]
    ):
        """メトリクスをバッチ更新（メモリ効率的）"""
        metrics = test_state.metrics_a if variant_id == "A" else test_state.metrics_b

        for sample in samples:
            metrics.add_sample(sample)

    def _perform_periodic_analysis(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState
    ):
        """定期的な統計分析を実行"""
        try:
            # 統計分析を実行
            statistical_result = self.analyzer.analyze_comparison(
                test_state.metrics_a,
                test_state.metrics_b,
                test_config.statistical_test
            )

            test_state.latest_statistical_result = statistical_result

            # 早期停止チェック
            if self.config.performance.enable_early_stopping:
                if self._check_early_stopping(test_config, test_state):
                    test_state.early_stop_triggered = True
                    test_state.status = ABTestStatus.COMPLETED

        except Exception as e:
            logger.error(f"Periodic analysis failed: {e}")

    def _check_early_stopping(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState
    ) -> bool:
        """早期停止条件をチェック"""
        if not test_state.latest_statistical_result:
            return False

        result = test_state.latest_statistical_result

        # サンプルサイズが最小値を超え、有意差が明確な場合
        min_samples = test_config.minimum_sample_size
        if (test_state.metrics_a.sample_count >= min_samples and
            test_state.metrics_b.sample_count >= min_samples):

            # p値が非常に小さく、効果量が十分な場合
            if result.p_value < 0.01 and result.effect_size > test_config.minimum_effect_size:
                return True

        return False

    def _check_risks(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState
    ):
        """リスクをチェック"""
        # 回帰検知
        if self._detect_regression(test_state):
            test_state.regression_detected = True
            logger.warning(f"Regression detected in test {test_config.test_id}")

            # 自動ロールバック設定の場合
            if self.config.risk.enable_automatic_rollback:
                test_state.status = ABTestStatus.CANCELLED

    def _detect_regression(self, test_state: ABTestState) -> bool:
        """回帰を検知"""
        # 簡易的な回帰検知（実際にはより複雑なロジック）
        if test_state.metrics_a.sample_count < 100 or test_state.metrics_b.sample_count < 100:
            return False

        # Bバリアントのパフォーマンスが大幅に低下した場合
        perf_a = test_state.metrics_a.get_rmse()
        perf_b = test_state.metrics_b.get_rmse()

        if perf_b > perf_a * (1 + self.config.risk.max_regression_rate):
            return True

        return False

    def _check_completion_conditions(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState
    ) -> bool:
        """完了条件をチェック"""
        # 最大サンプルサイズ到達
        if test_state.current_sample_count >= test_config.maximum_sample_size:
            return True

        # 最大時間経過
        if (test_state.start_time and
            datetime.now() - test_state.start_time > timedelta(hours=test_config.max_duration_hours)):
            return True

        # 統計的有意性到達
        if (test_state.latest_statistical_result and
            test_state.latest_statistical_result.p_value < test_config.confidence_level):
            return True

        return False

    def _perform_final_analysis(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState
    ):
        """最終分析を実行"""
        try:
            # 最終統計分析
            final_result = self.analyzer.analyze_comparison(
                test_state.metrics_a,
                test_state.metrics_b,
                test_config.statistical_test
            )

            test_state.latest_statistical_result = final_result

        except Exception as e:
            logger.error(f"Final analysis failed: {e}")

    def _generate_test_result(self, test_id: str) -> Optional[ABTestResultSummary]:
        """テスト結果を生成"""
        test_state = self.active_tests.get(test_id)
        if not test_state or not test_state.latest_statistical_result:
            return None

        result = test_state.latest_statistical_result

        # 勝者を決定
        if result.p_value < (self.config.statistics.alpha if self.config.statistics else 0.05):
            if result.mean_a < result.mean_b:  # RMSEが小さい方が良い
                winner = "A"
                ab_result = ABTestResult.WINNER_A
            else:
                winner = "B"
                ab_result = ABTestResult.WINNER_B
        else:
            winner = None
            ab_result = ABTestResult.INCONCLUSIVE

        return ABTestResultSummary(
            test_id=test_id,
            result=ab_result,
            winner_variant_id=winner,
            confidence_level=result.confidence_interval[1] - result.confidence_interval[0],
            statistical_result=result,
            risk_assessment=self._assess_risks(test_state),
            recommendations=self._generate_recommendations(test_state, ab_result)
        )

    def _assess_risks(self, test_state: ABTestState) -> Dict[str, Any]:
        """リスクを評価"""
        return {
            "regression_detected": test_state.regression_detected,
            "early_stop_triggered": test_state.early_stop_triggered,
            "sample_balance": abs(test_state.metrics_a.sample_count - test_state.metrics_b.sample_count),
            "processing_efficiency": test_state.processing_time_ms / max(test_state.current_sample_count, 1)
        }

    def _generate_recommendations(
        self,
        test_state: ABTestState,
        result: ABTestResult
    ) -> List[str]:
        """推奨事項を生成"""
        recommendations = []

        if result == ABTestResult.WINNER_A:
            recommendations.append("Deploy variant A - shows statistically significant improvement")
        elif result == ABTestResult.WINNER_B:
            recommendations.append("Deploy variant B - shows statistically significant improvement")
        else:
            recommendations.append("Continue testing - no clear winner yet")

        if test_state.regression_detected:
            recommendations.append("WARNING: Regression detected - monitor closely after deployment")

        if test_state.early_stop_triggered:
            recommendations.append("Test stopped early due to clear results")

        return recommendations

    def _cleanup_test(self, test_id: str):
        """テストのクリーンアップ"""
        if test_id in self.active_tests:
            del self.active_tests[test_id]

        if test_id in self.sample_queues:
            del self.sample_queues[test_id]

        if test_id in self.processing_threads:
            del self.processing_threads[test_id]

        # メモリクリーンアップ
        gc.collect()

    def _start_memory_monitor(self):
        """メモリ監視を開始"""
        if self.config.performance.max_memory_mb > 0:
            self.memory_monitor_thread = threading.Thread(
                target=self._monitor_memory_usage,
                daemon=True
            )
            self.memory_monitor_thread.start()

    def _monitor_memory_usage(self):
        """メモリ使用量を監視"""
        while True:
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

                if memory_mb > self.config.performance.max_memory_mb:
                    logger.warning(f"Memory usage high: {memory_mb:.1f}MB")
                    gc.collect()  # 強制ガベージコレクション

                time.sleep(self.config.performance.cleanup_interval_seconds)

            except Exception as e:
                logger.error(f"Memory monitoring failed: {e}")
                time.sleep(60)

    def __del__(self):
        """クリーンアップ"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

        if hasattr(self, 'memory_monitor_thread') and self.memory_monitor_thread:
            self.memory_monitor_thread.join(timeout=1)