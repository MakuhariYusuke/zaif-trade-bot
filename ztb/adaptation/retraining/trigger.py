"""
Automatic Retraining Trigger System
パフォーマンス監視と自動再訓練トリガー
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import threading
import weakref
from collections import deque
import gc

from .config import RetrainingConfig
from .types import (
    TriggerType, TriggerPriority, TriggerStatus, TriggerCondition,
    PerformanceMetrics, DataDistributionMetrics, RetrainingRequest,
    RetrainingResult, TriggerState, RetrainingSchedule, ResourceUsage,
    RetrainingHistory
)

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """自動再訓練トリガーシステム"""

    def __init__(self, config: Optional[RetrainingConfig] = None):
        self.config = config or RetrainingConfig()
        self.trigger_states: Dict[str, TriggerState] = {}
        self.performance_history: deque = deque(maxlen=self.config.max_history_size)
        self.distribution_history: deque = deque(maxlen=self.config.max_history_size)
        self.retraining_history: deque = deque(maxlen=self.config.max_history_size)
        self.schedules: List[RetrainingSchedule] = []
        self.active_requests: Dict[str, RetrainingRequest] = {}

        # メモリ管理用の弱参照
        self._metric_callbacks: weakref.WeakSet = weakref.WeakSet()

        # クリーンアップタイマー
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()

        # トリガー状態の初期化
        self._initialize_triggers()

        logger.info("RetrainingTrigger initialized")

    def __del__(self) -> None:
        """デストラクタ：リソースのクリーンアップ"""
        self._stop_cleanup_timer()

    def _initialize_triggers(self) -> None:
        """トリガーを初期化"""
        for condition in self.config.trigger_conditions:
            trigger_id = f"{condition.trigger_type.value}_{condition.metric_name}"
            self.trigger_states[trigger_id] = TriggerState(
                trigger_id=trigger_id,
                condition=condition,
                status=TriggerStatus.MONITORING,
                last_check=datetime.now(),
                last_triggered=None,
                consecutive_violations=0,
                cooldown_until=None
            )

        # スケジュールされたトリガーの初期化
        self._initialize_schedules()

    def _initialize_schedules(self) -> None:
        """スケジュールされたトリガーを初期化"""
        for schedule_config in self.config.time_based_schedules:
            schedule = RetrainingSchedule(
                schedule_id=f"time_based_{schedule_config['interval_hours']}h",
                trigger_type=TriggerType.TIME_BASED,
                cron_expression=None,
                interval_minutes=schedule_config['interval_hours'] * 60,
                next_run=datetime.now() + timedelta(minutes=schedule_config['interval_hours'] * 60)
            )
            self.schedules.append(schedule)

    def update_performance_metrics(self, metrics: PerformanceMetrics) -> List[RetrainingRequest]:
        """パフォーマンス指標を更新"""
        self.performance_history.append(metrics)

        # トリガーチェック
        triggered_requests = self._check_performance_triggers(metrics)

        # 履歴サイズの制限（メモリリーク防止）
        self._cleanup_history_if_needed()

        return triggered_requests

    def update_distribution_metrics(self, metrics: DataDistributionMetrics) -> List[RetrainingRequest]:
        """データ分布指標を更新"""
        self.distribution_history.append(metrics)

        # トリガーチェック
        triggered_requests = self._check_distribution_triggers(metrics)

        # 履歴サイズの制限
        self._cleanup_history_if_needed()

        return triggered_requests

    def check_scheduled_triggers(self) -> List[RetrainingRequest]:
        """スケジュールされたトリガーをチェック"""
        now = datetime.now()
        triggered_requests = []

        for schedule in self.schedules:
            if not schedule.enabled:
                continue

            if schedule.next_run <= now:
                request = RetrainingRequest(
                    request_id=f"scheduled_{schedule.schedule_id}_{now.isoformat()}",
                    trigger_type=schedule.trigger_type,
                    trigger_reason=f"Scheduled retraining: {schedule.schedule_id}",
                    priority=TriggerPriority.MEDIUM,
                    requested_at=now,
                    estimated_duration=timedelta(hours=2),
                    required_resources={"cpu": 2, "memory_gb": 4}
                )

                triggered_requests.append(request)

                # 次回の実行時間を設定
                if schedule.interval_minutes:
                    schedule.next_run = now + timedelta(minutes=schedule.interval_minutes)
                schedule.last_run = now
                schedule.run_count += 1

        return triggered_requests

    def check_volume_based_triggers(self, new_samples_count: int) -> List[RetrainingRequest]:
        """出来高ベースのトリガーをチェック"""
        min_samples = self.config.volume_based_thresholds["min_new_samples"]

        if new_samples_count >= min_samples:
            now = datetime.now()
            request = RetrainingRequest(
                request_id=f"volume_based_{now.isoformat()}",
                trigger_type=TriggerType.VOLUME_BASED,
                trigger_reason=f"New samples threshold reached: {new_samples_count}",
                priority=TriggerPriority.MEDIUM,
                requested_at=now,
                estimated_duration=timedelta(hours=1),
                required_resources={"cpu": 1, "memory_gb": 2}
            )
            return [request]

        return []

    def _check_performance_triggers(self, metrics: PerformanceMetrics) -> List[RetrainingRequest]:
        """パフォーマンストリガーをチェック"""
        triggered_requests = []

        for trigger_id, state in self.trigger_states.items():
            if state.condition.trigger_type != TriggerType.PERFORMANCE:
                continue

            # クールダウンチェック
            if state.cooldown_until and datetime.now() < state.cooldown_until:
                continue

            # 条件チェック
            if self._check_performance_condition(state.condition, metrics):
                state.consecutive_violations += 1

                # 十分な期間条件を満たしているかチェック
                if self._check_duration_condition(state, metrics.timestamp):
                    request = self._create_retraining_request(state, metrics.timestamp)
                    triggered_requests.append(request)

                    # 状態更新
                    state.status = TriggerStatus.TRIGGERED
                    state.last_triggered = metrics.timestamp
                    state.cooldown_until = metrics.timestamp + timedelta(
                        minutes=state.condition.cooldown_minutes
                    )
                    state.consecutive_violations = 0
            else:
                state.consecutive_violations = 0

            state.last_check = metrics.timestamp

        return triggered_requests

    def _check_distribution_triggers(self, metrics: DataDistributionMetrics) -> List[RetrainingRequest]:
        """データ分布トリガーをチェック"""
        triggered_requests = []

        for trigger_id, state in self.trigger_states.items():
            if state.condition.trigger_type != TriggerType.DATA_DISTRIBUTION:
                continue

            # クールダウンチェック
            if state.cooldown_until and datetime.now() < state.cooldown_until:
                continue

            # 分布変化の計算
            drift_score = self._calculate_distribution_drift(metrics)

            if drift_score > state.condition.threshold:
                state.consecutive_violations += 1

                if self._check_duration_condition(state, metrics.timestamp):
                    request = self._create_retraining_request(state, metrics.timestamp,
                                                            f"Distribution drift: {drift_score:.3f}")
                    triggered_requests.append(request)

                    state.status = TriggerStatus.TRIGGERED
                    state.last_triggered = metrics.timestamp
                    state.cooldown_until = metrics.timestamp + timedelta(
                        minutes=state.condition.cooldown_minutes
                    )
                    state.consecutive_violations = 0
            else:
                state.consecutive_violations = 0

            state.last_check = metrics.timestamp

        return triggered_requests

    def _check_performance_condition(self, condition: TriggerCondition, metrics: PerformanceMetrics) -> bool:
        """パフォーマンス条件をチェック"""
        value = getattr(metrics, condition.metric_name, None)
        if value is None:
            return False

        return self._evaluate_condition(value, condition.operator, condition.threshold)

    def _calculate_distribution_drift(self, current_metrics: DataDistributionMetrics) -> float:
        """データ分布の変化を計算"""
        if len(self.distribution_history) < 2:
            return 0.0

        # 最新のベースラインメトリクスを取得
        baseline = self.distribution_history[-2]  # 1つ前のデータ

        # 特徴量ごとの変化を計算
        drift_scores = []

        for feature in current_metrics.feature_means.keys():
            if feature in baseline.feature_means:
                # 平均の変化
                mean_diff = abs(current_metrics.feature_means[feature] - baseline.feature_means[feature])

                # 標準偏差で正規化
                if baseline.feature_stds.get(feature, 1.0) > 0:
                    normalized_diff = mean_diff / baseline.feature_stds[feature]
                    drift_scores.append(normalized_diff)

        return np.mean(drift_scores) if drift_scores else 0.0

    def _check_duration_condition(self, state: TriggerState, current_time: datetime) -> bool:
        """期間条件をチェック"""
        if state.consecutive_violations == 0:
            return False

        # 簡易的な期間チェック（実際の実装ではより詳細なチェックが必要）
        return state.consecutive_violations >= (state.condition.duration_minutes // self.config.performance_check_interval_minutes)

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """条件を評価"""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return abs(value - threshold) < 1e-6
        elif operator == "ne":
            return abs(value - threshold) >= 1e-6
        return False

    def _create_retraining_request(self, state: TriggerState, timestamp: datetime,
                                 additional_reason: str = "") -> RetrainingRequest:
        """再訓練リクエストを作成"""
        reason = f"{state.condition.metric_name} {state.condition.operator} {state.condition.threshold}"
        if additional_reason:
            reason += f" ({additional_reason})"

        return RetrainingRequest(
            request_id=f"{state.trigger_id}_{timestamp.isoformat()}",
            trigger_type=state.condition.trigger_type,
            trigger_reason=reason,
            priority=state.condition.priority,
            requested_at=timestamp,
            estimated_duration=timedelta(hours=2),
            required_resources={"cpu": 1, "memory_gb": 2}
        )

    def record_retraining_result(self, result: RetrainingResult) -> None:
        """再訓練結果を記録"""
        history = RetrainingHistory(
            request_id=result.request_id,
            trigger_type=result.trigger_type if hasattr(result, 'trigger_type') else TriggerType.MANUAL,
            start_time=result.completed_at - result.training_duration,
            end_time=result.completed_at,
            success=result.success,
            performance_change=result.performance_improvement,
            resource_usage=None,  # 必要に応じて追加
            error_details=result.error_message
        )

        self.retraining_history.append(history)

        # アクティブリクエストから削除
        if result.request_id in self.active_requests:
            del self.active_requests[result.request_id]

        # 履歴サイズの制限
        self._cleanup_history_if_needed()

    def get_trigger_states(self) -> Dict[str, TriggerState]:
        """トリガー状態を取得"""
        return self.trigger_states.copy()

    def get_retraining_history(self, limit: int = 50) -> List[RetrainingHistory]:
        """再訓練履歴を取得"""
        return list(self.retraining_history)[-limit:]

    def reset_triggers(self) -> None:
        """トリガーをリセット"""
        for state in self.trigger_states.values():
            state.status = TriggerStatus.MONITORING
            state.consecutive_violations = 0
            state.cooldown_until = None

        self.active_requests.clear()

    def _cleanup_history_if_needed(self) -> None:
        """必要に応じて履歴をクリーンアップ"""
        # 設定された最大サイズを超えている場合のみクリーンアップ
        if len(self.performance_history) > self.config.max_history_size:
            # 古いデータを削除（デックなので自動的に管理されるが、明示的にクリーンアップ）
            excess = len(self.performance_history) - self.config.max_history_size
            for _ in range(excess):
                self.performance_history.popleft()

        if len(self.distribution_history) > self.config.max_history_size:
            excess = len(self.distribution_history) - self.config.max_history_size
            for _ in range(excess):
                self.distribution_history.popleft()

        if len(self.retraining_history) > self.config.max_history_size:
            excess = len(self.retraining_history) - self.config.max_history_size
            for _ in range(excess):
                self.retraining_history.popleft()

    def _start_cleanup_timer(self) -> None:
        """クリーンアップタイマーを開始"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        # 定期的なクリーンアップを実行
        self._cleanup_timer = threading.Timer(
            self.config.cleanup_interval_hours * 3600,
            self._periodic_cleanup
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _stop_cleanup_timer(self) -> None:
        """クリーンアップタイマーを停止"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    def _periodic_cleanup(self) -> None:
        """定期的なクリーンアップ"""
        try:
            # 明示的なガベージコレクション
            gc.collect()

            # 弱参照のクリーンアップ
            self._metric_callbacks.clear()

            # 履歴の圧縮（設定されている場合）
            if self.config.compression_enabled:
                self._compress_old_history()

            logger.info("Periodic cleanup completed")

        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")
        finally:
            # 次回のタイマーを設定
            self._start_cleanup_timer()

    def _compress_old_history(self) -> None:
        """古い履歴を圧縮"""
        # 古いデータをサンプリングして圧縮
        compression_ratio = 0.1  # 10%を保持

        if len(self.performance_history) > 100:
            compressed = []
            step = max(1, int(1 / compression_ratio))
            for i in range(0, len(self.performance_history), step):
                compressed.append(self.performance_history[i])
            self.performance_history.clear()
            self.performance_history.extend(compressed)