"""
Safety Mechanisms and Fallback System
安全メカニズムとフォールバックシステム
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np

from .types import (
    AnomalyDetection,
    AnomalyType,
    FallbackAction,
    FallbackType,
    MetricType,
    MetricValue,
    RecoveryPlan,
    SafetyCheck,
    SafetyLevel,
    SafetyStatus,
)


@dataclass
class SafetyConfig:
    """安全設定"""

    # 異常検知設定
    anomaly_detection_enabled: bool = True
    statistical_anomaly_threshold: float = 3.0  # 標準偏差の倍数
    performance_anomaly_threshold: float = 0.2  # パフォーマンス変化率
    anomaly_detection_window_minutes: int = 60

    # 安全チェック設定
    safety_check_interval_seconds: int = 30
    critical_safety_threshold: float = 0.7  # システムヘルススコアの閾値
    emergency_shutdown_threshold: float = 0.3

    # フォールバック設定
    max_concurrent_fallbacks: int = 3
    fallback_timeout_seconds: int = 300
    gradual_rollback_steps: int = 5
    conservative_mode_duration_hours: int = 24

    # 回復設定
    auto_recovery_enabled: bool = True
    recovery_attempt_limit: int = 3
    recovery_cooldown_minutes: int = 15
    recovery_success_window_minutes: int = 30

    # 安全レベル閾値
    warning_health_threshold: float = 0.8
    critical_health_threshold: float = 0.6
    emergency_health_threshold: float = 0.4

    def __post_init__(self):
        """設定検証"""
        if not (0.0 <= self.critical_safety_threshold <= 1.0):
            raise ValueError("critical_safety_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.emergency_shutdown_threshold <= 1.0):
            raise ValueError("emergency_shutdown_threshold must be between 0.0 and 1.0")
        if self.emergency_shutdown_threshold >= self.critical_safety_threshold:
            raise ValueError(
                "emergency_shutdown_threshold must be less than critical_safety_threshold"
            )


logger = logging.getLogger(__name__)


class AnomalyDetector:
    """異常検知器"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def detect_anomalies(
        self, metrics: Dict[str, MetricValue]
    ) -> List[AnomalyDetection]:
        """異常検知"""
        anomalies = []

        for metric_name, metric_value in metrics.items():
            # メトリクス履歴を更新
            self.metric_history[metric_name].append(metric_value.value)

            # ベースライン統計を計算または更新
            if len(self.metric_history[metric_name]) >= 50:  # 最低50サンプル必要
                self._update_baseline_stats(metric_name)

                # 異常検知を実行
                anomaly = self._detect_single_anomaly(metric_name, metric_value.value)
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _update_baseline_stats(self, metric_name: str) -> None:
        """ベースライン統計更新"""
        values = list(self.metric_history[metric_name])
        self.baseline_stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    def _detect_single_anomaly(
        self, metric_name: str, value: float
    ) -> Optional[AnomalyDetection]:
        """単一メトリクスの異常検知"""
        if metric_name not in self.baseline_stats:
            return None

        stats = self.baseline_stats[metric_name]
        mean, std = stats["mean"], stats["std"]

        # Z-scoreベースの異常検知
        if std > 0:
            z_score = abs(value - mean) / std
            if z_score > self.config.statistical_anomaly_threshold:
                return AnomalyDetection(
                    anomaly_type=AnomalyType.STATISTICAL,
                    metric_name=metric_name,
                    detected_value=value,
                    expected_range=(mean - 2 * std, mean + 2 * std),
                    confidence=min(
                        z_score / self.config.statistical_anomaly_threshold, 1.0
                    ),
                    timestamp=datetime.now(),
                    context={
                        "z_score": z_score,
                        "baseline_mean": mean,
                        "baseline_std": std,
                    },
                )

        return None


class SafetyChecker:
    """安全チェック実行器"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.check_functions: Dict[str, Callable] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """デフォルトの安全チェックを登録"""
        self.check_functions = {
            "system_health": self._check_system_health,
            "performance_stability": self._check_performance_stability,
            "resource_usage": self._check_resource_usage,
            "error_rates": self._check_error_rates,
            "market_conditions": self._check_market_conditions,
        }

    def perform_safety_checks(
        self, metrics: Dict[str, MetricValue], anomalies: List[AnomalyDetection]
    ) -> List[SafetyCheck]:
        """安全チェック実行"""
        checks = []

        for check_name, check_func in self.check_functions.items():
            try:
                check = check_func(metrics, anomalies)
                checks.append(check)
            except Exception as e:
                logger.error(f"Safety check {check_name} failed: {e}")
                checks.append(
                    SafetyCheck(
                        check_name=check_name,
                        safety_level=SafetyLevel.CRITICAL,
                        passed=False,
                        message=f"Check execution failed: {str(e)}",
                        timestamp=datetime.now(),
                        details={"error": str(e)},
                    )
                )

        return checks

    def _check_system_health(
        self, metrics: Dict[str, MetricValue], anomalies: List[AnomalyDetection]
    ) -> SafetyCheck:
        """システムヘルスチェック"""
        # CPU使用率チェック
        cpu_metric = metrics.get("cpu_usage_percent")
        if cpu_metric and cpu_metric.value > 90:
            return SafetyCheck(
                check_name="system_health",
                safety_level=SafetyLevel.CRITICAL,
                passed=False,
                message=f"High CPU usage: {cpu_metric.value:.1f}%",
                timestamp=datetime.now(),
                details={"cpu_usage": cpu_metric.value},
            )

        # メモリ使用率チェック
        memory_metric = metrics.get("memory_usage_percent")
        if memory_metric and memory_metric.value > 85:
            return SafetyCheck(
                check_name="system_health",
                safety_level=SafetyLevel.WARNING,
                passed=False,
                message=f"High memory usage: {memory_metric.value:.1f}%",
                timestamp=datetime.now(),
                details={"memory_usage": memory_metric.value},
            )

        return SafetyCheck(
            check_name="system_health",
            safety_level=SafetyLevel.NORMAL,
            passed=True,
            message="System health is normal",
            timestamp=datetime.now(),
            details={
                "cpu_usage": cpu_metric.value if cpu_metric else None,
                "memory_usage": memory_metric.value if memory_metric else None,
            },
        )

    def _check_performance_stability(
        self, metrics: Dict[str, MetricValue], anomalies: List[AnomalyDetection]
    ) -> SafetyCheck:
        """パフォーマンス安定性チェック"""
        # 勝率チェック
        win_rate = metrics.get("win_rate")
        if win_rate and win_rate.value < 0.4:
            return SafetyCheck(
                check_name="performance_stability",
                safety_level=SafetyLevel.CRITICAL,
                passed=False,
                message=f"Low win rate: {win_rate.value:.3f}",
                timestamp=datetime.now(),
                details={"win_rate": win_rate.value},
            )

        # 最大ドローダウンチェック
        max_drawdown = metrics.get("max_drawdown")
        if max_drawdown and max_drawdown.value > 0.2:
            return SafetyCheck(
                check_name="performance_stability",
                safety_level=SafetyLevel.WARNING,
                passed=False,
                message=f"High drawdown: {max_drawdown.value:.3f}",
                timestamp=datetime.now(),
                details={"max_drawdown": max_drawdown.value},
            )

        return SafetyCheck(
            check_name="performance_stability",
            safety_level=SafetyLevel.NORMAL,
            passed=True,
            message="Performance is stable",
            timestamp=datetime.now(),
            details={
                "win_rate": win_rate.value if win_rate else None,
                "max_drawdown": max_drawdown.value if max_drawdown else None,
            },
        )

    def _check_resource_usage(
        self, metrics: Dict[str, MetricValue], anomalies: List[AnomalyDetection]
    ) -> SafetyCheck:
        """リソース使用チェック"""
        # 基本的なリソースチェックはシステムヘルスチェックでカバー
        return SafetyCheck(
            check_name="resource_usage",
            safety_level=SafetyLevel.NORMAL,
            passed=True,
            message="Resource usage is normal",
            timestamp=datetime.now(),
            details={},
        )

    def _check_error_rates(
        self, metrics: Dict[str, MetricValue], anomalies: List[AnomalyDetection]
    ) -> SafetyCheck:
        """エラーレートチェック"""
        error_rate = metrics.get("error_rate")
        if error_rate and error_rate.value > 0.1:
            return SafetyCheck(
                check_name="error_rates",
                safety_level=SafetyLevel.CRITICAL,
                passed=False,
                message=f"High error rate: {error_rate.value:.3f}",
                timestamp=datetime.now(),
                details={"error_rate": error_rate.value},
            )

        return SafetyCheck(
            check_name="error_rates",
            safety_level=SafetyLevel.NORMAL,
            passed=True,
            message="Error rates are normal",
            timestamp=datetime.now(),
            details={"error_rate": error_rate.value if error_rate else None},
        )

    def _check_market_conditions(
        self, metrics: Dict[str, MetricValue], anomalies: List[AnomalyDetection]
    ) -> SafetyCheck:
        """市場状況チェック"""
        # ボラティリティチェック
        volatility = metrics.get("market_volatility")
        if volatility and volatility.value > 0.5:
            return SafetyCheck(
                check_name="market_conditions",
                safety_level=SafetyLevel.WARNING,
                passed=False,
                message=f"High market volatility: {volatility.value:.3f}",
                timestamp=datetime.now(),
                details={"volatility": volatility.value},
            )

        return SafetyCheck(
            check_name="market_conditions",
            safety_level=SafetyLevel.NORMAL,
            passed=True,
            message="Market conditions are normal",
            timestamp=datetime.now(),
            details={"volatility": volatility.value if volatility else None},
        )


class FallbackHandler:
    """フォールバックハンドラー"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.active_fallbacks: Dict[str, FallbackAction] = {}
        self.fallback_functions: Dict[FallbackType, Callable] = {
            FallbackType.GRADUAL: self._execute_gradual_rollback,
            FallbackType.IMMEDIATE: self._execute_immediate_rollback,
            FallbackType.CONSERVATIVE: self._execute_conservative_mode,
            FallbackType.SHUTDOWN: self._execute_emergency_shutdown,
        }

    def initiate_fallback(
        self, fallback_type: FallbackType, reason: str, priority: int = 1
    ) -> str:
        """フォールバック開始"""
        action_id = f"fallback_{int(time.time())}_{fallback_type.value}"

        action = FallbackAction(
            action_id=action_id,
            fallback_type=fallback_type,
            description=f"{fallback_type.value} fallback initiated: {reason}",
            priority=priority,
            estimated_duration_seconds=self._get_estimated_duration(fallback_type),
            rollback_steps=self._get_rollback_steps(fallback_type),
            recovery_steps=self._get_recovery_steps(fallback_type),
        )

        self.active_fallbacks[action_id] = action

        # フォールバック実行
        try:
            self.fallback_functions[fallback_type](action)
            logger.info(f"Fallback {action_id} initiated successfully")
        except Exception as e:
            logger.error(f"Fallback {action_id} failed: {e}")
            action.description += f" (FAILED: {str(e)})"

        return action_id

    def _get_estimated_duration(self, fallback_type: FallbackType) -> int:
        """推定時間を取得"""
        durations = {
            FallbackType.GRADUAL: 300,  # 5分
            FallbackType.IMMEDIATE: 60,  # 1分
            FallbackType.CONSERVATIVE: 3600,  # 1時間
            FallbackType.SHUTDOWN: 30,  # 30秒
        }
        return durations.get(fallback_type, 60)

    def _get_rollback_steps(self, fallback_type: FallbackType) -> List[str]:
        """ロールバックステップを取得"""
        steps = {
            FallbackType.GRADUAL: [
                "Reduce trading frequency by 50%",
                "Disable high-risk strategies",
                "Enable conservative position sizing",
                "Monitor performance for 5 minutes",
                "Gradually restore normal operation",
            ],
            FallbackType.IMMEDIATE: [
                "Stop all active trades",
                "Switch to cash-only mode",
                "Disable automated trading",
                "Wait for manual intervention",
            ],
            FallbackType.CONSERVATIVE: [
                "Enable conservative trading mode",
                "Reduce position sizes by 75%",
                "Disable leverage",
                "Enable strict risk limits",
            ],
            FallbackType.SHUTDOWN: [
                "Stop all trading activities",
                "Close all positions",
                "Shutdown trading system",
                "Require manual restart",
            ],
        }
        return steps.get(fallback_type, ["Execute fallback procedure"])

    def _get_recovery_steps(self, fallback_type: FallbackType) -> List[str]:
        """回復ステップを取得"""
        steps = {
            FallbackType.GRADUAL: [
                "Verify system stability",
                "Gradually increase trading frequency",
                "Re-enable strategies one by one",
                "Monitor performance metrics",
            ],
            FallbackType.IMMEDIATE: [
                "Perform system diagnostics",
                "Manually verify market conditions",
                "Gradually resume trading",
                "Monitor for anomalies",
            ],
            FallbackType.CONSERVATIVE: [
                "Verify conservative mode effectiveness",
                "Gradually increase position sizes",
                "Re-enable advanced features",
                "Monitor risk metrics",
            ],
            FallbackType.SHUTDOWN: [
                "Perform complete system check",
                "Verify all components",
                "Manual restart with monitoring",
                "Gradual return to normal operation",
            ],
        }
        return steps.get(fallback_type, ["Perform recovery procedure"])

    def _execute_gradual_rollback(self, action: FallbackAction) -> None:
        """段階的ロールバック実行"""
        logger.info("Executing gradual rollback...")
        # 実際の実装では取引システムと連携
        time.sleep(1)  # シミュレーション

    def _execute_immediate_rollback(self, action: FallbackAction) -> None:
        """即時ロールバック実行"""
        logger.warning("Executing immediate rollback!")
        # 実際の実装では取引システムと連携
        time.sleep(0.5)  # シミュレーション

    def _execute_conservative_mode(self, action: FallbackAction) -> None:
        """保守的モード実行"""
        logger.info("Switching to conservative mode...")
        # 実際の実装では取引システムと連携
        time.sleep(1)  # シミュレーション

    def _execute_emergency_shutdown(self, action: FallbackAction) -> None:
        """緊急シャットダウン実行"""
        logger.critical("Executing emergency shutdown!")
        # 実際の実装では取引システムと連携
        time.sleep(0.1)  # シミュレーション

    def get_active_fallbacks(self) -> List[FallbackAction]:
        """アクティブなフォールバックを取得"""
        return list(self.active_fallbacks.values())

    def cancel_fallback(self, action_id: str) -> bool:
        """フォールバックキャンセル"""
        if action_id in self.active_fallbacks:
            del self.active_fallbacks[action_id]
            logger.info(f"Fallback {action_id} cancelled")
            return True
        return False


class RecoveryManager:
    """回復マネージャー"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_attempts: Dict[str, int] = defaultdict(int)

    def create_recovery_plan(
        self, trigger_reason: str, steps: List[str], success_criteria: List[str]
    ) -> str:
        """回復計画作成"""
        plan_id = f"recovery_{int(time.time())}"

        plan = RecoveryPlan(
            plan_id=plan_id,
            triggered_by=trigger_reason,
            steps=steps,
            estimated_completion_time=datetime.now() + timedelta(minutes=30),
            success_criteria=success_criteria,
            rollback_plan=["Cancel recovery", "Return to safe state"],
        )

        self.recovery_plans[plan_id] = plan
        logger.info(f"Recovery plan {plan_id} created: {trigger_reason}")

        return plan_id

    def execute_recovery(self, plan_id: str) -> bool:
        """回復実行"""
        if plan_id not in self.recovery_plans:
            logger.error(f"Recovery plan {plan_id} not found")
            return False

        plan = self.recovery_plans[plan_id]

        # 試行回数チェック
        if (
            self.recovery_attempts[plan.triggered_by]
            >= self.config.recovery_attempt_limit
        ):
            logger.warning(f"Recovery attempt limit reached for {plan.triggered_by}")
            return False

        try:
            logger.info(f"Executing recovery plan {plan_id}")

            # 回復ステップ実行（シミュレーション）
            for step in plan.steps:
                logger.info(f"Recovery step: {step}")
                time.sleep(1)  # シミュレーション

            # 成功判定（実際の実装ではメトリクスで判定）
            success = True

            if success:
                logger.info(f"Recovery plan {plan_id} completed successfully")
                self.recovery_attempts[plan.triggered_by] = 0  # リセット
                return True
            else:
                logger.warning(f"Recovery plan {plan_id} failed")
                self.recovery_attempts[plan.triggered_by] += 1
                return False

        except Exception as e:
            logger.error(f"Recovery plan {plan_id} execution failed: {e}")
            self.recovery_attempts[plan.triggered_by] += 1
            return False


class SafetyManager:
    """安全マネージャー"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.safety_checker = SafetyChecker(config)
        self.fallback_handler = FallbackHandler(config)
        self.recovery_manager = RecoveryManager(config)

        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False

        # 安全ステータス
        self.current_status = SafetyStatus(
            overall_safety_level=SafetyLevel.NORMAL,
            active_anomalies=[],
            recent_checks=[],
            active_fallbacks=[],
            last_updated=datetime.now(),
            system_health_score=1.0,
        )

    def start_safety_monitoring(self) -> None:
        """安全監視開始"""
        if self.is_monitoring:
            logger.warning("Safety monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._safety_monitoring_worker, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Safety monitoring started")

    def stop_safety_monitoring(self) -> None:
        """安全監視停止"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Safety monitoring stopped")

    def _safety_monitoring_worker(self) -> None:
        """安全監視ワーカー"""
        while self.is_monitoring:
            try:
                # 安全チェック実行（実際の実装ではメトリクスを取得）
                self._perform_safety_assessment()

                time.sleep(self.config.safety_check_interval_seconds)

            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(5)  # エラー時は短い待機

    def _perform_safety_assessment(self) -> None:
        """安全評価実行"""
        # 実際の実装では最新メトリクスを取得
        # ここではシミュレーション
        mock_metrics = self._get_mock_metrics()

        # 異常検知
        anomalies = self.anomaly_detector.detect_anomalies(mock_metrics)

        # 安全チェック
        checks = self.safety_checker.perform_safety_checks(mock_metrics, anomalies)

        # 安全レベル判定
        safety_level = self._calculate_overall_safety_level(checks, anomalies)

        # ヘルススコア計算
        health_score = self._calculate_health_score(checks)

        # ステータス更新
        self.current_status = SafetyStatus(
            overall_safety_level=safety_level,
            active_anomalies=anomalies,
            recent_checks=checks[-10:],  # 最新10件
            active_fallbacks=self.fallback_handler.get_active_fallbacks(),
            last_updated=datetime.now(),
            system_health_score=health_score,
        )

        # 自動対応
        self._handle_automatic_actions(safety_level, anomalies)

    def _get_mock_metrics(self) -> Dict[str, MetricValue]:
        """モックメトリクス取得（実際の実装では監視システムから取得）"""
        return {
            "cpu_usage_percent": MetricValue(
                name="cpu_usage_percent",
                value=45.0 + np.random.normal(0, 5),
                timestamp=datetime.now(),
                metric_type=MetricType.SYSTEM,
            ),
            "memory_usage_percent": MetricValue(
                name="memory_usage_percent",
                value=60.0 + np.random.normal(0, 3),
                timestamp=datetime.now(),
                metric_type=MetricType.SYSTEM,
            ),
            "win_rate": MetricValue(
                name="win_rate",
                value=0.55 + np.random.normal(0, 0.05),
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE,
            ),
        }

    def _calculate_overall_safety_level(
        self, checks: List[SafetyCheck], anomalies: List[AnomalyDetection]
    ) -> SafetyLevel:
        """全体安全レベル計算"""
        # クリティカルチェックがある場合
        if any(
            check.safety_level == SafetyLevel.CRITICAL and not check.passed
            for check in checks
        ):
            return SafetyLevel.CRITICAL

        # 緊急レベルの異常がある場合
        if any(anomaly.confidence > 0.9 for anomaly in anomalies):
            return SafetyLevel.EMERGENCY

        # 警告チェックがある場合
        if any(
            check.safety_level == SafetyLevel.WARNING and not check.passed
            for check in checks
        ):
            return SafetyLevel.WARNING

        # 異常がある場合
        if anomalies:
            return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _calculate_health_score(self, checks: List[SafetyCheck]) -> float:
        """ヘルススコア計算"""
        if not checks:
            return 1.0

        total_score = 0.0
        for check in checks:
            if check.passed:
                score = 1.0
            elif check.safety_level == SafetyLevel.WARNING:
                score = 0.7
            elif check.safety_level == SafetyLevel.CRITICAL:
                score = 0.3
            else:  # EMERGENCY
                score = 0.0
            total_score += score

        return total_score / len(checks)

    def _handle_automatic_actions(
        self, safety_level: SafetyLevel, anomalies: List[AnomalyDetection]
    ) -> None:
        """自動対応処理"""
        # 緊急レベルの場合
        if safety_level == SafetyLevel.EMERGENCY:
            self.fallback_handler.initiate_fallback(
                FallbackType.SHUTDOWN, "Emergency safety level detected", priority=10
            )

        # クリティカルレベルの場合
        elif safety_level == SafetyLevel.CRITICAL:
            if (
                self.current_status.system_health_score
                < self.config.emergency_shutdown_threshold
            ):
                self.fallback_handler.initiate_fallback(
                    FallbackType.IMMEDIATE,
                    "Critical health score threshold breached",
                    priority=8,
                )
            else:
                self.fallback_handler.initiate_fallback(
                    FallbackType.GRADUAL, "Critical safety level detected", priority=6
                )

        # 警告レベルの場合
        elif safety_level == SafetyLevel.WARNING:
            if len(anomalies) > 3:  # 複数の異常
                self.fallback_handler.initiate_fallback(
                    FallbackType.CONSERVATIVE, "Multiple anomalies detected", priority=4
                )

    def get_safety_status(self) -> SafetyStatus:
        """安全ステータス取得"""
        return self.current_status

    def initiate_manual_fallback(self, fallback_type: FallbackType, reason: str) -> str:
        """手動フォールバック開始"""
        return self.fallback_handler.initiate_fallback(
            fallback_type, reason, priority=5
        )

    def create_recovery_plan(
        self, trigger_reason: str, steps: List[str], success_criteria: List[str]
    ) -> str:
        """回復計画作成"""
        return self.recovery_manager.create_recovery_plan(
            trigger_reason, steps, success_criteria
        )

    def execute_recovery(self, plan_id: str) -> bool:
        """回復実行"""
        return self.recovery_manager.execute_recovery(plan_id)
