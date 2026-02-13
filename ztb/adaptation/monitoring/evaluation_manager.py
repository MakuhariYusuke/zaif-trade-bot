"""
Continuous Evaluation and Monitoring Manager
継続的評価と監視マネージャー
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

try:
    from ..concept_drift.manager import ConceptDriftManager
except ImportError:
    ConceptDriftManager = None  # type: ignore[misc,assignment]  # archived in 030#
from ..monitoring.monitor import PerformanceMonitor
from ..monitoring.safety import SafetyManager

if TYPE_CHECKING:
    # Import for type checking only; avoid importing heavy modules (e.g., torch) at module import time
    from ..online_learning.pipeline import OnlineLearningPipeline
    from ..explainability.analyzer import ExplainabilityAnalyzer

from .evaluation_types import (
    AlertLevel,
    AlertType,
    EvaluationMetrics,
    EvaluationResult,
    MonitoringAlert,
    SystemMetrics,
)

logger = logging.getLogger(__name__)


class ContinuousEvaluationManager:
    """継続的評価マネージャー"""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        safety_manager: SafetyManager,
        drift_manager: ConceptDriftManager,
        online_learning: Optional["OnlineLearningPipeline"] = None,
        explainability_analyzer: Optional["ExplainabilityAnalyzer"] = None,
    ):
        self.monitor = monitor
        self.safety_manager = safety_manager
        self.drift_manager = drift_manager
        self.online_learning = online_learning
        self.explainability_analyzer = explainability_analyzer

        self.is_running = False
        self.evaluation_interval_seconds = 60  # 1分間隔
        self.alert_check_interval_seconds = 30  # 30秒間隔

        self.evaluation_history: List[EvaluationResult] = []
        self.active_alerts: List[MonitoringAlert] = []
        self.system_metrics_history: List[SystemMetrics] = []

        self.alert_callbacks: List[Callable[[MonitoringAlert], None]] = []
        self.evaluation_callbacks: List[Callable[[EvaluationResult], None]] = []

        # スレッド管理
        self.threads: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=4)

    def start_continuous_evaluation(self) -> bool:
        """継続的評価を開始"""
        try:
            logger.info("Starting continuous evaluation...")

            self.is_running = True

            # 評価スレッドの起動
            evaluation_thread = threading.Thread(
                target=self._evaluation_worker, daemon=True
            )
            evaluation_thread.start()
            self.threads.append(evaluation_thread)

            # アラート監視スレッドの起動
            alert_thread = threading.Thread(
                target=self._alert_monitor_worker, daemon=True
            )
            alert_thread.start()
            self.threads.append(alert_thread)

            # システムメトリクス収集スレッドの起動
            metrics_thread = threading.Thread(
                target=self._metrics_collection_worker, daemon=True
            )
            metrics_thread.start()
            self.threads.append(metrics_thread)

            logger.info("Continuous evaluation started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start continuous evaluation: {e}")
            self.stop_continuous_evaluation()
            return False

    def stop_continuous_evaluation(self) -> None:
        """継続的評価を停止"""
        logger.info("Stopping continuous evaluation...")

        self.is_running = False

        # スレッドの停止待ち
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        self.threads.clear()
        self.executor.shutdown(wait=True)

        logger.info("Continuous evaluation stopped")

    def perform_evaluation(self) -> EvaluationResult:
        """評価を実行"""
        try:
            evaluation_start = datetime.now()

            # 並列で各種評価を実行
            futures = {}

            # パフォーマンス評価
            futures["performance"] = self.executor.submit(self._evaluate_performance)

            # 安全評価
            futures["safety"] = self.executor.submit(self._evaluate_safety)

            # ドリフト検知
            futures["drift"] = self.executor.submit(self._evaluate_drift)

            # オンライン学習評価（利用可能な場合）
            if self.online_learning:
                futures["online_learning"] = self.executor.submit(
                    self._evaluate_online_learning
                )

            # 結果の収集
            results = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Error in {key} evaluation: {e}")
                    results[key] = None

            # 統合評価結果の作成
            evaluation_result = EvaluationResult(
                timestamp=evaluation_start,
                performance_metrics=results.get("performance"),
                safety_metrics=results.get("safety"),
                drift_detected=results.get("drift", {}).get("drift_detected", False),
                drift_severity=results.get("drift", {}).get("severity"),
                online_learning_metrics=results.get("online_learning"),
                overall_score=self._calculate_overall_score(results),
                recommendations=self._generate_recommendations(results),
                processing_time_seconds=(
                    datetime.now() - evaluation_start
                ).total_seconds(),
            )

            # 履歴に追加
            self.evaluation_history.append(evaluation_result)

            # コールバックの実行
            self._trigger_evaluation_callbacks(evaluation_result)

            return evaluation_result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                timestamp=datetime.now(),
                performance_metrics=None,
                safety_metrics=None,
                drift_detected=False,
                processing_time_seconds=0.0,
                error=str(e),
            )

    def _evaluate_performance(self) -> Optional[EvaluationMetrics]:
        """パフォーマンス評価"""
        try:
            # 最新のメトリクスを取得
            latest_metrics = self.monitor.get_latest_metrics()

            # メトリクス値を抽出
            metrics_dict = {mv.name: mv.value for mv in latest_metrics.values()}

            return EvaluationMetrics(
                accuracy=metrics_dict.get("win_rate", 0.0),
                precision=metrics_dict.get("precision", 0.0),
                recall=metrics_dict.get("recall", 0.0),
                f1_score=metrics_dict.get("f1_score", 0.0),
                sharpe_ratio=metrics_dict.get("sharpe_ratio", 0.0),
                max_drawdown=metrics_dict.get("max_drawdown", 0.0),
                total_return=metrics_dict.get("total_return", 0.0),
                volatility=metrics_dict.get("volatility", 0.0),
            )

        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return None

    def _evaluate_safety(self) -> Optional[Dict[str, Any]]:
        """安全評価"""
        try:
            safety_status = self.safety_manager.get_safety_status()

            return {
                "overall_safety_level": safety_status.overall_safety_level.value,
                "active_anomalies": len(safety_status.active_anomalies),
                "recent_checks": len(safety_status.recent_checks),
                "safety_score": safety_status.system_health_score,
            }

        except Exception as e:
            logger.error(f"Safety evaluation failed: {e}")
            return None

    def _evaluate_drift(self) -> Dict[str, Any]:
        """ドリフト評価"""
        try:
            # 最新のデータを用いたドリフト検知
            recent_data = self._get_recent_data_for_drift_detection()

            if recent_data is not None:
                drift_results = self.drift_manager.detect_drift(recent_data)

                # 最も深刻なドリフトを特定
                max_severity = max(
                    (result.severity.value for result in drift_results), default=0
                )

                return {
                    "drift_detected": any(
                        result.drift_detected for result in drift_results
                    ),
                    "severity": max_severity,
                    "drift_types": [
                        result.drift_type.value
                        for result in drift_results
                        if result.drift_detected
                    ],
                }
            else:
                return {"drift_detected": False, "severity": 0, "drift_types": []}

        except Exception as e:
            logger.error(f"Drift evaluation failed: {e}")
            return {
                "drift_detected": False,
                "severity": 0,
                "drift_types": [],
                "error": str(e),
            }

    def _evaluate_online_learning(self) -> Optional[Dict[str, Any]]:
        """オンライン学習評価"""
        if not self.online_learning:
            return None

        try:
            learning_status = self.online_learning.get_status()

            return {
                "is_active": learning_status.get("active", False),
                "total_samples_processed": learning_status.get("total_samples", 0),
                "current_learning_rate": learning_status.get("learning_rate", 0.0),
                "loss_history": learning_status.get("loss_history", []),
                "gradient_norm": learning_status.get("gradient_norm", 0.0),
            }

        except Exception as e:
            logger.error(f"Online learning evaluation failed: {e}")
            return None

    def _get_recent_data_for_drift_detection(self) -> Optional[np.ndarray]:
        """ドリフト検知用の最近のデータを取得"""
        try:
            # モニターから最近の特徴量データを取得
            latest_metrics = self.monitor.get_latest_metrics()
            feature_data = []

            # 特徴量関連のメトリクスを収集
            for metric_value in latest_metrics.values():
                if (
                    "feature" in metric_value.name.lower()
                    or "input" in metric_value.name.lower()
                ):
                    feature_data.append(metric_value.value)

            if feature_data:
                return np.array(feature_data).reshape(1, -1)
            else:
                # ダミーデータを生成（テスト用）
                return np.random.randn(1, 10)

        except Exception as e:
            logger.error(f"Failed to get recent data for drift detection: {e}")
            return None

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """総合スコアの計算"""
        try:
            score = 0.0
            weight_sum = 0.0

            # パフォーマンススコア（重み: 0.4）
            if results.get("performance"):
                perf = results["performance"]
                perf_score = (
                    perf.accuracy * 0.3
                    + (1 - perf.max_drawdown) * 0.3
                    + (perf.sharpe_ratio / 3.0) * 0.4
                )  # Sharpe 3.0を満点とする
                score += perf_score * 0.4
                weight_sum += 0.4

            # 安全スコア（重み: 0.3）
            if results.get("safety"):
                safety = results["safety"]
                safety_score = safety.get("safety_score", 0.5)
                score += safety_score * 0.3
                weight_sum += 0.3

            # ドリフトペナルティ（重み: 0.3）
            drift_penalty = 0.0
            if results.get("drift", {}).get("drift_detected"):
                severity = results["drift"].get("severity", 1)
                drift_penalty = min(severity / 10.0, 1.0)  # 最大1.0のペナルティ

            score += (1.0 - drift_penalty) * 0.3
            weight_sum += 0.3

            return score / weight_sum if weight_sum > 0 else 0.5

        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            return 0.5

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []

        try:
            # パフォーマンスベースの推奨
            if results.get("performance"):
                perf = results["performance"]
                if perf.accuracy < 0.5:
                    recommendations.append("モデルの再学習を検討してください")
                if perf.max_drawdown > 0.2:
                    recommendations.append(
                        "リスク管理パラメータの見直しを検討してください"
                    )
                if perf.sharpe_ratio < 1.0:
                    recommendations.append("リターンの安定化を検討してください")

            # 安全ベースの推奨
            if results.get("safety"):
                safety = results["safety"]
                if safety.get("active_anomalies", 0) > 0:
                    recommendations.append("異常検知された問題の調査を推奨します")
                if safety.get("safety_score", 1.0) < 0.7:
                    recommendations.append(
                        "システムの安全性を向上させる検討を推奨します"
                    )

            # ドリフトベースの推奨
            if results.get("drift", {}).get("drift_detected"):
                severity = results["drift"].get("severity", 1)
                if severity >= 3:
                    recommendations.append(
                        "市場環境の大きな変化が検知されました。再学習を強く推奨します"
                    )
                else:
                    recommendations.append(
                        "市場環境の変化が検知されました。モデルの調整を検討してください"
                    )

            # デフォルト推奨
            if not recommendations:
                recommendations.append("システムは正常に動作しています")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations = ["評価中にエラーが発生しました"]

        return recommendations

    def _evaluation_worker(self) -> None:
        """評価ワーカー"""
        while self.is_running:
            try:
                self.perform_evaluation()
                time.sleep(self.evaluation_interval_seconds)

            except Exception as e:
                logger.error(f"Error in evaluation worker: {e}")
                time.sleep(30)

    def _alert_monitor_worker(self) -> None:
        """アラート監視ワーカー"""
        while self.is_running:
            try:
                self._check_and_generate_alerts()
                time.sleep(self.alert_check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in alert monitor worker: {e}")
                time.sleep(30)

    def _metrics_collection_worker(self) -> None:
        """メトリクス収集ワーカー"""
        while self.is_running:
            try:
                self._collect_system_metrics()
                time.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"Error in metrics collection worker: {e}")
                time.sleep(30)

    def _check_and_generate_alerts(self) -> None:
        """アラート生成"""
        try:
            alerts = []

            # パフォーマンスアラート
            perf_alerts = self._check_performance_alerts()
            alerts.extend(perf_alerts)

            # 安全アラート
            safety_alerts = self._check_safety_alerts()
            alerts.extend(safety_alerts)

            # ドリフトアラート
            drift_alerts = self._check_drift_alerts()
            alerts.extend(drift_alerts)

            # 新しいアラートの処理
            for alert in alerts:
                if alert not in self.active_alerts:
                    self.active_alerts.append(alert)
                    self._trigger_alert_callbacks(alert)

            # 解決されたアラートのクリア
            resolved_alerts = []
            for alert in self.active_alerts:
                if self._is_alert_resolved(alert):
                    resolved_alerts.append(alert)

            for alert in resolved_alerts:
                self.active_alerts.remove(alert)

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")

    def _check_performance_alerts(self) -> List[MonitoringAlert]:
        """パフォーマンスアラートのチェック"""
        alerts = []

        try:
            if len(self.evaluation_history) > 0:
                latest_eval = self.evaluation_history[-1]

                if latest_eval.performance_metrics:
                    perf = latest_eval.performance_metrics

                    # 精度アラート
                    if perf.accuracy < 0.4:
                        alerts.append(
                            MonitoringAlert(
                                alert_id=f"perf_accuracy_{datetime.now().timestamp()}",
                                alert_type=AlertType.PERFORMANCE,
                                alert_level=AlertLevel.CRITICAL,
                                message=f"モデル精度が critically 低い: {perf.accuracy:.3f}",
                                timestamp=datetime.now(),
                                details={"accuracy": perf.accuracy},
                            )
                        )

                    # ドローダウンアラート
                    elif perf.max_drawdown > 0.25:
                        alerts.append(
                            MonitoringAlert(
                                alert_id=f"perf_drawdown_{datetime.now().timestamp()}",
                                alert_type=AlertType.PERFORMANCE,
                                alert_level=AlertLevel.HIGH,
                                message=f"最大ドローダウンが高い: {perf.max_drawdown:.3f}",
                                timestamp=datetime.now(),
                                details={"max_drawdown": perf.max_drawdown},
                            )
                        )

        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")

        return alerts

    def _check_safety_alerts(self) -> List[MonitoringAlert]:
        """安全アラートのチェック"""
        alerts = []

        try:
            safety_status = self.safety_manager.get_safety_status()

            if len(safety_status.active_anomalies) > 5:
                alerts.append(
                    MonitoringAlert(
                        alert_id=f"safety_anomalies_{datetime.now().timestamp()}",
                        alert_type=AlertType.SAFETY,
                        alert_level=AlertLevel.HIGH,
                        message=f"多数の異常が検知されました: {len(safety_status.active_anomalies)}件",
                        timestamp=datetime.now(),
                        details={"anomaly_count": len(safety_status.active_anomalies)},
                    )
                )

            if safety_status.system_health_score < 0.6:
                alerts.append(
                    MonitoringAlert(
                        alert_id=f"safety_score_{datetime.now().timestamp()}",
                        alert_type=AlertType.SAFETY,
                        alert_level=AlertLevel.CRITICAL,
                        message=f"システム安全スコアが critically 低い: {safety_status.system_health_score:.3f}",
                        timestamp=datetime.now(),
                        details={"safety_score": safety_status.system_health_score},
                    )
                )

        except Exception as e:
            logger.error(f"Safety alert check failed: {e}")

        return alerts

    def _check_drift_alerts(self) -> List[MonitoringAlert]:
        """ドリフトアラートのチェック"""
        alerts = []

        try:
            if len(self.evaluation_history) > 0:
                latest_eval = self.evaluation_history[-1]

                if latest_eval.drift_detected:
                    severity_level = AlertLevel.MEDIUM
                    if latest_eval.drift_severity >= 3:
                        severity_level = AlertLevel.HIGH
                    elif latest_eval.drift_severity >= 5:
                        severity_level = AlertLevel.CRITICAL

                    alerts.append(
                        MonitoringAlert(
                            alert_id=f"drift_detected_{datetime.now().timestamp()}",
                            alert_type=AlertType.DRIFT,
                            alert_level=severity_level,
                            message=f"コンセプトドリフトが検知されました (深刻度: {latest_eval.drift_severity})",
                            timestamp=datetime.now(),
                            details={"drift_severity": latest_eval.drift_severity},
                        )
                    )

        except Exception as e:
            logger.error(f"Drift alert check failed: {e}")

        return alerts

    def _is_alert_resolved(self, alert: MonitoringAlert) -> bool:
        """アラートが解決されたかチェック"""
        try:
            # アラートの種類に応じて解決条件をチェック
            if alert.alert_type == AlertType.PERFORMANCE:
                if len(self.evaluation_history) > 0:
                    latest_perf = self.evaluation_history[-1].performance_metrics
                    if latest_perf:
                        if "accuracy" in alert.details:
                            return latest_perf.accuracy >= 0.5
                        elif "max_drawdown" in alert.details:
                            return latest_perf.max_drawdown <= 0.2

            elif alert.alert_type == AlertType.SAFETY:
                safety_status = self.safety_manager.get_safety_status()
                if "anomaly_count" in alert.details:
                    return len(safety_status.active_anomalies) <= 2
                elif "safety_score" in alert.details:
                    return safety_status.system_health_score >= 0.8

            elif alert.alert_type == AlertType.DRIFT:
                if len(self.evaluation_history) > 0:
                    return not self.evaluation_history[-1].drift_detected

            # デフォルト: 5分後に解決とみなす
            return (datetime.now() - alert.timestamp).total_seconds() > 300

        except Exception as e:
            logger.error(f"Alert resolution check failed: {e}")
            return False

    def _collect_system_metrics(self) -> None:
        """システムメトリクスの収集"""
        try:
            import psutil

            system_metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage("/").percent,
                network_connections=len(psutil.net_connections()),
                active_threads=threading.active_count(),
            )

            self.system_metrics_history.append(system_metrics)

            # 古いメトリクスのクリーンアップ（保持期間: 24時間）
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.system_metrics_history = [
                m for m in self.system_metrics_history if m.timestamp > cutoff_time
            ]

        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]) -> None:
        """アラートコールバックの追加"""
        self.alert_callbacks.append(callback)

    def add_evaluation_callback(
        self, callback: Callable[[EvaluationResult], None]
    ) -> None:
        """評価コールバックの追加"""
        self.evaluation_callbacks.append(callback)

    def _trigger_alert_callbacks(self, alert: MonitoringAlert) -> None:
        """アラートコールバックの実行"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _trigger_evaluation_callbacks(self, result: EvaluationResult) -> None:
        """評価コールバックの実行"""
        for callback in self.evaluation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Evaluation callback failed: {e}")

    def get_evaluation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """評価サマリーの取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_evaluations = [
                e for e in self.evaluation_history if e.timestamp > cutoff_time
            ]

            if not recent_evaluations:
                return {"message": "No recent evaluations available"}

            # サマリー統計の計算
            scores = [
                e.overall_score
                for e in recent_evaluations
                if e.overall_score is not None
            ]
            drift_detections = sum(1 for e in recent_evaluations if e.drift_detected)

            return {
                "period_hours": hours,
                "total_evaluations": len(recent_evaluations),
                "average_score": np.mean(scores) if scores else 0.0,
                "score_std": np.std(scores) if scores else 0.0,
                "drift_detections": drift_detections,
                "drift_rate": drift_detections / len(recent_evaluations)
                if recent_evaluations
                else 0.0,
                "latest_score": recent_evaluations[-1].overall_score
                if recent_evaluations
                else None,
                "latest_recommendations": recent_evaluations[-1].recommendations
                if recent_evaluations
                else [],
            }

        except Exception as e:
            logger.error(f"Evaluation summary generation failed: {e}")
            return {"error": str(e)}

    def get_active_alerts(self) -> List[MonitoringAlert]:
        """アクティブなアラートの取得"""
        return self.active_alerts.copy()

    def clear_resolved_alerts(self) -> int:
        """解決されたアラートをクリア"""
        resolved_count = 0
        alerts_to_remove = []

        for alert in self.active_alerts:
            if self._is_alert_resolved(alert):
                alerts_to_remove.append(alert)
                resolved_count += 1

        for alert in alerts_to_remove:
            self.active_alerts.remove(alert)

        return resolved_count
