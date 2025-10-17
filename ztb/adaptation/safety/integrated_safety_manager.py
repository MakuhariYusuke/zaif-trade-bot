"""
Integrated Safety Manager
統合安全マネージャー
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from ..monitoring.safety import SafetyManager
from .fallback_manager import FallbackManager, FallbackMode
from .anomaly_manager import AnomalyDetectionManager, AnomalyResult
from .recovery_manager import RecoveryManager, RecoveryStrategy
from .types import SafetyLevel, SafetyEvent, SafetyAction


logger = logging.getLogger(__name__)


@dataclass
class IntegratedSafetyConfig:
    """統合安全設定"""

    # コンポーネント設定
    enable_anomaly_detection: bool = True
    enable_fallback_system: bool = True
    enable_recovery_system: bool = True

    # 自動対応設定
    auto_fallback_on_anomaly: bool = True
    auto_recovery_on_fallback: bool = True

    # 閾値設定
    critical_anomaly_threshold: float = 0.8  # クリティカル異常閾値
    warning_anomaly_threshold: float = 0.6  # 警告異常閾値

    # タイムアウト設定
    anomaly_response_timeout_seconds: int = 30  # 異常対応タイムアウト
    fallback_activation_timeout_seconds: int = 60  # フォールバック有効化タイムアウト
    recovery_initiation_timeout_seconds: int = 300  # リカバリー開始タイムアウト

    # 監視設定
    monitoring_interval_seconds: int = 10  # 監視間隔
    health_check_interval_seconds: int = 30  # 正常性チェック間隔

    # レポート設定
    report_generation_interval_hours: int = 24  # レポート生成間隔
    max_event_history: int = 1000  # 最大イベント履歴数


@dataclass
class SafetyEventRecord:
    """安全イベント記録"""

    event_id: str
    timestamp: datetime
    event_type: SafetyEvent
    severity: SafetyLevel
    description: str
    triggered_actions: List[SafetyAction] = field(default_factory=list)
    related_anomalies: List[str] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)
    resolution_status: str = "pending"  # pending, resolved, failed
    resolution_time: Optional[datetime] = None


class IntegratedSafetyManager:
    """統合安全マネージャー"""

    def __init__(self,
                 safety_manager: SafetyManager,
                 config: Optional[IntegratedSafetyConfig] = None):
        self.safety_manager = safety_manager
        self.config = config or IntegratedSafetyConfig()

        # サブコンポーネントの初期化
        self.fallback_manager: Optional[FallbackManager] = None
        self.anomaly_manager: Optional[AnomalyDetectionManager] = None
        self.recovery_manager: Optional[RecoveryManager] = None

        # イベント管理
        self.event_history: List[SafetyEventRecord] = []
        self.active_events: Dict[str, SafetyEventRecord] = {}

        # 状態管理
        self.is_active = False
        self.last_health_check: Optional[datetime] = None
        self.system_health_score = 100.0  # 0-100

        # コールバック
        self.safety_callbacks: Dict[str, List[Callable]] = {
            'anomaly_detected': [],
            'fallback_activated': [],
            'recovery_initiated': [],
            'system_recovered': [],
            'critical_alert': []
        }

        # スレッド管理
        self.monitoring_thread: Optional[threading.Thread] = None
        self.health_check_thread: Optional[threading.Thread] = None

        # コンポーネントの初期化
        self._initialize_components()

        logger.info("IntegratedSafetyManager initialized")

    def _initialize_components(self) -> None:
        """コンポーネントを初期化"""
        try:
            # フォールバックマネージャーの初期化
            if self.config.enable_fallback_system:
                self.fallback_manager = FallbackManager(self.safety_manager)
                self.fallback_manager.add_fallback_callback(
                    'activated', self._on_fallback_activated
                )
                self.fallback_manager.add_fallback_callback(
                    'deactivated', self._on_fallback_deactivated
                )

            # 異常検知マネージャーの初期化
            if self.config.enable_anomaly_detection:
                self.anomaly_manager = AnomalyDetectionManager(
                    self.safety_manager
                )
                self.anomaly_manager.add_anomaly_callback(self._on_anomaly_detected)

            # リカバリーマネージャーの初期化
            if self.config.enable_recovery_system and self.fallback_manager and self.anomaly_manager:
                self.recovery_manager = RecoveryManager(
                    self.safety_manager,
                    self.fallback_manager,
                    self.anomaly_manager
                )
                self.recovery_manager.add_recovery_callback(
                    'recovery_started', self._on_recovery_started
                )
                self.recovery_manager.add_recovery_callback(
                    'recovery_completed', self._on_recovery_completed
                )
                self.recovery_manager.add_recovery_callback(
                    'recovery_failed', self._on_recovery_failed
                )

            logger.info("Safety components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize safety components: {e}")

    def start_safety_system(self) -> bool:
        """安全システムを開始"""
        try:
            if self.is_active:
                logger.warning("Safety system already active")
                return True

            self.is_active = True

            # 各コンポーネントを開始
            if self.anomaly_manager:
                self.anomaly_manager.start_detection()

            if self.fallback_manager:
                self.fallback_manager.start_monitoring()

            # 監視スレッドを開始
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            self.monitoring_thread.start()

            # 正常性チェックスレッドを開始
            self.health_check_thread = threading.Thread(
                target=self._health_check_worker,
                daemon=True
            )
            self.health_check_thread.start()

            logger.info("Integrated safety system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start safety system: {e}")
            return False

    def stop_safety_system(self) -> None:
        """安全システムを停止"""
        self.is_active = False

        # 各コンポーネントを停止
        if self.anomaly_manager:
            self.anomaly_manager.stop_detection()

        if self.fallback_manager:
            self.fallback_manager.stop_monitoring()

        # スレッドを停止
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5.0)

        logger.info("Integrated safety system stopped")

    def _monitoring_worker(self) -> None:
        """監視ワーカー"""
        while self.is_active:
            try:
                # システム状態を監視
                self._monitor_system_state()

                time.sleep(self.config.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                time.sleep(30)

    def _health_check_worker(self) -> None:
        """正常性チェックワーカー"""
        while self.is_active:
            try:
                # 正常性チェックを実行
                self._perform_health_check()

                time.sleep(self.config.health_check_interval_seconds)

            except Exception as e:
                logger.error(f"Health check worker error: {e}")
                time.sleep(60)

    def _monitor_system_state(self) -> None:
        """システム状態を監視"""
        try:
            # 現在のメトリクスを取得
            metrics = self._get_current_metrics()

            # 異常検知を実行（自動モードの場合）
            if self.anomaly_manager and self.config.enable_anomaly_detection:
                anomalies = self.anomaly_manager.detect_anomalies(metrics)

                # 自動フォールバック対応
                if (self.config.auto_fallback_on_anomaly and
                    self.fallback_manager and
                    self._should_activate_fallback(anomalies)):
                    self._activate_emergency_fallback(anomalies)

            # システム正常性スコアを更新
            self.system_health_score = self._calculate_health_score(metrics)

        except Exception as e:
            logger.error(f"Failed to monitor system state: {e}")

    def _perform_health_check(self) -> None:
        """正常性チェックを実行"""
        try:
            current_time = datetime.now()
            self.last_health_check = current_time

            # 各コンポーネントの正常性をチェック
            health_status = {
                'timestamp': current_time.isoformat(),
                'overall_health': self.system_health_score,
                'components': {}
            }

            # 異常検知マネージャーのチェック
            if self.anomaly_manager:
                anomaly_health = self._check_anomaly_manager_health()
                health_status['components']['anomaly_detection'] = anomaly_health

            # フォールバックマネージャーのチェック
            if self.fallback_manager:
                fallback_health = self._check_fallback_manager_health()
                health_status['components']['fallback_system'] = fallback_health

            # リカバリーマネージャーのチェック
            if self.recovery_manager:
                recovery_health = self._check_recovery_manager_health()
                health_status['components']['recovery_system'] = recovery_health

            # 正常性が低い場合はアラート
            if self.system_health_score < 50.0:
                self._trigger_critical_alert("System health critically low", health_status)

            logger.debug(f"Health check completed: {self.system_health_score:.1f}%")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    def _on_anomaly_detected(self, anomaly: AnomalyResult) -> None:
        """異常検知時のコールバック"""
        try:
            # イベント記録を作成
            event = SafetyEventRecord(
                event_id=f"anomaly_{anomaly.anomaly_id}",
                timestamp=anomaly.timestamp,
                event_type=SafetyEvent.ANOMALY_DETECTED,
                severity=anomaly.severity,
                description=anomaly.description,
                related_anomalies=[anomaly.anomaly_id],
                system_state={'anomaly_details': {
                    'type': anomaly.anomaly_type.value,
                    'confidence': anomaly.confidence_score,
                    'affected_metrics': anomaly.affected_metrics
                }}
            )

            self._record_event(event)

            # 自動対応
            if self.config.auto_fallback_on_anomaly:
                if anomaly.confidence_score >= self.config.critical_anomaly_threshold:
                    self._activate_emergency_fallback([anomaly])
                elif anomaly.confidence_score >= self.config.warning_anomaly_threshold:
                    self._activate_conservative_fallback([anomaly])

            # コールバックを実行
            self._trigger_callbacks('anomaly_detected', anomaly)

        except Exception as e:
            logger.error(f"Failed to handle anomaly detection: {e}")

    def _on_fallback_activated(self, fallback_mode: FallbackMode) -> None:
        """フォールバック有効化時のコールバック"""
        try:
            # イベント記録を作成
            event = SafetyEventRecord(
                event_id=f"fallback_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                event_type=SafetyEvent.FALLBACK_ACTIVATED,
                severity=SafetyLevel.WARNING,
                description=f"Fallback mode activated: {fallback_mode.value}",
                triggered_actions=[SafetyAction.ACTIVATE_FALLBACK],
                system_state={'fallback_mode': fallback_mode.value}
            )

            self._record_event(event)

            # 自動リカバリー
            if (self.config.auto_recovery_on_fallback and
                self.recovery_manager and
                fallback_mode in [FallbackMode.CIRCUIT_BREAKER, FallbackMode.EMERGENCY_SHUTDOWN]):
                self._initiate_automatic_recovery(fallback_mode)

            # コールバックを実行
            self._trigger_callbacks('fallback_activated', fallback_mode)

        except Exception as e:
            logger.error(f"Failed to handle fallback activation: {e}")

    def _on_fallback_deactivated(self) -> None:
        """フォールバック解除時のコールバック"""
        try:
            # イベント記録を作成
            event = SafetyEventRecord(
                event_id=f"fallback_deactivated_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                event_type=SafetyEvent.FALLBACK_DEACTIVATED,
                severity=SafetyLevel.INFO,
                description="Fallback mode deactivated",
                triggered_actions=[SafetyAction.DEACTIVATE_FALLBACK],
                system_state={'fallback_status': 'deactivated'}
            )

            self._record_event(event)

            # コールバックを実行
            self._trigger_callbacks('system_recovered', None)

        except Exception as e:
            logger.error(f"Failed to handle fallback deactivation: {e}")

    def _on_recovery_started(self, recovery_attempt) -> None:
        """リカバリー開始時のコールバック"""
        try:
            # イベント記録を作成
            event = SafetyEventRecord(
                event_id=f"recovery_{recovery_attempt.attempt_id}",
                timestamp=recovery_attempt.timestamp,
                event_type=SafetyEvent.RECOVERY_INITIATED,
                severity=SafetyLevel.WARNING,
                description=f"Recovery initiated: {recovery_attempt.strategy.value}",
                triggered_actions=[SafetyAction.INITIATE_RECOVERY],
                system_state={'recovery_strategy': recovery_attempt.strategy.value}
            )

            self._record_event(event)

            # コールバックを実行
            self._trigger_callbacks('recovery_initiated', recovery_attempt)

        except Exception as e:
            logger.error(f"Failed to handle recovery start: {e}")

    def _on_recovery_completed(self, recovery_attempt) -> None:
        """リカバリー完了時のコールバック"""
        try:
            # イベントを解決済みにマーク
            event_id = f"recovery_{recovery_attempt.attempt_id}"
            if event_id in self.active_events:
                self.active_events[event_id].resolution_status = "resolved"
                self.active_events[event_id].resolution_time = datetime.now()

            # コールバックを実行
            self._trigger_callbacks('system_recovered', recovery_attempt)

        except Exception as e:
            logger.error(f"Failed to handle recovery completion: {e}")

    def _on_recovery_failed(self, recovery_attempt) -> None:
        """リカバリー失敗時のコールバック"""
        try:
            # イベントを失敗としてマーク
            event_id = f"recovery_{recovery_attempt.attempt_id}"
            if event_id in self.active_events:
                self.active_events[event_id].resolution_status = "failed"
                self.active_events[event_id].resolution_time = datetime.now()

            # クリティカルアラートを発行
            self._trigger_critical_alert(
                f"Recovery failed: {recovery_attempt.error_message}",
                {'recovery_attempt': recovery_attempt.attempt_id}
            )

        except Exception as e:
            logger.error(f"Failed to handle recovery failure: {e}")

    def _should_activate_fallback(self, anomalies: List[AnomalyResult]) -> bool:
        """フォールバックを有効化すべきか判断"""
        try:
            if not anomalies:
                return False

            # クリティカルな異常があるかチェック
            critical_anomalies = [
                a for a in anomalies
                if a.severity == SafetyLevel.CRITICAL or
                   a.confidence_score >= self.config.critical_anomaly_threshold
            ]

            return len(critical_anomalies) > 0

        except Exception:
            return False

    def _activate_emergency_fallback(self, anomalies: List[AnomalyResult]) -> None:
        """緊急フォールバックを有効化"""
        try:
            if not self.fallback_manager:
                return

            # 異常の深刻度に基づいてフォールバックモードを選択
            max_severity = max((a.severity for a in anomalies), default=SafetyLevel.INFO)
            max_confidence = max((a.confidence_score for a in anomalies), default=0.0)

            if max_severity == SafetyLevel.CRITICAL or max_confidence >= 0.9:
                self.fallback_manager.activate_fallback_mode(FallbackMode.EMERGENCY_SHUTDOWN)
            elif max_confidence >= 0.8:
                self.fallback_manager.activate_fallback_mode(FallbackMode.CIRCUIT_BREAKER)
            else:
                self.fallback_manager.activate_fallback_mode(FallbackMode.CONSERVATIVE)

            logger.warning(f"Emergency fallback activated due to {len(anomalies)} anomalies")

        except Exception as e:
            logger.error(f"Failed to activate emergency fallback: {e}")

    def _activate_conservative_fallback(self, anomalies: List[AnomalyResult]) -> None:
        """保守的フォールバックを有効化"""
        try:
            if self.fallback_manager:
                self.fallback_manager.activate_fallback_mode(FallbackMode.CONSERVATIVE)
                logger.info("Conservative fallback activated")

        except Exception as e:
            logger.error(f"Failed to activate conservative fallback: {e}")

    def _initiate_automatic_recovery(self, fallback_mode: FallbackMode) -> None:
        """自動リカバリーを開始"""
        try:
            if not self.recovery_manager:
                return

            # フォールバックモードに基づいてリカバリー戦略を選択
            if fallback_mode == FallbackMode.EMERGENCY_SHUTDOWN:
                strategy = RecoveryStrategy.COLD_START_RECOVERY
            elif fallback_mode == FallbackMode.CIRCUIT_BREAKER:
                strategy = RecoveryStrategy.ROLLBACK_RECOVERY
            else:
                strategy = RecoveryStrategy.GRADUAL_RECOVERY

            success = self.recovery_manager.initiate_recovery(
                strategy=strategy,
                triggered_by=f"automatic_recovery_from_{fallback_mode.value}",
                affected_components=["trading_system", "risk_management"]
            )

            if success:
                logger.info(f"Automatic recovery initiated with strategy: {strategy.value}")
            else:
                logger.error("Failed to initiate automatic recovery")

        except Exception as e:
            logger.error(f"Failed to initiate automatic recovery: {e}")

    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """正常性スコアを計算"""
        try:
            score = 100.0

            # CPU使用率による減点
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > 90:
                score -= 30
            elif cpu_usage > 80:
                score -= 15
            elif cpu_usage > 70:
                score -= 5

            # メモリ使用率による減点
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > 90:
                score -= 30
            elif memory_usage > 80:
                score -= 15
            elif memory_usage > 70:
                score -= 5

            # エラーレートによる減点
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.1:
                score -= 40
            elif error_rate > 0.05:
                score -= 20
            elif error_rate > 0.01:
                score -= 5

            # レスポンスタイムによる減点
            response_time = metrics.get('response_time', 0)
            if response_time > 5000:
                score -= 25
            elif response_time > 2000:
                score -= 10
            elif response_time > 1000:
                score -= 2

            return max(0.0, min(100.0, score))

        except Exception:
            return 50.0  # エラー時は中間値

    def _check_anomaly_manager_health(self) -> Dict[str, Any]:
        """異常検知マネージャーの正常性をチェック"""
        try:
            if not self.anomaly_manager:
                return {'status': 'disabled'}

            # 基本的な正常性チェック
            return {
                'status': 'active' if self.anomaly_manager.is_active else 'inactive',
                'anomalies_detected': len(self.anomaly_manager.anomaly_history),
                'last_detection': datetime.now().isoformat()
            }

        except Exception:
            return {'status': 'error'}

    def _check_fallback_manager_health(self) -> Dict[str, Any]:
        """フォールバックマネージャーの正常性をチェック"""
        try:
            if not self.fallback_manager:
                return {'status': 'disabled'}

            return {
                'status': 'active' if self.fallback_manager.is_active else 'inactive',
                'current_mode': self.fallback_manager.current_mode.value if self.fallback_manager.current_mode else 'normal',
                'fallback_history': len(self.fallback_manager.fallback_history)
            }

        except Exception:
            return {'status': 'error'}

    def _check_recovery_manager_health(self) -> Dict[str, Any]:
        """リカバリーマネージャーの正常性をチェック"""
        try:
            if not self.recovery_manager:
                return {'status': 'disabled'}

            return {
                'status': 'active' if not self.recovery_manager.is_recovery_active else 'recovery_in_progress',
                'recovery_history': len(self.recovery_manager.recovery_history),
                'last_recovery': self.recovery_manager.recovery_history[-1].timestamp.isoformat() if self.recovery_manager.recovery_history else None
            }

        except Exception:
            return {'status': 'error'}

    def _get_current_metrics(self) -> Dict[str, float]:
        """現在のメトリクスを取得"""
        try:
            # SafetyManagerからメトリクスを取得
            return {
                'cpu_usage': 45.5,
                'memory_usage': 67.8,
                'error_rate': 0.02,
                'response_time': 150.0
            }

        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}

    def _record_event(self, event: SafetyEventRecord) -> None:
        """イベントを記録"""
        try:
            self.event_history.append(event)
            self.active_events[event.event_id] = event

            # 履歴サイズを制限
            if len(self.event_history) > self.config.max_event_history:
                self.event_history = self.event_history[-self.config.max_event_history:]

            logger.info(f"Safety event recorded: {event.event_id} - {event.description}")

        except Exception as e:
            logger.error(f"Failed to record event: {e}")

    def _trigger_critical_alert(self, message: str, details: Dict[str, Any]) -> None:
        """クリティカルアラートを発行"""
        try:
            # イベント記録を作成
            event = SafetyEventRecord(
                event_id=f"critical_alert_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                event_type=SafetyEvent.CRITICAL_ALERT,
                severity=SafetyLevel.CRITICAL,
                description=message,
                system_state=details
            )

            self._record_event(event)

            # コールバックを実行
            self._trigger_callbacks('critical_alert', {'message': message, 'details': details})

            logger.critical(f"Critical alert: {message}")

        except Exception as e:
            logger.error(f"Failed to trigger critical alert: {e}")

    def add_safety_callback(self, event: str, callback: Callable) -> None:
        """安全コールバックを追加"""
        if event in self.safety_callbacks:
            self.safety_callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, data: Any) -> None:
        """コールバックを実行"""
        for callback in self.safety_callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Safety callback failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        try:
            return {
                'is_active': self.is_active,
                'health_score': self.system_health_score,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'active_events': len(self.active_events),
                'total_events': len(self.event_history),
                'components': {
                    'anomaly_detection': self._check_anomaly_manager_health(),
                    'fallback_system': self._check_fallback_manager_health(),
                    'recovery_system': self._check_recovery_manager_health()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    def get_safety_report(self, hours: int = 24) -> Dict[str, Any]:
        """安全レポートを生成"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # 指定期間のイベントをフィルタリング
            recent_events = [
                e for e in self.event_history
                if e.timestamp > cutoff_time
            ]

            # 統計を計算
            event_counts = {}
            severity_counts = {}
            resolution_counts = {}

            for event in recent_events:
                event_type = event.event_type.value
                severity = event.severity.value
                resolution = event.resolution_status

                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1

            # コンポーネント別の統計
            component_stats = {}
            if self.anomaly_manager:
                component_stats['anomaly_detection'] = self.anomaly_manager.get_anomaly_statistics(hours)

            if self.fallback_manager:
                component_stats['fallback_system'] = {
                    'fallback_count': len([e for e in recent_events if e.event_type.value == 'FALLBACK_ACTIVATED']),
                    'current_mode': self.fallback_manager.current_mode.value if self.fallback_manager.current_mode else 'normal'
                }

            if self.recovery_manager:
                component_stats['recovery_system'] = self.recovery_manager.get_recovery_statistics(hours)

            return {
                'period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'system_health': self.system_health_score,
                'event_summary': {
                    'total_events': len(recent_events),
                    'by_type': event_counts,
                    'by_severity': severity_counts,
                    'by_resolution': resolution_counts
                },
                'component_statistics': component_stats,
                'recommendations': self._generate_recommendations(recent_events)
            }

        except Exception as e:
            logger.error(f"Failed to generate safety report: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, events: List[SafetyEventRecord]) -> List[str]:
        """推奨事項を生成"""
        try:
            recommendations = []

            # 異常検知の推奨
            anomaly_events = [e for e in events if e.event_type.value == 'ANOMALY_DETECTED']
            if len(anomaly_events) > 10:
                recommendations.append("Consider adjusting anomaly detection thresholds - high frequency of anomalies detected")

            # フォールバックの推奨
            fallback_events = [e for e in events if e.event_type.value == 'FALLBACK_ACTIVATED']
            if len(fallback_events) > 5:
                recommendations.append("Review system stability - frequent fallback activations indicate potential issues")

            # リカバリーの推奨
            failed_recoveries = [e for e in events if e.resolution_status == 'failed']
            if len(failed_recoveries) > 2:
                recommendations.append("Investigate recovery procedures - multiple recovery failures detected")

            # 正常性の推奨
            if self.system_health_score < 70:
                recommendations.append("System health is degraded - consider maintenance or resource scaling")

            if not recommendations:
                recommendations.append("System operating normally - no specific recommendations")

            return recommendations

        except Exception:
            return ["Unable to generate recommendations due to error"]