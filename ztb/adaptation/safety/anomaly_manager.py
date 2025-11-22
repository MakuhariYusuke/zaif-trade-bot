"""
Anomaly Detection Manager
異常検知マネージャー
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..monitoring.safety import SafetyManager
from .types import AnomalyType, SafetyLevel

logger = logging.getLogger(__name__)


class AnomalyDetectionMethod(Enum):
    """異常検知手法"""

    STATISTICAL = "statistical"  # 統計的手法（Z-score, IQR）
    ISOLATION_FOREST = "isolation_forest"  # 孤立森
    ONE_CLASS_SVM = "one_class_svm"  # 一クラスSVM
    AUTOENCODER = "autoencoder"  # オートエンコーダー
    PROPhet = "prophet"  # Prophetベースの時系列異常検知


@dataclass
class AnomalyConfig:
    """異常検知設定"""

    # 検知手法設定
    enabled_methods: List[AnomalyDetectionMethod] = field(
        default_factory=lambda: [
            AnomalyDetectionMethod.STATISTICAL,
            AnomalyDetectionMethod.ISOLATION_FOREST,
        ]
    )

    # 統計的検知設定
    statistical_threshold_sigma: float = 3.0  # 標準偏差の閾値
    statistical_min_samples: int = 100  # 最小サンプル数
    statistical_window_size: int = 1000  # 分析ウィンドウサイズ

    # 機械学習検知設定
    ml_contamination: float = 0.1  # 異常データの割合（推定）
    ml_min_samples: int = 1000  # 学習のための最小サンプル数
    ml_retrain_interval_hours: int = 24  # 再学習間隔

    # アラート設定
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "high_confidence": 0.8,  # 高確信度閾値
            "medium_confidence": 0.6,  # 中確信度閾値
            "low_confidence": 0.4,  # 低確信度閾値
        }
    )

    # 検知間隔設定
    detection_interval_seconds: int = 60  # 検知間隔
    batch_processing_size: int = 100  # バッチ処理サイズ

    # 履歴保持設定
    max_anomaly_history: int = 1000  # 最大異常履歴数
    anomaly_retention_days: int = 30  # 異常データ保持期間


@dataclass
class AnomalyResult:
    """異常検知結果"""

    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    detection_method: AnomalyDetectionMethod
    confidence_score: float
    severity: SafetyLevel
    affected_metrics: List[str]
    anomaly_value: float
    expected_value: Optional[float]
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetectionManager:
    """異常検知マネージャー"""

    def __init__(
        self, safety_manager: SafetyManager, config: Optional[AnomalyConfig] = None
    ):
        self.safety_manager = safety_manager
        self.config = config or AnomalyConfig()

        # 検知器の初期化
        self.detectors: Dict[AnomalyDetectionMethod, Any] = {}
        self._initialize_detectors()

        # データ管理
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.anomaly_history: List[AnomalyResult] = []
        self.scalers: Dict[str, StandardScaler] = {}

        # 状態管理
        self.is_active = False
        self.last_training_time: Optional[datetime] = None

        # コールバック
        self.anomaly_detected_callbacks: List[Callable[[AnomalyResult], None]] = []

        # スレッド管理
        self.detection_thread: Optional[threading.Thread] = None

        logger.info("AnomalyDetectionManager initialized")

    def start_detection(self) -> bool:
        """検知を開始"""
        try:
            if self.is_active:
                logger.warning("Detection already active")
                return True

            self.is_active = True
            self.detection_thread = threading.Thread(
                target=self._detection_worker, daemon=True
            )
            self.detection_thread.start()

            logger.info("Anomaly detection started")
            return True

        except Exception as e:
            logger.error(f"Failed to start anomaly detection: {e}")
            return False

    def stop_detection(self) -> None:
        """検知を停止"""
        self.is_active = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
        logger.info("Anomaly detection stopped")

    def detect_anomalies(self, metrics_data: Dict[str, float]) -> List[AnomalyResult]:
        """異常を検知"""
        anomalies = []
        current_time = datetime.now()

        try:
            # メトリクスデータを履歴に追加
            for metric_name, value in metrics_data.items():
                self.metric_history[metric_name].append((current_time, value))

                # 履歴サイズを制限
                if (
                    len(self.metric_history[metric_name])
                    > self.config.statistical_window_size
                ):
                    self.metric_history[metric_name] = self.metric_history[metric_name][
                        -self.config.statistical_window_size :
                    ]

            # 各検知手法で異常を検知
            for method in self.config.enabled_methods:
                try:
                    method_anomalies = self._detect_with_method(
                        method, metrics_data, current_time
                    )
                    anomalies.extend(method_anomalies)
                except Exception as e:
                    logger.error(f"Error in {method.value} detection: {e}")

            # 重複除去とフィルタリング
            anomalies = self._filter_anomalies(anomalies)

            # 異常を履歴に追加
            self.anomaly_history.extend(anomalies)
            if len(self.anomaly_history) > self.config.max_anomaly_history:
                self.anomaly_history = self.anomaly_history[
                    -self.config.max_anomaly_history :
                ]

            # コールバックを実行
            for anomaly in anomalies:
                self._trigger_anomaly_callbacks(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return []

    def _initialize_detectors(self) -> None:
        """検知器を初期化"""
        try:
            for method in self.config.enabled_methods:
                if method == AnomalyDetectionMethod.ISOLATION_FOREST:
                    self.detectors[method] = IsolationForest(
                        contamination=self.config.ml_contamination, random_state=42
                    )
                elif method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                    from sklearn.svm import OneClassSVM

                    self.detectors[method] = OneClassSVM(
                        nu=self.config.ml_contamination, kernel="rbf"
                    )
                # 他の手法も必要に応じて初期化

            logger.info(f"Initialized {len(self.detectors)} anomaly detectors")

        except Exception as e:
            logger.error(f"Failed to initialize detectors: {e}")

    def _detect_with_method(
        self,
        method: AnomalyDetectionMethod,
        metrics_data: Dict[str, float],
        timestamp: datetime,
    ) -> List[AnomalyResult]:
        """指定された手法で異常を検知"""
        anomalies = []

        try:
            if method == AnomalyDetectionMethod.STATISTICAL:
                anomalies.extend(self._statistical_detection(metrics_data, timestamp))
            elif method == AnomalyDetectionMethod.ISOLATION_FOREST:
                anomalies.extend(
                    self._isolation_forest_detection(metrics_data, timestamp)
                )
            elif method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                anomalies.extend(self._one_class_svm_detection(metrics_data, timestamp))

            return anomalies

        except Exception as e:
            logger.error(f"Error in {method.value} detection: {e}")
            return []

    def _statistical_detection(
        self, metrics_data: Dict[str, float], timestamp: datetime
    ) -> List[AnomalyResult]:
        """統計的手法による異常検知"""
        anomalies = []

        try:
            for metric_name, value in metrics_data.items():
                history = self.metric_history[metric_name]

                if len(history) < self.config.statistical_min_samples:
                    continue

                # 最近のデータを取得
                recent_values = [
                    v for _, v in history[-self.config.statistical_window_size :]
                ]

                # Z-scoreを計算
                if len(recent_values) > 1:
                    mean_val = np.mean(recent_values)
                    std_val = np.std(recent_values)

                    if std_val > 0:
                        z_score = abs(value - mean_val) / std_val

                        if z_score > self.config.statistical_threshold_sigma:
                            confidence = min(
                                z_score / (self.config.statistical_threshold_sigma * 2),
                                1.0,
                            )
                            severity = self._calculate_severity(confidence)

                            anomaly = AnomalyResult(
                                anomaly_id=f"statistical_{metric_name}_{timestamp.timestamp()}",
                                timestamp=timestamp,
                                anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                                detection_method=AnomalyDetectionMethod.STATISTICAL,
                                confidence_score=confidence,
                                severity=severity,
                                affected_metrics=[metric_name],
                                anomaly_value=value,
                                expected_value=mean_val,
                                description=f"Statistical anomaly detected in {metric_name}: Z-score = {z_score:.2f}",
                                raw_data={
                                    "z_score": z_score,
                                    "mean": mean_val,
                                    "std": std_val,
                                    "window_size": len(recent_values),
                                },
                            )
                            anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Statistical detection failed: {e}")
            return []

    def _isolation_forest_detection(
        self, metrics_data: Dict[str, float], timestamp: datetime
    ) -> List[AnomalyResult]:
        """孤立森による異常検知"""
        anomalies = []

        try:
            # 十分なデータがあるかチェック
            total_samples = sum(
                len(history) for history in self.metric_history.values()
            )
            if total_samples < self.config.ml_min_samples:
                return anomalies

            # 特徴量行列を作成
            feature_names = list(metrics_data.keys())
            feature_matrix = []

            for metric_name in feature_names:
                history = self.metric_history[metric_name]
                if len(history) >= 100:  # 最低100サンプル
                    recent_values = [v for _, v in history[-100:]]
                    feature_matrix.append(recent_values)
                else:
                    feature_matrix.append([0] * 100)  # パディング

            if len(feature_matrix[0]) == 0:
                return anomalies

            # 特徴量行列を転置
            X = np.array(feature_matrix).T

            # スケーリング
            if "isolation_forest" not in self.scalers:
                self.scalers["isolation_forest"] = StandardScaler()
                X_scaled = self.scalers["isolation_forest"].fit_transform(X)
            else:
                X_scaled = self.scalers["isolation_forest"].transform(X)

            # モデルが学習済みかチェック
            detector = self.detectors.get(AnomalyDetectionMethod.ISOLATION_FOREST)
            if detector is None:
                return anomalies

            # 再学習が必要かチェック
            if (
                self.last_training_time is None
                or (datetime.now() - self.last_training_time).total_seconds()
                > self.config.ml_retrain_interval_hours * 3600
            ):
                detector.fit(X_scaled)
                self.last_training_time = datetime.now()

            # 予測
            current_features = np.array(
                [[metrics_data.get(name, 0) for name in feature_names]]
            )
            current_scaled = self.scalers["isolation_forest"].transform(
                current_features
            )

            # 異常スコアを計算（-1: 正常, 1: 異常）
            prediction = detector.predict(current_scaled)[0]
            score = detector.score_samples(current_scaled)[0]

            # 異常スコアを確信度に変換（低いスコアほど異常）
            confidence = 1.0 / (1.0 + np.exp(score))  # シグモイド変換

            if (
                prediction == -1
                and confidence > self.config.alert_thresholds["low_confidence"]
            ):
                severity = self._calculate_severity(confidence)

                anomaly = AnomalyResult(
                    anomaly_id=f"iforest_{timestamp.timestamp()}",
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.ML_OUTLIER,
                    detection_method=AnomalyDetectionMethod.ISOLATION_FOREST,
                    confidence_score=confidence,
                    severity=severity,
                    affected_metrics=feature_names,
                    anomaly_value=score,
                    expected_value=None,
                    description=f"Isolation Forest anomaly detected: score = {score:.4f}",
                    raw_data={
                        "anomaly_score": score,
                        "prediction": prediction,
                        "features": feature_names,
                    },
                )
                anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            return []

    def _one_class_svm_detection(
        self, metrics_data: Dict[str, float], timestamp: datetime
    ) -> List[AnomalyResult]:
        """一クラスSVMによる異常検知"""
        anomalies = []

        try:
            # 孤立森と同様の実装
            # （簡略化のため、ここでは基本的な実装のみ）
            detector = self.detectors.get(AnomalyDetectionMethod.ONE_CLASS_SVM)
            if detector is None:
                return anomalies

            # 特徴量の準備と予測
            # （実際の実装ではより詳細な特徴量エンジニアリングが必要）

            return anomalies

        except Exception as e:
            logger.error(f"One-Class SVM detection failed: {e}")
            return []

    def _calculate_severity(self, confidence: float) -> SafetyLevel:
        """確信度から深刻度を計算"""
        try:
            if confidence >= self.config.alert_thresholds["high_confidence"]:
                return SafetyLevel.CRITICAL
            elif confidence >= self.config.alert_thresholds["medium_confidence"]:
                return SafetyLevel.WARNING
            else:
                return SafetyLevel.INFO

        except Exception:
            return SafetyLevel.WARNING

    def _filter_anomalies(self, anomalies: List[AnomalyResult]) -> List[AnomalyResult]:
        """異常をフィルタリング（重複除去など）"""
        try:
            # 同じメトリクスに対する最近の異常をフィルタリング
            filtered = []
            recent_anomalies = {}

            # 過去5分間の異常をチェック
            cutoff_time = datetime.now() - timedelta(minutes=5)

            for anomaly in self.anomaly_history:
                if anomaly.timestamp > cutoff_time:
                    for metric in anomaly.affected_metrics:
                        if metric not in recent_anomalies:
                            recent_anomalies[metric] = []
                        recent_anomalies[metric].append(anomaly)

            # 新しい異常をフィルタリング
            for anomaly in anomalies:
                should_include = True

                for metric in anomaly.affected_metrics:
                    if metric in recent_anomalies:
                        # 同じタイプの異常が最近検知されていないかチェック
                        recent_same_type = [
                            a
                            for a in recent_anomalies[metric]
                            if a.anomaly_type == anomaly.anomaly_type
                        ]
                        if recent_same_type:
                            should_include = False
                            break

                if should_include:
                    filtered.append(anomaly)

            return filtered

        except Exception as e:
            logger.error(f"Failed to filter anomalies: {e}")
            return anomalies

    def _detection_worker(self) -> None:
        """検知ワーカー"""
        while self.is_active:
            try:
                # 最新のメトリクスを取得
                latest_metrics = self._get_latest_metrics()

                if latest_metrics:
                    # 異常検知を実行
                    anomalies = self.detect_anomalies(latest_metrics)

                    if anomalies:
                        logger.info(f"Detected {len(anomalies)} anomalies")

                time.sleep(self.config.detection_interval_seconds)

            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
                time.sleep(30)

    def _get_latest_metrics(self) -> Dict[str, float]:
        """最新のメトリクスを取得"""
        try:
            # SafetyManagerからメトリクスを取得
            # （実際の実装では適切なメソッドを呼び出す）
            return {
                "cpu_usage": 45.5,
                "memory_usage": 67.8,
                "error_rate": 0.02,
                "response_time": 150.0,
            }

        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return {}

    def add_anomaly_callback(self, callback: Callable[[AnomalyResult], None]) -> None:
        """異常検知コールバックを追加"""
        self.anomaly_detected_callbacks.append(callback)

    def _trigger_anomaly_callbacks(self, anomaly: AnomalyResult) -> None:
        """異常検知コールバックを実行"""
        for callback in self.anomaly_detected_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Anomaly callback failed: {e}")

    def get_anomaly_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """異常履歴を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_anomalies = [
                a for a in self.anomaly_history if a.timestamp > cutoff_time
            ]

            return [
                {
                    "anomaly_id": a.anomaly_id,
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.anomaly_type.value,
                    "method": a.detection_method.value,
                    "confidence": a.confidence_score,
                    "severity": a.severity.value,
                    "affected_metrics": a.affected_metrics,
                    "description": a.description,
                }
                for a in recent_anomalies
            ]

        except Exception as e:
            logger.error(f"Failed to get anomaly history: {e}")
            return []

    def get_anomaly_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """異常統計を取得"""
        try:
            history = self.get_anomaly_history(hours)

            if not history:
                return {"message": "No anomalies in the specified period"}

            # 統計計算
            total_anomalies = len(history)
            severity_counts = {}
            type_counts = {}
            method_counts = {}

            for anomaly in history:
                severity = anomaly["severity"]
                anomaly_type = anomaly["type"]
                method = anomaly["method"]

                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
                method_counts[method] = method_counts.get(method, 0) + 1

            avg_confidence = np.mean([a["confidence"] for a in history])

            return {
                "period_hours": hours,
                "total_anomalies": total_anomalies,
                "severity_distribution": severity_counts,
                "type_distribution": type_counts,
                "method_distribution": method_counts,
                "average_confidence": float(avg_confidence),
                "anomalies_per_hour": total_anomalies / hours,
            }

        except Exception as e:
            logger.error(f"Failed to get anomaly statistics: {e}")
            return {"error": str(e)}
