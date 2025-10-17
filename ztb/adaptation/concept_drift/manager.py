"""
Concept Drift Detection Manager
複数のドリフト検知アルゴリズムを統合管理
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .detector import (
    DriftDetector,
    KolmogorovSmirnovDetector,
    ADWINDetector,
    DDMDetector,
    EDDMDetector
)
from .drift_types import (
    DriftType,
    DriftSeverity,
    DriftDetectionResult
)
from .config import ConceptDriftConfig
from .config import ConceptDriftConfig

logger = logging.getLogger(__name__)


class ConceptDriftManager:
    """コンセプトドリフト検知マネージャー"""

    def __init__(self, config: Optional[ConceptDriftConfig] = None):
        self.config = config or ConceptDriftConfig()
        self.detectors: Dict[str, DriftDetector] = {}
        self.drift_history: List[DriftDetectionResult] = []
        self.last_detection_time: Optional[datetime] = None

        # 検知器の初期化
        self._initialize_detectors()

    def _initialize_detectors(self):
        """検知器を初期化"""
        detector_classes = {
            'ks_test': KolmogorovSmirnovDetector,
            'adwin': ADWINDetector,
            'ddm': DDMDetector,
            'eddm': EDDMDetector
        }

        for name, detector_class in detector_classes.items():
            if getattr(self.config, f'enable_{name}', True):
                self.detectors[name] = detector_class(self.config)
                logger.info(f"Initialized drift detector: {name}")

    def detect_drift(self,
                    data: Union[np.ndarray, pd.DataFrame],
                    error_data: Optional[np.ndarray] = None,
                    parallel: bool = True) -> DriftDetectionResult:
        """
        ドリフトを検知

        Args:
            data: 検知対象のデータ
            error_data: エラー指標（DDM, EDDM用）
            parallel: 並列実行フラグ

        Returns:
            DriftDetectionResult: 検知結果
        """
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合、数値列のみを使用
            numeric_data = data.select_dtypes(include=[np.number]).values
            if numeric_data.size == 0:
                raise ValueError("No numeric columns found in DataFrame")
            data_array = numeric_data.flatten()
        else:
            data_array = np.asarray(data).flatten()

        # データの検証
        if len(data_array) == 0:
            raise ValueError("Empty data provided")

        current_time = datetime.now()

        # 検知間隔チェック
        if (self.last_detection_time and
            (current_time - self.last_detection_time).seconds < self.config.detection_interval_seconds):
            # 前回の結果を返す
            return self.drift_history[-1] if self.drift_history else DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                confidence=1.0,
                timestamp=current_time
            )

        self.last_detection_time = current_time

        try:
            if parallel and len(self.detectors) > 1:
                # 並列実行
                results = self._detect_parallel(data_array, error_data)
            else:
                # 順次実行
                results = self._detect_sequential(data_array, error_data)

            # 結果の統合
            final_result = self._aggregate_results(results)
            final_result.timestamp = current_time

            # 履歴に追加
            self.drift_history.append(final_result)

            # 履歴サイズを制限
            if len(self.drift_history) > self.config.max_history_size:
                self.drift_history.pop(0)

            logger.info(f"Drift detection completed: {final_result.drift_detected}, "
                       f"severity: {final_result.severity}")

            return final_result

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            error_result = DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                confidence=0.0,
                timestamp=current_time,
                error=str(e)
            )
            return error_result

    def _detect_parallel(self, data: np.ndarray, error_data: Optional[np.ndarray]) -> List[DriftDetectionResult]:
        """並列でドリフト検知を実行"""
        results = []

        with ThreadPoolExecutor(max_workers=min(len(self.detectors), 4)) as executor:
            future_to_detector = {
                executor.submit(self._call_detector, name, detector, data, error_data): name
                for name, detector in self.detectors.items()
            }

            for future in as_completed(future_to_detector, timeout=30.0):  # 30秒タイムアウト
                detector_name = future_to_detector[future]
                try:
                    result = future.result(timeout=10.0)  # 個別の結果取得にもタイムアウト
                    results.append(result)
                except Exception as e:
                    logger.error(f"Detector {detector_name} failed: {e}")
                    # エラーの場合は中立的結果を追加
                    results.append(DriftDetectionResult(
                        drift_detected=False,
                        drift_type=DriftType.NONE,
                        severity=DriftSeverity.NONE,
                        confidence=0.5,
                        timestamp=datetime.now(),
                        error=str(e)
                    ))

        return results

    def _call_detector(self, name: str, detector, data: np.ndarray, error_data: Optional[np.ndarray]):
        """検知器を適切なパラメータで呼び出す"""
        # DDMとEDDMはerror_dataを必要とする
        if name in ['ddm', 'eddm']:
            return detector.detect_drift(data, error_data)
        else:
            # KS-testとADWINはerror_dataを必要としない
            return detector.detect_drift(data)

    def _detect_sequential(self, data: np.ndarray, error_data: Optional[np.ndarray]) -> List[DriftDetectionResult]:
        """順次でドリフト検知を実行"""
        results = []

        for name, detector in self.detectors.items():
            try:
                # 検知器タイプに応じて適切なパラメータを渡す
                if name in ['ddm', 'eddm']:
                    result = detector.detect_drift(data, error_data)
                else:
                    result = detector.detect_drift(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Detector {name} failed: {e}")
                results.append(DriftDetectionResult(
                    drift_detected=False,
                    drift_type=DriftType.NONE,
                    severity=DriftSeverity.NONE,
                    confidence=0.5,
                    timestamp=datetime.now(),
                    error=str(e)
                ))

        return results

    def _aggregate_results(self, results: List[DriftDetectionResult]) -> DriftDetectionResult:
        """複数の検知結果を集約"""
        if not results:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                confidence=1.0,
                timestamp=datetime.now()
            )

        # ドリフト検知の投票
        drift_votes = sum(1 for r in results if r.drift_detected)
        total_detectors = len(results)

        # 過半数の検知器がドリフトを検知した場合
        drift_detected = drift_votes >= (total_detectors // 2 + 1)

        if not drift_detected:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                confidence=1.0 - (drift_votes / total_detectors),
                timestamp=datetime.now()
            )

        # ドリフト検知時の集約
        # 最も厳しい重大度を使用
        severities = [r.severity for r in results if r.drift_detected]
        max_severity = max(severities) if severities else DriftSeverity.LOW

        # 平均信頼度
        confidences = [r.confidence for r in results if r.drift_detected]
        avg_confidence = np.mean(confidences) if confidences else 0.5

        # 平均p値
        p_values = [r.p_value for r in results if r.drift_detected and r.p_value is not None]
        avg_p_value = np.mean(p_values) if p_values else 0.5

        # 平均統計量
        statistics = [r.statistic for r in results if r.drift_detected and r.statistic is not None]
        avg_statistic = np.mean(statistics) if statistics else 0.0

        return DriftDetectionResult(
            drift_detected=True,
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=max_severity,
            confidence=avg_confidence,
            p_value=avg_p_value,
            statistic=avg_statistic,
            timestamp=datetime.now(),
            metadata={
                'detector_votes': drift_votes,
                'total_detectors': total_detectors,
                'individual_results': [
                    {
                        'detector': list(self.detectors.keys())[i],
                        'drift_detected': r.drift_detected,
                        'severity': r.severity.value,
                        'confidence': r.confidence
                    } for i, r in enumerate(results)
                ]
            }
        )

    def update_reference_data(self, data: Union[np.ndarray, pd.DataFrame]):
        """参照データを更新"""
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number]).values
            if numeric_data.size == 0:
                raise ValueError("No numeric columns found in DataFrame")
            data_array = numeric_data.flatten()
        else:
            data_array = np.asarray(data).flatten()

        for detector in self.detectors.values():
            detector.update_reference(data_array)

        logger.info("Reference data updated for all detectors")

    def get_drift_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[DriftDetectionResult]:
        """ドリフト履歴を取得"""
        history = self.drift_history

        if start_time:
            history = [r for r in history if r.timestamp >= start_time]
        if end_time:
            history = [r for r in history if r.timestamp <= end_time]

        return history

    def get_detector_stats(self) -> Dict[str, Any]:
        """検知器の統計情報を取得"""
        stats = {}

        for name, detector in self.detectors.items():
            history = detector.get_drift_history()
            stats[name] = {
                'total_detections': len(history),
                'last_detection': history[-1] if history else None,
                'avg_score': np.mean([score for _, score in history]) if history else 0.0
            }

        return stats

    def reset_detectors(self):
        """すべての検知器をリセット"""
        for detector in self.detectors.values():
            detector.reference_data = None
            detector.drift_scores.clear()
            detector.timestamps.clear()

        self.drift_history.clear()
        self.last_detection_time = None

        logger.info("All detectors reset")