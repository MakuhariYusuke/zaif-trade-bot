"""
Concept Drift Detection Algorithms
市場変化検知のための統計的・機械学習ベースのアルゴリズム実装
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from .drift_types import DriftType, DriftSeverity, DriftDetectionResult
from .config import ConceptDriftConfig

logger = logging.getLogger(__name__)


class DriftDetector(ABC):
    """ドリフト検知器の基底クラス"""

    def __init__(self, config: ConceptDriftConfig):
        self.config = config
        self.reference_data: Optional[np.ndarray] = None
        self.drift_scores: List[float] = []
        self.timestamps: List[datetime] = []

    @abstractmethod
    def detect_drift(self, new_data: np.ndarray, error_data: Optional[np.ndarray] = None) -> DriftDetectionResult:
        """ドリフトを検知"""
        # 空のデータチェック
        if len(new_data) == 0:
            raise ValueError("Empty data provided for drift detection")
        pass

    def update_reference(self, data: np.ndarray):
        """参照データを更新"""
        self.reference_data = data.copy()

    def get_drift_history(self) -> List[Tuple[datetime, float]]:
        """ドリフト履歴を取得"""
        return list(zip(self.timestamps, self.drift_scores))


class KolmogorovSmirnovDetector(DriftDetector):
    """Kolmogorov-Smirnov検定によるドリフト検知"""

    def detect_drift(self, new_data: np.ndarray, error_data: Optional[np.ndarray] = None) -> DriftDetectionResult:
        if self.reference_data is None:
            self.update_reference(new_data)
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                confidence=1.0,
                p_value=1.0,
                statistic=0.0,
                timestamp=datetime.now()
            )

        try:
            # KS検定を実行
            statistic, p_value = stats.ks_2samp(self.reference_data, new_data)

            # ドリフト判定
            drift_detected = p_value < self.config.ks_test_significance_level
            confidence = 1.0 - p_value

            # 重大度判定
            severity = self._calculate_severity(statistic, p_value)

            result = DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type=DriftType.CONCEPT_DRIFT if drift_detected else DriftType.NONE,
                severity=severity,
                confidence=confidence,
                p_value=p_value,
                statistic=statistic,
                timestamp=datetime.now()
            )

            # 履歴を更新
            self.drift_scores.append(statistic)
            self.timestamps.append(result.timestamp)

            # 履歴サイズを制限
            if len(self.drift_scores) > self.config.max_history_size:
                self.drift_scores.pop(0)
                self.timestamps.pop(0)

            return result

        except Exception as e:
            logger.error(f"KS test failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                confidence=0.0,
                p_value=1.0,
                statistic=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )

    def _calculate_severity(self, statistic: float, p_value: float) -> DriftSeverity:
        """重大度を計算"""
        if p_value > 0.1:
            return DriftSeverity.NONE
        elif statistic > 0.3 or p_value < 0.001:
            return DriftSeverity.CRITICAL
        elif statistic > 0.2 or p_value < 0.01:
            return DriftSeverity.HIGH
        elif statistic > 0.1 or p_value < 0.05:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW


class ADWINDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) によるドリフト検知"""

    def __init__(self, config: ConceptDriftConfig):
        super().__init__(config)
        self.delta = config.adwin_delta
        self.window_data: List[float] = []
        self.window_size = 0
        self.total = 0.0
        self.variance = 0.0

    def detect_drift(self, new_data: np.ndarray) -> DriftDetectionResult:
        drift_detected = False
        cut_point = -1

        # 新しいデータを追加
        for value in new_data:
            self._add_element(value)

            # ウィンドウ分割をチェック
            cut_point = self._find_cut_point()
            if cut_point >= 0:
                drift_detected = True
                # ドリフト検知時は古いデータを削除
                self._remove_elements(cut_point)
                break

        severity = DriftSeverity.HIGH if drift_detected else DriftSeverity.NONE

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.CONCEPT_DRIFT if drift_detected else DriftType.NONE,
            severity=severity,
            confidence=0.8 if drift_detected else 1.0,
            p_value=0.0 if drift_detected else 1.0,
            statistic=float(cut_point) if cut_point >= 0 else 0.0,
            timestamp=datetime.now()
        )

        # 履歴を更新
        self.drift_scores.append(result.statistic or 0.0)
        self.timestamps.append(result.timestamp)

        if len(self.drift_scores) > self.config.max_history_size:
            self.drift_scores.pop(0)
            self.timestamps.pop(0)

        return result

    def _add_element(self, value: float):
        """要素を追加"""
        self.window_data.append(value)
        self.window_size += 1
        self.total += value
        # 分散の更新（オンラインアルゴリズム）
        if self.window_size > 1:
            old_mean = (self.total - value) / (self.window_size - 1)
            new_mean = self.total / self.window_size
            self.variance += (value - old_mean) * (value - new_mean)

    def _find_cut_point(self) -> int:
        """カットポイントを見つける"""
        if self.window_size < 2:
            return -1

        min_cut = -1
        min_epsilon = float('inf')

        for i in range(1, self.window_size):
            epsilon = self._calculate_epsilon(i)
            if epsilon < min_epsilon:
                min_epsilon = epsilon
                min_cut = i

        # 閾値チェック
        if min_epsilon > self.delta:
            return -1

        return min_cut

    def _calculate_epsilon(self, cut_point: int) -> float:
        """イプシロンを計算"""
        n0 = cut_point
        n1 = self.window_size - cut_point

        if n0 == 0 or n1 == 0:
            return float('inf')

        # 平均の計算
        sum0 = sum(self.window_data[:cut_point])
        sum1 = sum(self.window_data[cut_point:])

        mu0 = sum0 / n0
        mu1 = sum1 / n1

        # 分散の計算
        var0 = np.var(self.window_data[:cut_point]) if n0 > 1 else 0
        var1 = np.var(self.window_data[cut_point:]) if n1 > 1 else 0

        # ADWINのイプシロン計算
        epsilon = abs(mu0 - mu1) - self._calculate_confidence_interval(n0, n1, var0, var1)

        return epsilon

    def _calculate_confidence_interval(self, n0: int, n1: int, var0: float, var1: float) -> float:
        """信頼区間を計算"""
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        delta_prime = self.delta / (self.window_size * np.log(self.window_size))

        return np.sqrt(2 * m * (var0 / n0 + var1 / n1) * np.log(1.0 / delta_prime))

    def _remove_elements(self, cut_point: int):
        """要素を削除"""
        self.window_data = self.window_data[cut_point:]
        self.window_size = len(self.window_data)
        self.total = sum(self.window_data)
        # 分散の再計算
        self.variance = np.var(self.window_data) * self.window_size if self.window_size > 1 else 0.0


class DDMDetector(DriftDetector):
    """DDM (Drift Detection Method)"""

    def __init__(self, config: ConceptDriftConfig):
        super().__init__(config)
        self.min_error_rate = float('inf')
        self.error_count = 0
        self.sample_count = 0
        self.warning_level = 0
        self.drift_level = 0

    def detect_drift(self, new_data: np.ndarray, error_data: Optional[np.ndarray] = None) -> DriftDetectionResult:
        """
        DDM検知
        new_data: 予測値または特徴量
        error_data: エラー指標（指定されない場合はランダムに生成）
        """
        if error_data is None:
            # 簡易的なエラー生成（実際の実装では真のエラー値を使用）
            error_data = np.random.random(len(new_data)) < 0.1  # 10%エラー率

        drift_detected = False
        warning_detected = False

        for error in error_data:
            self.sample_count += 1
            if error:
                self.error_count += 1

            if self.sample_count >= self.config.ddm_min_samples:
                error_rate = self.error_count / self.sample_count

                # 最小エラー率を更新
                if error_rate < self.min_error_rate:
                    self.min_error_rate = error_rate

                # DDM統計量の計算
                p = error_rate
                s = np.sqrt(p * (1 - p) / self.sample_count)

                # 警告レベルとドリフトレベルの計算
                self.warning_level = p + s
                self.drift_level = p + 2 * s

                # ドリフト検知
                if error_rate > self.drift_level:
                    drift_detected = True
                    break
                elif error_rate > self.warning_level:
                    warning_detected = True

        severity = (DriftSeverity.HIGH if drift_detected
                   else DriftSeverity.MEDIUM if warning_detected
                   else DriftSeverity.NONE)

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.CONCEPT_DRIFT if drift_detected else DriftType.NONE,
            severity=severity,
            confidence=0.9 if drift_detected else 0.5 if warning_detected else 1.0,
            p_value=0.0 if drift_detected else 0.5 if warning_detected else 1.0,
            statistic=self.drift_level,
            timestamp=datetime.now()
        )

        # 履歴を更新
        self.drift_scores.append(result.statistic or 0.0)
        self.timestamps.append(result.timestamp)

        if len(self.drift_scores) > self.config.max_history_size:
            self.drift_scores.pop(0)
            self.timestamps.pop(0)

        return result


class EDDMDetector(DriftDetector):
    """EDDM (Early Drift Detection Method)"""

    def __init__(self, config: ConceptDriftConfig):
        super().__init__(config)
        self.alert_distance = 0.95
        self.warning_distance = 0.9
        self.max_distance = 0.0
        self.last_max_distance = 0.0
        self.alert_level = 0.0
        self.warning_level = 0.0
        self.distance_list: List[float] = []

    def detect_drift(self, new_data: np.ndarray, error_data: Optional[np.ndarray] = None) -> DriftDetectionResult:
        """
        EDDM検知
        error_data: エラー指標（1: エラー, 0: 正解）
        """
        if error_data is None:
            error_data = np.random.randint(0, 2, len(new_data))

        assert error_data is not None  # 型チェッカー用

        drift_detected = False
        warning_detected = False

        for i, error in enumerate(error_data):
            if error == 0:  # 正解の場合
                self.distance_list.append(i + 1)  # 1-indexed

                if len(self.distance_list) >= 2:
                    # 距離の平均と標準偏差を計算
                    distances = np.array(self.distance_list[-self.config.eddm_window_size:])
                    if len(distances) >= 2:
                        mean_distance = np.mean(distances)
                        std_distance = np.std(distances)

                        if mean_distance + 2 * std_distance > self.max_distance:
                            self.max_distance = mean_distance + 2 * std_distance

                        # EDDM統計量
                        if self.max_distance > 0:
                            relative_distance = mean_distance / self.max_distance

                            if relative_distance < self.alert_distance:
                                drift_detected = True
                                break
                            elif relative_distance < self.warning_distance:
                                warning_detected = True

        severity = (DriftSeverity.HIGH if drift_detected
                   else DriftSeverity.MEDIUM if warning_detected
                   else DriftSeverity.NONE)

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.CONCEPT_DRIFT if drift_detected else DriftType.NONE,
            severity=severity,
            confidence=0.9 if drift_detected else 0.6 if warning_detected else 1.0,
            p_value=0.0 if drift_detected else 0.4 if warning_detected else 1.0,
            statistic=self.max_distance,
            timestamp=datetime.now()
        )

        # 履歴を更新
        self.drift_scores.append(result.statistic or 0.0)
        self.timestamps.append(result.timestamp)

        if len(self.drift_scores) > self.config.max_history_size:
            self.drift_scores.pop(0)
            self.timestamps.pop(0)

        return result