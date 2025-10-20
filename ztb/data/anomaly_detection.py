"""
Anomaly Detection for SAC v421
データ品質管理と異常値検知のための包括的システム
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    """異常検知結果"""

    is_anomaly: bool
    anomaly_score: float
    method: str
    confidence: float
    details: Dict[str, Any]


@dataclass
class AnomalyStats:
    """異常検知統計"""

    total_samples: int
    anomaly_count: int
    anomaly_rate: float
    methods_used: List[str]
    detection_history: List[AnomalyResult]


class StatisticalAnomalyDetector:
    """統計的手法による異常検知"""

    def __init__(
        self, method: str = "zscore", threshold: float = 3.0, window_size: int = 100
    ):
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.history = []
        self.scaler = StandardScaler()

    def detect(
        self, data: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> AnomalyResult:
        """異常検知実行"""
        try:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # 履歴更新
            self.history.append(data.flatten())
            if len(self.history) > self.window_size:
                self.history = self.history[-self.window_size :]

            if len(self.history) < 10:  # 十分なデータがない場合
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    method=self.method,
                    confidence=0.5,
                    details={"reason": "insufficient_data"},
                )

            # 統計的異常検知
            if self.method == "zscore":
                return self._zscore_detection(data)
            elif self.method == "iqr":
                return self._iqr_detection(data)
            elif self.method == "mad":
                return self._mad_detection(data)
            else:
                return self._zscore_detection(data)

        except Exception as e:
            logger.warning(f"Statistical anomaly detection failed: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                method=self.method,
                confidence=0.0,
                details={"error": str(e)},
            )

    def _zscore_detection(self, data: np.ndarray) -> AnomalyResult:
        """Z-scoreベースの異常検知"""
        # 履歴データの標準化
        history_array = np.array(self.history[-self.window_size :])
        if len(history_array.shape) == 1:
            history_array = history_array.reshape(-1, 1)

        self.scaler.fit(history_array)
        z_scores = np.abs(self.scaler.transform(data.reshape(1, -1))[0])

        max_z_score = np.max(z_scores)
        is_anomaly = max_z_score > self.threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=max_z_score,
            method="zscore",
            confidence=min(max_z_score / (self.threshold * 2), 1.0),
            details={
                "z_scores": z_scores.tolist(),
                "max_z_score": max_z_score,
                "threshold": self.threshold,
            },
        )

    def _iqr_detection(self, data: np.ndarray) -> AnomalyResult:
        """IQRベースの異常検知"""
        history_flat = np.array(
            [item for sublist in self.history[-self.window_size :] for item in sublist]
        )
        q1, q3 = np.percentile(history_flat, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (self.threshold * iqr)
        upper_bound = q3 + (self.threshold * iqr)

        data_flat = data.flatten()
        outliers = [(x < lower_bound or x > upper_bound) for x in data_flat]
        anomaly_score = sum(outliers) / len(outliers) if outliers else 0.0

        return AnomalyResult(
            is_anomaly=any(outliers),
            anomaly_score=anomaly_score,
            method="iqr",
            confidence=anomaly_score,
            details={
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": sum(outliers),
            },
        )

    def _mad_detection(self, data: np.ndarray) -> AnomalyResult:
        """MAD (Median Absolute Deviation) ベースの異常検知"""
        history_flat = np.array(
            [item for sublist in self.history[-self.window_size :] for item in sublist]
        )
        median = np.median(history_flat)
        mad = np.median(np.abs(history_flat - median))

        if mad == 0:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                method="mad",
                confidence=0.5,
                details={"reason": "zero_mad"},
            )

        data_flat = data.flatten()
        modified_z_scores = 0.6745 * (data_flat - median) / mad
        max_z_score = np.max(np.abs(modified_z_scores))

        return AnomalyResult(
            is_anomaly=max_z_score > self.threshold,
            anomaly_score=max_z_score,
            method="mad",
            confidence=min(max_z_score / (self.threshold * 2), 1.0),
            details={
                "median": median,
                "mad": mad,
                "modified_z_scores": modified_z_scores.tolist(),
                "max_z_score": max_z_score,
            },
        )


class MLAnomalyDetector:
    """機械学習ベースの異常検知"""

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

        # モデル初期化
        if method == "isolation_forest":
            self.model = IsolationForest(
                contamination=contamination, random_state=random_state, n_estimators=100
            )
        elif method == "elliptic_envelope":
            self.model = EllipticEnvelope(
                contamination=contamination, random_state=random_state
            )

    def fit(self, data: np.ndarray) -> bool:
        """モデル学習"""
        try:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            if len(data) < 10:
                logger.warning("Insufficient data for ML anomaly detection training")
                return False

            self.model.fit(data)
            self.is_fitted = True
            return True

        except Exception as e:
            logger.error(f"ML anomaly detector training failed: {e}")
            return False

    def detect(self, data: np.ndarray) -> AnomalyResult:
        """異常検知実行"""
        try:
            if not self.is_fitted:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    method=self.method,
                    confidence=0.0,
                    details={"reason": "model_not_fitted"},
                )

            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # 予測実行
            if self.method == "isolation_forest":
                scores = self.model.decision_function(data)
                predictions = self.model.predict(data)
                # IsolationForest: -1 for outliers, 1 for inliers
                is_anomaly = predictions[0] == -1
                anomaly_score = -scores[0]  # Convert to positive score
            else:  # elliptic_envelope
                scores = self.model.decision_function(data)
                predictions = self.model.predict(data)
                is_anomaly = predictions[0] == -1
                anomaly_score = -scores[0]

            confidence = min(abs(anomaly_score) / 0.5, 1.0)  # Normalize confidence

            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                method=self.method,
                confidence=confidence,
                details={
                    "raw_score": scores[0]
                    if hasattr(scores, "__getitem__")
                    else scores,
                    "prediction": predictions[0]
                    if hasattr(predictions, "__getitem__")
                    else predictions,
                },
            )

        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                method=self.method,
                confidence=0.0,
                details={"error": str(e)},
            )


class AutoencoderAnomalyDetector(nn.Module):
    """オートエンコーダーベースの異常検知"""

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = None, dropout_rate: float = 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 64]

        self.input_dim = input_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[: len(hidden_dims) // 2 + 1]:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        for i, hidden_dim in enumerate(hidden_dims_reversed[:-1]):
            decoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.threshold = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> bool:
        """オートエンコーダー学習"""
        try:
            self.to(device)
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # データ準備
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

            self.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_data in dataloader:
                    batch_data = batch_data[0].to(device)

                    optimizer.zero_grad()
                    outputs = self(batch_data)
                    loss = criterion(outputs, batch_data)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if (epoch + 1) % 20 == 0:
                    logger.info(
                        f"Autoencoder epoch {epoch+1}/{epochs}, loss: {total_loss/len(dataloader):.6f}"
                    )

            # 閾値設定（トレーニングデータの再構成誤差の95パーセンタイル）
            self.eval()
            with torch.no_grad():
                reconstructed = self(data.to(device))
                reconstruction_errors = torch.mean(
                    (reconstructed - data.to(device)) ** 2, dim=1
                )
                self.threshold = torch.quantile(reconstruction_errors, 0.95).item()

            logger.info(
                f"Autoencoder trained successfully, threshold: {self.threshold:.6f}"
            )
            return True

        except Exception as e:
            logger.error(f"Autoencoder training failed: {e}")
            return False

    def detect(self, data: np.ndarray, device: str = "cpu") -> AnomalyResult:
        """異常検知実行"""
        try:
            if self.threshold is None:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    method="autoencoder",
                    confidence=0.0,
                    details={"reason": "model_not_trained"},
                )

            self.eval()
            with torch.no_grad():
                if isinstance(data, np.ndarray):
                    data_tensor = torch.FloatTensor(data).to(device)
                else:
                    data_tensor = data.to(device)

                reconstructed = self(data_tensor)
                reconstruction_error = torch.mean(
                    (reconstructed - data_tensor) ** 2
                ).item()

                is_anomaly = reconstruction_error > self.threshold
                anomaly_score = (
                    reconstruction_error / self.threshold if self.threshold > 0 else 0.0
                )

                return AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    method="autoencoder",
                    confidence=min(anomaly_score, 1.0),
                    details={
                        "reconstruction_error": reconstruction_error,
                        "threshold": self.threshold,
                        "normalized_score": anomaly_score,
                    },
                )

        except Exception as e:
            logger.warning(f"Autoencoder anomaly detection failed: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                method="autoencoder",
                confidence=0.0,
                details={"error": str(e)},
            )


class ComprehensiveAnomalyDetector:
    """包括的な異常検知システム"""

    def __init__(
        self,
        statistical_methods: List[str] = None,
        ml_methods: List[str] = None,
        enable_autoencoder: bool = False,
        voting_threshold: float = 0.5,
    ):
        self.statistical_detectors = {}
        self.ml_detectors = {}
        self.autoencoder_detector = None
        self.voting_threshold = voting_threshold
        self.stats = AnomalyStats(0, 0, 0.0, [], [])

        # デフォルト設定
        if statistical_methods is None:
            statistical_methods = ["zscore", "iqr"]
        if ml_methods is None:
            ml_methods = ["isolation_forest"]

        # 統計的検知器初期化
        for method in statistical_methods:
            if method == "zscore":
                self.statistical_detectors[method] = StatisticalAnomalyDetector(
                    "zscore", threshold=3.0
                )
            elif method == "iqr":
                self.statistical_detectors[method] = StatisticalAnomalyDetector(
                    "iqr", threshold=1.5
                )
            elif method == "mad":
                self.statistical_detectors[method] = StatisticalAnomalyDetector(
                    "mad", threshold=3.5
                )

        # ML検知器初期化
        for method in ml_methods:
            if method == "isolation_forest":
                self.ml_detectors[method] = MLAnomalyDetector(
                    "isolation_forest", contamination=0.1
                )
            elif method == "elliptic_envelope":
                self.ml_detectors[method] = MLAnomalyDetector(
                    "elliptic_envelope", contamination=0.1
                )

        # オートエンコーダー初期化
        if enable_autoencoder:
            self.autoencoder_detector = AutoencoderAnomalyDetector(
                input_dim=10
            )  # 仮の次元

    def fit_ml_detectors(self, training_data: np.ndarray) -> bool:
        """ML検知器の学習"""
        success_count = 0
        total_count = len(self.ml_detectors)

        for name, detector in self.ml_detectors.items():
            if detector.fit(training_data):
                success_count += 1
                logger.info(f"ML detector {name} trained successfully")
            else:
                logger.warning(f"ML detector {name} training failed")

        if self.autoencoder_detector is not None:
            if self.autoencoder_detector.fit(training_data):
                success_count += 1
                logger.info("Autoencoder detector trained successfully")
            else:
                logger.warning("Autoencoder detector training failed")
                total_count += 1

        return success_count >= total_count * 0.5  # 50%以上成功したらOK

    def detect_anomalies(
        self, data: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """包括的な異常検知"""
        self.stats.total_samples += 1

        results = {}
        anomaly_votes = 0
        total_votes = 0

        # 統計的検知
        for name, detector in self.statistical_detectors.items():
            result = detector.detect(data, feature_names)
            results[f"statistical_{name}"] = result
            if result.is_anomaly:
                anomaly_votes += result.confidence
            total_votes += 1

        # ML検知
        for name, detector in self.ml_detectors.items():
            result = detector.detect(data)
            results[f"ml_{name}"] = result
            if result.is_anomaly:
                anomaly_votes += result.confidence
            total_votes += 1

        # オートエンコーダー検知
        if self.autoencoder_detector is not None:
            result = self.autoencoder_detector.detect(data)
            results["autoencoder"] = result
            if result.is_anomaly:
                anomaly_votes += result.confidence
            total_votes += 1

        # 投票による最終判定
        anomaly_score = anomaly_votes / total_votes if total_votes > 0 else 0.0
        is_anomaly = anomaly_score >= self.voting_threshold

        if is_anomaly:
            self.stats.anomaly_count += 1

        self.stats.anomaly_rate = self.stats.anomaly_count / self.stats.total_samples
        self.stats.detection_history.append(results)

        # 履歴サイズ制限
        if len(self.stats.detection_history) > 1000:
            self.stats.detection_history = self.stats.detection_history[-1000:]

        return is_anomaly, {
            "anomaly_score": anomaly_score,
            "voting_threshold": self.voting_threshold,
            "method_results": results,
            "stats": {
                "total_samples": self.stats.total_samples,
                "anomaly_count": self.stats.anomaly_count,
                "anomaly_rate": self.stats.anomaly_rate,
            },
        }

    def get_stats(self) -> AnomalyStats:
        """統計情報取得"""
        return self.stats

    def reset_stats(self):
        """統計情報リセット"""
        self.stats = AnomalyStats(
            0,
            0,
            0.0,
            list(self.statistical_detectors.keys())
            + list(self.ml_detectors.keys())
            + (["autoencoder"] if self.autoencoder_detector else []),
            [],
        )
