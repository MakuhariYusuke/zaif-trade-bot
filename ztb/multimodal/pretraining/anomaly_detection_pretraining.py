"""
Time Series Anomaly Detection Pre-training
時系列異常検知事前学習

This module implements self-supervised pre-training for time series anomaly detection
using reconstruction-based and prediction-based approaches.

実装内容:
- 再構成ベースの異常検知（オートエンコーダー）
- 予測ベースの異常検知（未来予測）
- ハイブリッドアプローチの統合
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
try:
    import torch.nn.functional as F
except Exception:
    # Provide minimal fallback for functional APIs used during test collection
    class _F:
        @staticmethod
        def relu(x):
            return x

        @staticmethod
        def mse_loss(a, b):
            return 0

    F = _F

from ztb.trading.environment.components.memory_manager import MemoryManager


class ReconstructionAnomalyDetector(nn.Module):
    """
    Reconstruction-based anomaly detection using autoencoder
    オートエンコーダーによる再構成ベース異常検知
    """

    def __init__(
        self,
        input_dim: int = 156,
        hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 32,
        seq_len: int = 100,
    ) -> None:
        """
        Initialize Reconstruction Anomaly Detector

        Args:
            input_dim: 入力特徴量次元
            hidden_dims: エンコーダー/デコーダーの隠れ層次元
            latent_dim: 潜在空間次元
            seq_len: シーケンス長
        """
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        # Latent layer
        encoder_layers.extend([nn.Linear(prev_dim, latent_dim), nn.ReLU()])

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space

        Args:
            x: 入力 (batch_size, seq_len, input_dim)

        Returns:
            潜在表現 (batch_size, seq_len, latent_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # Reshape for linear layers
        x_flat = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)

        # Encode
        encoded_flat = self.encoder(x_flat)  # (batch_size * seq_len, latent_dim)

        # Reshape back
        encoded = encoded_flat.view(batch_size, seq_len, -1)
        return encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space

        Args:
            z: 潜在表現 (batch_size, seq_len, latent_dim)

        Returns:
            再構成出力 (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, latent_dim = z.shape

        # Reshape for linear layers
        z_flat = z.view(-1, latent_dim)  # (batch_size * seq_len, latent_dim)

        # Decode
        decoded_flat = self.decoder(z_flat)  # (batch_size * seq_len, input_dim)

        # Reshape back
        decoded = decoded_flat.view(batch_size, seq_len, self.input_dim)
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode -> decode

        Args:
            x: 入力 (batch_size, seq_len, input_dim)

        Returns:
            再構成出力 (batch_size, seq_len, input_dim)
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed

    def compute_reconstruction_loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE)

        Args:
            x: 元の入力
            reconstructed: 再構成出力

        Returns:
            再構成損失
        """
        return F.mse_loss(reconstructed, x)

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score based on reconstruction error

        Args:
            x: 入力データ

        Returns:
            異常スコア (batch_size,)
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Compute MSE for each sample
            mse_per_sample = F.mse_loss(reconstructed, x, reduction="none")
            mse_per_sample = mse_per_sample.mean(
                dim=(1, 2)
            )  # Average over seq_len and features
            return mse_per_sample


class PredictionAnomalyDetector(nn.Module):
    """
    Prediction-based anomaly detection using LSTM
    LSTMによる予測ベース異常検知
    """

    def __init__(
        self,
        input_dim: int = 156,
        hidden_dim: int = 128,
        num_layers: int = 2,
        prediction_horizon: int = 1,
    ) -> None:
        """
        Initialize Prediction Anomaly Detector

        Args:
            input_dim: 入力特徴量次元
            hidden_dim: LSTM隠れ層次元
            num_layers: LSTM層数
            prediction_horizon: 予測ホライズン
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict next time step

        Args:
            x: 入力シーケンス (batch_size, seq_len, input_dim)

        Returns:
            予測値 (batch_size, input_dim)
        """
        # Encode sequence
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers, batch_size, hidden_dim)

        # Use last layer's hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Predict next time step
        prediction = self.predictor(last_hidden)  # (batch_size, input_dim)
        return prediction

    def compute_prediction_loss(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction loss

        Args:
            x: 入力シーケンス
            target: 正解値 (batch_size, input_dim)

        Returns:
            予測損失
        """
        prediction = self.forward(x)
        return F.mse_loss(prediction, target)

    def compute_anomaly_score(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomaly score based on prediction error

        Args:
            x: 入力シーケンス
            target: 正解値

        Returns:
            異常スコア (batch_size,)
        """
        with torch.no_grad():
            prediction = self.forward(x)
            mse_per_sample = F.mse_loss(prediction, target, reduction="none")
            mse_per_sample = mse_per_sample.mean(dim=1)  # Average over features
            return mse_per_sample


class HybridAnomalyDetector(nn.Module):
    """
    Hybrid anomaly detection combining reconstruction and prediction
    再構成と予測を組み合わせたハイブリッド異常検知
    """

    def __init__(
        self,
        input_dim: int = 156,
        hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        seq_len: int = 100,
        alpha: float = 0.5,
    ) -> None:
        """
        Initialize Hybrid Anomaly Detector

        Args:
            input_dim: 入力特徴量次元
            hidden_dims: オートエンコーダーの隠れ層次元
            latent_dim: 潜在空間次元
            lstm_hidden_dim: LSTM隠れ層次元
            lstm_num_layers: LSTM層数
            seq_len: シーケンス長
            alpha: 再構成と予測の重み付け (0: reconstruction only, 1: prediction only)
        """
        super().__init__()

        self.alpha = alpha

        # Reconstruction-based detector
        self.reconstruction_detector = ReconstructionAnomalyDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            seq_len=seq_len,
        )

        # Prediction-based detector
        self.prediction_detector = PredictionAnomalyDetector(
            input_dim=input_dim, hidden_dim=lstm_hidden_dim, num_layers=lstm_num_layers
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both detectors

        Args:
            x: 入力シーケンス (batch_size, seq_len, input_dim)

        Returns:
            再構成出力, 予測値
        """
        reconstructed = self.reconstruction_detector(x)

        # For prediction, use all but last time step as input
        pred_input = x[:, :-1, :]  # (batch_size, seq_len-1, input_dim)
        prediction = self.prediction_detector(pred_input)

        return reconstructed, prediction

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss

        Args:
            x: 入力シーケンス

        Returns:
            複合損失
        """
        reconstructed, prediction = self.forward(x)

        # Reconstruction loss
        recon_loss = self.reconstruction_detector.compute_reconstruction_loss(
            x, reconstructed
        )

        # Prediction loss (predict last time step)
        target = x[:, -1, :]  # Last time step
        pred_input = x[:, :-1, :]
        pred_loss = self.prediction_detector.compute_prediction_loss(pred_input, target)

        # Combined loss
        total_loss = self.alpha * pred_loss + (1 - self.alpha) * recon_loss
        return total_loss

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid anomaly score

        Args:
            x: 入力シーケンス

        Returns:
            異常スコア (batch_size,)
        """
        with torch.no_grad():
            reconstructed, prediction = self.forward(x)

            # Reconstruction anomaly score
            recon_score = self.reconstruction_detector.compute_anomaly_score(x)

            # Prediction anomaly score
            target = x[:, -1, :]
            pred_input = x[:, :-1, :]
            pred_score = self.prediction_detector.compute_anomaly_score(
                pred_input, target
            )

            # Combine scores
            hybrid_score = self.alpha * pred_score + (1 - self.alpha) * recon_score
            return hybrid_score


class AnomalyDetectionPretrainer:
    """
    Pre-trainer for anomaly detection models
    異常検知モデルの事前学習トレーナー
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize pre-trainer

        Args:
            model: Anomaly detection model
            optimizer: Optimizer
            device: デバイス
            memory_manager: メモリマネージャー（オプション）
        """
        # Avoid moving model to CPU explicitly to prevent CUDA lazy init
        try:
            if device != "cpu":
                self.model = model.to(device)
            else:
                self.model = model
        except Exception:
            self.model = model
        self.optimizer = optimizer
        self.device = device
        self.memory_manager = memory_manager or MemoryManager(
            memory_logging_enabled=False
        )
        self.step_counter = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step

        Args:
            batch: バッチデータ (batch_size, seq_len, input_dim)

        Returns:
            メトリクス辞書
        """
        self.model.train()
        if self.device != "cpu":
            try:
                batch = batch.to(self.device)
            except Exception:
                pass

        # Compute loss
        loss = self.model.compute_loss(batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Memory management: periodic GPU cache clearing
        self.step_counter += 1
        if self.step_counter % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"loss": loss.item()}

    def validate(
        self, val_data: torch.Tensor, batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Validation on dataset

        Args:
            val_data: 検証データ
            batch_size: バッチサイズ

        Returns:
            検証メトリクス
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]
                if self.device != "cpu":
                    try:
                        batch = batch.to(self.device)
                    except Exception:
                        pass
                loss = self.model.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def compute_anomaly_scores(
        self, data: torch.Tensor, batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute anomaly scores for dataset

        Args:
            data: 入力データ
            batch_size: バッチサイズ

        Returns:
            異常スコア (num_samples,)
        """
        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                if self.device != "cpu":
                    try:
                        batch = batch.to(self.device)
                    except Exception:
                        pass
                scores = self.model.compute_anomaly_score(batch)
                all_scores.append(scores.cpu())

        return torch.cat(all_scores, dim=0)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
