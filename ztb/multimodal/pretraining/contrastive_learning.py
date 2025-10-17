"""
Contrastive Learning for Time Series - SimCLR-style Self-supervised Learning
時系列データ向けコントラスト学習 - SimCLRスタイル自己教師あり学習

This module implements contrastive learning for financial time series data,
where the model learns to distinguish between similar and dissimilar time series
through data augmentation and contrastive loss.

実装内容:
- 時系列データのデータ拡張（時間シフト、ノイズ追加、スケーリング）
- SimCLRスタイルのコントラスト損失
- 表現学習を通じた金融時系列の特徴抽出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ..core.encoders import PriceEncoder
from ztb.trading.environment.components.memory_manager import MemoryManager


class TimeSeriesAugmentation:
    """
    Data augmentation techniques for time series data
    時系列データ向けデータ拡張手法
    """

    def __init__(self,
                 shift_prob: float = 0.5,
                 noise_prob: float = 0.3,
                 scale_prob: float = 0.2,
                 max_shift: int = 5,
                 noise_std: float = 0.1,
                 scale_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initialize augmentation parameters

        Args:
            shift_prob: 時間シフト適用確率
            noise_prob: ノイズ追加適用確率
            scale_prob: スケーリング適用確率
            max_shift: 最大シフト量
            noise_std: ノイズ標準偏差
            scale_range: スケーリング範囲
        """
        self.shift_prob = shift_prob
        self.noise_prob = noise_prob
        self.scale_prob = scale_prob
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to time series

        Args:
            x: 入力時系列 (batch_size, seq_len, features)

        Returns:
            拡張された時系列
        """
        augmented = x.clone()

        # Random time shift
        if torch.rand(1) < self.shift_prob:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            if shift > 0:
                augmented = torch.cat([
                    augmented[:, shift:, :],
                    augmented[:, :shift, :]
                ], dim=1)
            elif shift < 0:
                augmented = torch.cat([
                    augmented[:, -shift:, :],
                    augmented[:, :-shift, :]
                ], dim=1)

        # Add random noise
        if torch.rand(1) < self.noise_prob:
            noise = torch.randn_like(augmented) * self.noise_std
            augmented = augmented + noise

        # Random scaling
        if torch.rand(1) < self.scale_prob:
            scale = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            augmented = augmented * scale

        return augmented


class ContrastiveLearningModel(nn.Module):
    """
    Contrastive Learning model for time series representation learning
    時系列表現学習向けコントラスト学習モデル
    """

    def __init__(self,
                 input_dim: int = 156,
                 hidden_dim: int = 512,
                 projection_dim: int = 128,
                 temperature: float = 0.5):
        """
        Initialize Contrastive Learning Model

        Args:
            input_dim: 入力特徴量次元
            hidden_dim: 隠れ層次元
            projection_dim: 射影次元
            temperature: 温度パラメータ
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature

        # Encoder (shared for both views)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to representations

        Args:
            x: 入力時系列 (batch_size, seq_len, input_dim)

        Returns:
            表現ベクトル (batch_size, hidden_dim)
        """
        # Global average pooling across time dimension
        pooled = x.mean(dim=1)  # (batch_size, input_dim)

        # Encode
        encoded = self.encoder(pooled)  # (batch_size, hidden_dim)
        return encoded

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project representations to lower dimensional space

        Args:
            h: 表現ベクトル (batch_size, hidden_dim)

        Returns:
            射影ベクトル (batch_size, projection_dim)
        """
        return self.projection_head(h)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning

        Args:
            x1: 第一ビュー (batch_size, seq_len, input_dim)
            x2: 第二ビュー (batch_size, seq_len, input_dim)

        Returns:
            両ビューの射影ベクトル
        """
        # Encode both views
        h1 = self.encode(x1)
        h2 = self.encode(x2)

        # Project to lower dimension
        z1 = self.project(h1)
        z2 = self.project(h2)

        return z1, z2

    def compute_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss (Normalized Temperature-scaled Cross Entropy)

        Args:
            z1: 第一ビューの射影ベクトル
            z2: 第二ビューの射影ベクトル

        Returns:
            コントラスト損失
        """
        batch_size = z1.shape[0]

        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create labels (positive pairs are at diagonal positions)
        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels + batch_size) % (2 * batch_size)

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class ContrastiveLearningTrainer:
    """
    Trainer for Contrastive Learning
    コントラスト学習のトレーナー
    """

    def __init__(self,
                 model: ContrastiveLearningModel,
                 optimizer: torch.optim.Optimizer,
                 augmentation: TimeSeriesAugmentation,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize trainer

        Args:
            model: ContrastiveLearningModel instance
            optimizer: Optimizer
            augmentation: データ拡張インスタンス
            device: デバイス
            memory_manager: メモリマネージャー（オプション）
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.augmentation = augmentation
        self.device = device
        self.memory_manager = memory_manager or MemoryManager(memory_logging_enabled=False)
        self.step_counter = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with data augmentation

        Args:
            batch: バッチデータ (batch_size, seq_len, input_dim)

        Returns:
            メトリクス辞書
        """
        self.model.train()
        batch = batch.to(self.device)

        # Create two augmented views
        x1 = self.augmentation(batch)
        x2 = self.augmentation(batch)

        # Forward pass
        z1, z2 = self.model(x1, x2)

        # Compute contrastive loss
        loss = self.model.compute_contrastive_loss(z1, z2)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Memory management: periodic GPU cache clearing
        self.step_counter += 1
        if self.step_counter % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'loss': loss.item(),
            'z1_norm': z1.norm(dim=1).mean().item(),
            'z2_norm': z2.norm(dim=1).mean().item()
        }

    def validate(self, val_data: torch.Tensor, batch_size: int = 32) -> Dict[str, float]:
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
                batch = val_data[i:i+batch_size].to(self.device)

                # Create two augmented views
                x1 = self.augmentation(batch)
                x2 = self.augmentation(batch)

                z1, z2 = self.model(x1, x2)
                loss = self.model.compute_contrastive_loss(z1, z2)

                total_loss += loss.item()
                num_batches += 1

        return {
            'val_loss': total_loss / num_batches
        }

    def get_embeddings(self, data: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Get embeddings for data (without augmentation)

        Args:
            data: 入力データ
            batch_size: バッチサイズ

        Returns:
            エンベディング (num_samples, hidden_dim)
        """
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size].to(self.device)
                emb = self.model.encode(batch)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])