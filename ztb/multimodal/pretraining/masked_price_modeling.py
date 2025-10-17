"""
Masked Price Modeling (MPM) - BERT-style Self-supervised Learning for Financial Data
金融データ向けBERT-style自己教師あり学習

This module implements masked price modeling where random price features are masked
and the model learns to predict the masked values, similar to BERT's masked language modeling.

実装内容:
- 価格特徴量のランダムマスキング
- Transformerベースの予測モデル
- 金融時系列データの文脈理解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ..core.encoders import PriceEncoder
from ztb.trading.environment.components.memory_manager import MemoryManager


class MaskedPriceModel(nn.Module):
    """
    Masked Price Modeling model for financial time series data.
    金融時系列データ向けマスク価格モデリングモデル
    """

    def __init__(self,
                 input_dim: int = 156,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 100,
                 mask_prob: float = 0.15):
        """
        Initialize Masked Price Model

        Args:
            input_dim: 入力特徴量次元
            hidden_dim: 隠れ層次元
            num_layers: Transformer層数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
            max_seq_len: 最大シーケンス長
            mask_prob: マスキング確率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection for masked prediction
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(input_dim))

    def _create_masks(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Create random masks for input sequences
        入力シーケンスのランダムマスクを作成

        Args:
            batch_size: バッチサイズ
            seq_len: シーケンス長

        Returns:
            マスクテンソル (batch_size, seq_len)
        """
        # Create random mask
        mask = torch.rand(batch_size, seq_len) < self.mask_prob
        return mask

    def forward(self,
                x: torch.Tensor,
                mask_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional masking

        Args:
            x: 入力テンソル (batch_size, seq_len, input_dim)
            mask_indices: マスクインデックス (batch_size, seq_len)

        Returns:
            予測値とマスクインデックス
        """
        batch_size, seq_len, _ = x.shape

        # Create masks if not provided
        if mask_indices is None:
            mask_indices = self._create_masks(batch_size, seq_len)

        # Apply masking
        masked_x = x.clone()
        for b in range(batch_size):
            for t in range(seq_len):
                if mask_indices[b, t]:
                    masked_x[b, t] = self.mask_token

        # Input projection
        x_proj = self.input_projection(masked_x)

        # Add positional encoding
        x_proj = x_proj + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        encoded = self.transformer_encoder(x_proj)

        # Output projection
        predictions = self.output_projection(encoded)

        return predictions, mask_indices

    def compute_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    mask_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute masked prediction loss
        マスク予測損失を計算

        Args:
            predictions: 予測値 (batch_size, seq_len, input_dim)
            targets: 正解値 (batch_size, seq_len, input_dim)
            mask_indices: マスクインデックス (batch_size, seq_len)

        Returns:
            損失値
        """
        # Only compute loss for masked positions
        masked_predictions = predictions[mask_indices]
        masked_targets = targets[mask_indices]

        if len(masked_predictions) == 0:
            return torch.tensor(0.0, device=predictions.device)

        # MSE loss for price prediction
        loss = F.mse_loss(masked_predictions, masked_targets)
        return loss


class MaskedPriceModelingTrainer:
    """
    Trainer for Masked Price Modeling
    マスク価格モデリングのトレーナー
    """

    def __init__(self,
                 model: MaskedPriceModel,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize trainer

        Args:
            model: MaskedPriceModel instance
            optimizer: Optimizer
            device: デバイス
            memory_manager: メモリマネージャー（オプション）
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.memory_manager = memory_manager or MemoryManager(memory_logging_enabled=False)
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
        batch = batch.to(self.device)

        # Forward pass
        predictions, mask_indices = self.model(batch)

        # Compute loss
        loss = self.model.compute_loss(predictions, batch, mask_indices)

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
            'masked_ratio': mask_indices.float().mean().item()
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
        total_masked = 0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size].to(self.device)

                predictions, mask_indices = self.model(batch)
                loss = self.model.compute_loss(predictions, batch, mask_indices)

                total_loss += loss.item()
                total_masked += mask_indices.sum().item()
                num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_masked_ratio': total_masked / (len(val_data) * val_data.shape[1])
        }

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