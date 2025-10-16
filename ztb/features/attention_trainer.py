#!/usr/bin/env python3
"""
Attention Model Training for Adaptive Feature Selection
適応的特徴選択のための注意モデルトレーニング

実装内容:
- Attention modelの学習システム
- トレーニングデータ収集と管理
- メモリ効率的なトレーニング
- 特徴量重要度の動的学習
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ztb.utils.logging_utils import get_logger
from ztb.utils.memory.dtypes import optimize_dtypes
from ztb.utils.path_utils import ensure_dir

logger = get_logger(__name__)


class FeatureAttentionLayer(nn.Module):
    """Attention-based feature weighting layer"""

    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.n_features = n_features

        # Attention layers
        self.query_proj = nn.Linear(n_features, hidden_dim)
        self.key_proj = nn.Linear(n_features, hidden_dim)
        self.value_proj = nn.Linear(n_features, hidden_dim)

        self.attention_dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features)

        Returns:
            attention_weights: (batch_size, n_features)
        """
        # Query, Key, Value
        Q = self.query_proj(x)  # (batch, hidden)
        K = self.key_proj(x)    # (batch, hidden)
        V = self.value_proj(x)  # (batch, hidden)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.n_features ** 0.5)  # (batch, batch)
        attention_weights = F.softmax(scores, dim=-1)  # (batch, batch)

        # Apply attention
        attended = torch.matmul(attention_weights, V)  # (batch, hidden)
        attended = self.attention_dropout(attended)

        # Output projection
        output = self.output_proj(attended)  # (batch, n_features)
        weights = torch.sigmoid(output)  # (batch, n_features)

        return weights


class AttentionTrainingDataset(Dataset):
    """注意モデルトレーニング用データセット"""

    def __init__(self, feature_data: np.ndarray, rewards: np.ndarray, regimes: np.ndarray):
        """
        Args:
            feature_data: 特徴量データ (n_samples, n_features)
            rewards: 報酬データ (n_samples,)
            regimes: 市場状態ラベル (n_samples,)
        """
        self.feature_data = torch.FloatTensor(feature_data)
        self.rewards = torch.FloatTensor(rewards)
        self.regimes = torch.LongTensor(regimes)

    def __len__(self) -> int:
        return len(self.feature_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.feature_data[idx], self.rewards[idx], self.regimes[idx]


class AttentionTrainer:
    """注意モデルトレーニングマネージャー"""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 50,
        patience: int = 10,
        model_save_path: Optional[str] = None,
        memory_manager=None,
    ):
        """
        Args:
            n_features: 特徴量数
            hidden_dim: 注意レイヤーの隠れ層次元
            learning_rate: 学習率
            batch_size: バッチサイズ
            max_epochs: 最大エポック数
            patience: 早期停止の忍耐回数
            model_save_path: モデル保存パス
            memory_manager: メモリマネージャー
        """
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.memory_manager = memory_manager

        # モデル初期化
        self.model = FeatureAttentionLayer(n_features, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # トレーニング履歴
        self.training_history: List[Dict[str, float]] = []

        # モデル保存パス
        self.model_save_path = Path(model_save_path) if model_save_path else None
        if self.model_save_path:
            ensure_dir(self.model_save_path.parent)

        # トレーニングデータ
        self.feature_buffer: List[np.ndarray] = []
        self.reward_buffer: List[float] = []
        self.regime_buffer: List[int] = []

        logger.info(f"Initialized AttentionTrainer with {n_features} features, hidden_dim={hidden_dim}")

    def add_training_sample(
        self,
        features: np.ndarray,
        reward: float,
        regime: Union[str, int]
    ) -> None:
        """
        トレーニングサンプルを追加

        Args:
            features: 特徴量ベクトル
            reward: 報酬値
            regime: 市場状態 (文字列または整数)
        """
        if len(self.feature_buffer) >= 10000:  # バッファサイズ制限
            # 古いデータを削除してメモリ節約
            remove_count = len(self.feature_buffer) // 4
            self.feature_buffer = self.feature_buffer[remove_count:]
            self.reward_buffer = self.reward_buffer[remove_count:]
            self.regime_buffer = self.regime_buffer[remove_count:]

            # メモリ解放とGC
            gc.collect()
            if self.memory_manager:
                self.memory_manager.log_memory_usage("attention_trainer_buffer_trim")

        self.feature_buffer.append(features.copy())
        self.reward_buffer.append(reward)

        # 市場状態を整数に変換
        if isinstance(regime, str):
            regime_map = {
                "trending": 0,
                "ranging": 1,
                "high_volatility": 2,
                "low_volatility": 3
            }
            regime_int = regime_map.get(regime, 0)
        else:
            regime_int = int(regime)

        self.regime_buffer.append(regime_int)

    def has_enough_data(self, min_samples: int = 100) -> bool:
        """十分なトレーニングデータがあるか確認"""
        return len(self.feature_buffer) >= min_samples

    def prepare_dataset(self) -> Optional[AttentionTrainingDataset]:
        """トレーニングデータセットを作成"""
        if not self.has_enough_data():
            logger.warning(f"Insufficient training data: {len(self.feature_buffer)} samples")
            return None

        try:
            feature_data = np.array(self.feature_buffer)
            reward_data = np.array(self.reward_buffer)
            regime_data = np.array(self.regime_buffer)

            # メモリ最適化
            if self.memory_manager:
                self.memory_manager.log_memory_usage("attention_trainer_dataset_prep")

            dataset = AttentionTrainingDataset(feature_data, reward_data, regime_data)
            logger.info(f"Prepared dataset with {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            return None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """1エポックのトレーニング"""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_features, batch_rewards, batch_regimes in dataloader:
            self.optimizer.zero_grad()

            # 順伝播
            attention_weights = self.model(batch_features)

            # 損失計算 (報酬を予測するように学習)
            # 注意重みと報酬の相関を最大化
            predicted_rewards = torch.sum(attention_weights * batch_features, dim=1)
            loss = self.criterion(predicted_rewards, batch_rewards)

            # 逆伝播
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # メモリ解放
            del batch_features, batch_rewards, batch_regimes, attention_weights, predicted_rewards
            if n_batches % 10 == 0:
                gc.collect()

        avg_loss = epoch_loss / max(n_batches, 1)
        return {"loss": avg_loss, "n_batches": n_batches}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_features, batch_rewards, batch_regimes in dataloader:
                attention_weights = self.model(batch_features)
                predicted_rewards = torch.sum(attention_weights * batch_features, dim=1)
                loss = self.criterion(predicted_rewards, batch_rewards)

                val_loss += loss.item()
                n_batches += 1

                # メモリ解放
                del batch_features, batch_rewards, batch_regimes, attention_weights, predicted_rewards

        avg_loss = val_loss / max(n_batches, 1)
        return {"val_loss": avg_loss, "n_batches": n_batches}

    def train(self, val_split: float = 0.2) -> Dict[str, Any]:
        """
        モデルのトレーニング

        Args:
            val_split: 検証データ分割率

        Returns:
            トレーニング結果
        """
        dataset = self.prepare_dataset()
        if dataset is None:
            return {"success": False, "error": "Insufficient training data"}

        # データ分割
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # データローダー
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # トレーニングループ
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training for {self.max_epochs} epochs with {n_train} train, {n_val} val samples")

        for epoch in range(self.max_epochs):
            # トレーニング
            train_metrics = self.train_epoch(train_loader)

            # 検証
            val_metrics = self.validate(val_loader)

            # 履歴記録
            epoch_metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics
            }
            self.training_history.append(epoch_metrics)

            logger.info(
                f"Epoch {epoch + 1}/{self.max_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )

            # 早期停止判定
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # メモリログ
            if self.memory_manager and epoch % 5 == 0:
                self.memory_manager.log_memory_usage(f"attention_trainer_epoch_{epoch + 1}")

        # 最良モデルを復元
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # モデル保存
        if self.model_save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'n_features': self.n_features,
                'hidden_dim': self.hidden_dim,
            }, self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")

        final_metrics = self.training_history[-1] if self.training_history else {}
        return {
            "success": True,
            "final_metrics": final_metrics,
            "training_history": self.training_history,
            "best_val_loss": best_val_loss
        }

    def get_attention_weights(self, features: np.ndarray) -> np.ndarray:
        """
        特徴量に対する注意重みを取得

        Args:
            features: 特徴量ベクトル

        Returns:
            注意重み配列
        """
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, n_features)
            weights = self.model(features_tensor).squeeze(0).numpy()  # (n_features,)

        return weights

    def load_model(self, model_path: str) -> bool:
        """
        保存されたモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def create_attention_trainer(
    n_features: int,
    config: Optional[Dict[str, Any]] = None,
    memory_manager=None
) -> AttentionTrainer:
    """
    注意モデルトレーナーを作成

    Args:
        n_features: 特徴量数
        config: 設定辞書
        memory_manager: メモリマネージャー

    Returns:
        AttentionTrainerインスタンス
    """
    if config is None:
        config = {}

    return AttentionTrainer(
        n_features=n_features,
        hidden_dim=config.get('hidden_dim', 64),
        learning_rate=config.get('learning_rate', 1e-4),
        batch_size=config.get('batch_size', 32),
        max_epochs=config.get('max_epochs', 50),
        patience=config.get('patience', 10),
        model_save_path=config.get('model_save_path'),
        memory_manager=memory_manager,
    )