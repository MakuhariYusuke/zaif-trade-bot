"""
Self-Supervised Learning Trainer for SAC v421
SAC v421向け自己教師あり学習トレーナー

This module integrates multiple self-supervised learning techniques:
- Masked Price Modeling (MPM)
- Contrastive Learning (SimCLR-style)
- Anomaly Detection Pre-training

実装内容:
- 多様な事前学習手法の統合
- 段階的学習戦略
- 金融時系列データへの適応
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

from .masked_price_modeling import MaskedPriceModel, MaskedPriceModelingTrainer
from .contrastive_learning import ContrastiveLearningModel, ContrastiveLearningTrainer, TimeSeriesAugmentation
from .anomaly_detection_pretraining import HybridAnomalyDetector, AnomalyDetectionPretrainer
from ..core.encoders import PriceEncoder
from ztb.trading.environment.components.memory_manager import MemoryManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SelfSupervisedTrainer:
    """
    Integrated self-supervised learning trainer for financial data
    金融データ向け統合自己教師あり学習トレーナー
    """

    def __init__(self,
                 input_dim: int = 156,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'checkpoints/pretraining',
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize Self-Supervised Trainer

        Args:
            input_dim: 入力特徴量次元
            device: デバイス
            checkpoint_dir: チェックポイント保存ディレクトリ
            memory_manager: メモリマネージャー（オプション）
        """
        self.input_dim = input_dim
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory manager
        self.memory_manager = memory_manager or MemoryManager(
            memory_logging_enabled=True,
            memory_log_path=str(self.checkpoint_dir / 'memory_log.csv')
        )

        # Initialize models
        self.masked_price_model = None
        self.contrastive_model = None
        self.anomaly_model = None

        # Initialize trainers
        self.mpm_trainer = None
        self.cl_trainer = None
        self.ad_trainer = None

        # Training history
        self.training_history = {
            'mpm': {'epochs': [], 'train_loss': [], 'val_loss': []},
            'contrastive': {'epochs': [], 'train_loss': [], 'val_loss': []},
            'anomaly': {'epochs': [], 'train_loss': [], 'val_loss': []}
        }

        logger.info(f"SelfSupervisedTrainer initialized on device: {device}")
        self.memory_manager.log_memory_usage("SelfSupervisedTrainer_init")

    def initialize_masked_price_model(self,
                                    hidden_dim: int = 512,
                                    num_layers: int = 6,
                                    num_heads: int = 8,
                                    dropout: float = 0.1,
                                    max_seq_len: int = 100,
                                    mask_prob: float = 0.15,
                                    learning_rate: float = 1e-4):
        """
        Initialize Masked Price Modeling components
        マスク価格モデリングコンポーネントの初期化
        """
        self.masked_price_model = MaskedPriceModel(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            mask_prob=mask_prob
        )

        optimizer = optim.AdamW(self.masked_price_model.parameters(), lr=learning_rate)
        self.mpm_trainer = MaskedPriceModelingTrainer(
            self.masked_price_model, optimizer, self.device
        )

        logger.info("Masked Price Model initialized")

    def initialize_contrastive_model(self,
                                   hidden_dim: int = 512,
                                   projection_dim: int = 128,
                                   temperature: float = 0.5,
                                   learning_rate: float = 1e-4,
                                   augmentation_params: Optional[Dict] = None):
        """
        Initialize Contrastive Learning components
        コントラスト学習コンポーネントの初期化
        """
        self.contrastive_model = ContrastiveLearningModel(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            temperature=temperature
        )

        optimizer = optim.AdamW(self.contrastive_model.parameters(), lr=learning_rate)

        # Default augmentation parameters
        if augmentation_params is None:
            augmentation_params = {
                'shift_prob': 0.5,
                'noise_prob': 0.3,
                'scale_prob': 0.2,
                'max_shift': 5,
                'noise_std': 0.1,
                'scale_range': (0.8, 1.2)
            }

        augmentation = TimeSeriesAugmentation(**augmentation_params)

        self.cl_trainer = ContrastiveLearningTrainer(
            self.contrastive_model, optimizer, augmentation, self.device
        )

        logger.info("Contrastive Learning Model initialized")

    def initialize_anomaly_model(self,
                               hidden_dims: List[int] = [256, 128, 64],
                               latent_dim: int = 32,
                               lstm_hidden_dim: int = 128,
                               lstm_num_layers: int = 2,
                               seq_len: int = 100,
                               alpha: float = 0.5,
                               learning_rate: float = 1e-4):
        """
        Initialize Anomaly Detection components
        異常検知コンポーネントの初期化
        """
        self.anomaly_model = HybridAnomalyDetector(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            seq_len=seq_len,
            alpha=alpha
        )

        optimizer = optim.AdamW(self.anomaly_model.parameters(), lr=learning_rate)
        self.ad_trainer = AnomalyDetectionPretrainer(
            self.anomaly_model, optimizer, self.device
        )

        logger.info("Anomaly Detection Model initialized")

    def train_masked_price_modeling(self,
                                  train_data: torch.Tensor,
                                  val_data: torch.Tensor,
                                  epochs: int = 100,
                                  batch_size: int = 32,
                                  patience: int = 10,
                                  save_best: bool = True):
        """
        Train Masked Price Modeling
        マスク価格モデリングの学習
        """
        if self.mpm_trainer is None:
            raise ValueError("Masked Price Model not initialized")

        logger.info(f"Starting Masked Price Modeling training for {epochs} epochs")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch_mpm(train_data, batch_size)

            # Validation
            val_metrics = self.mpm_trainer.validate(val_data, batch_size)

            # Log progress
            logger.info(f"MPM Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Record history
            self.training_history['mpm']['epochs'].append(epoch + 1)
            self.training_history['mpm']['train_loss'].append(train_metrics['loss'])
            self.training_history['mpm']['val_loss'].append(val_metrics['val_loss'])

            # Memory management: log usage and force GC every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.memory_manager.log_memory_usage(f"MPM_epoch_{epoch+1}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('mpm_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.memory_manager.log_memory_usage("MPM_training_complete")

        logger.info("Masked Price Modeling training completed")

    def train_contrastive_learning(self,
                                 train_data: torch.Tensor,
                                 val_data: torch.Tensor,
                                 epochs: int = 100,
                                 batch_size: int = 32,
                                 patience: int = 10,
                                 save_best: bool = True):
        """
        Train Contrastive Learning
        コントラスト学習の学習
        """
        if self.cl_trainer is None:
            raise ValueError("Contrastive Learning Model not initialized")

        logger.info(f"Starting Contrastive Learning training for {epochs} epochs")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch_cl(train_data, batch_size)

            # Validation
            val_metrics = self.cl_trainer.validate(val_data, batch_size)

            # Log progress
            logger.info(f"CL Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Record history
            self.training_history['contrastive']['epochs'].append(epoch + 1)
            self.training_history['contrastive']['train_loss'].append(train_metrics['loss'])
            self.training_history['contrastive']['val_loss'].append(val_metrics['val_loss'])

            # Memory management: log usage and force GC every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.memory_manager.log_memory_usage(f"CL_epoch_{epoch+1}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('cl_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.memory_manager.log_memory_usage("CL_training_complete")

        logger.info("Contrastive Learning training completed")

    def train_anomaly_detection(self,
                              train_data: torch.Tensor,
                              val_data: torch.Tensor,
                              epochs: int = 100,
                              batch_size: int = 32,
                              patience: int = 10,
                              save_best: bool = True):
        """
        Train Anomaly Detection
        異常検知の学習
        """
        if self.ad_trainer is None:
            raise ValueError("Anomaly Detection Model not initialized")

        logger.info(f"Starting Anomaly Detection training for {epochs} epochs")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch_ad(train_data, batch_size)

            # Validation
            val_metrics = self.ad_trainer.validate(val_data, batch_size)

            # Log progress
            logger.info(f"AD Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Record history
            self.training_history['anomaly']['epochs'].append(epoch + 1)
            self.training_history['anomaly']['train_loss'].append(train_metrics['loss'])
            self.training_history['anomaly']['val_loss'].append(val_metrics['val_loss'])

            # Memory management: log usage and force GC every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.memory_manager.log_memory_usage(f"AD_epoch_{epoch+1}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('ad_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.memory_manager.log_memory_usage("AD_training_complete")

        logger.info("Anomaly Detection training completed")

    def train_all_stages(self,
                        train_data: torch.Tensor,
                        val_data: torch.Tensor,
                        config: Dict[str, Any]):
        """
        Train all self-supervised learning stages sequentially
        全自己教師あり学習段階の逐次学習
        """
        logger.info("Starting comprehensive self-supervised pre-training")

        # Stage 1: Masked Price Modeling
        if 'mpm' in config:
            logger.info("Stage 1: Masked Price Modeling")
            self.initialize_masked_price_model(**config['mpm'])
            self.train_masked_price_modeling(
                train_data, val_data, **config['mpm_training']
            )

        # Stage 2: Contrastive Learning
        if 'contrastive' in config:
            logger.info("Stage 2: Contrastive Learning")
            self.initialize_contrastive_model(**config['contrastive'])
            self.train_contrastive_learning(
                train_data, val_data, **config['contrastive_training']
            )

        # Stage 3: Anomaly Detection
        if 'anomaly' in config:
            logger.info("Stage 3: Anomaly Detection")
            self.initialize_anomaly_model(**config['anomaly'])
            self.train_anomaly_detection(
                train_data, val_data, **config['anomaly_training']
            )

        logger.info("Self-supervised pre-training completed")

    def get_pretrained_encoders(self) -> Dict[str, nn.Module]:
        """
        Get pretrained encoders for downstream tasks
        ダウンストリームタスク向け事前学習済みエンコーダーの取得
        """
        encoders = {}

        if self.masked_price_model is not None:
            # Extract encoder from MPM model
            encoders['mpm_encoder'] = self.masked_price_model.transformer_encoder

        if self.contrastive_model is not None:
            # Use contrastive model's encoder
            encoders['contrastive_encoder'] = self.contrastive_model.encoder

        if self.anomaly_model is not None:
            # Extract encoder from anomaly model
            encoders['anomaly_encoder'] = self.anomaly_model.reconstruction_detector.encoder

        return encoders

    def compute_anomaly_scores(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute anomaly scores using trained anomaly detector
        学習済み異常検知器による異常スコア計算
        """
        if self.ad_trainer is None:
            logger.warning("Anomaly detection model not trained")
            return None

        return self.ad_trainer.compute_anomaly_scores(data)

    def get_embeddings(self, data: torch.Tensor, method: str = 'contrastive') -> Optional[torch.Tensor]:
        """
        Get embeddings from specified method
        指定手法によるエンベディング取得
        """
        if method == 'contrastive' and self.cl_trainer is not None:
            return self.cl_trainer.get_embeddings(data)
        elif method == 'mpm' and self.masked_price_model is not None:
            # Use MPM encoder for embeddings
            self.masked_price_model.eval()
            with torch.no_grad():
                # Get encoded representations before projection
                batch_size, seq_len, _ = data.shape
                data = data.to(self.device)
                x_proj = self.masked_price_model.input_projection(data)
                x_proj = x_proj + self.masked_price_model.positional_encoding[:, :seq_len, :]
                encoded = self.masked_price_model.transformer_encoder(x_proj)
                # Global average pooling
                embeddings = encoded.mean(dim=1)
                return embeddings.cpu()
        else:
            logger.warning(f"Method {method} not available or not trained")
            return None

    def save_checkpoint(self, name: str):
        """Save checkpoint with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.checkpoint_dir / f"{name}_{timestamp}.pt"

        checkpoint = {
            'timestamp': timestamp,
            'training_history': self.training_history
        }

        if self.masked_price_model is not None:
            checkpoint['mpm_model'] = self.masked_price_model.state_dict()
            checkpoint['mpm_optimizer'] = self.mpm_trainer.optimizer.state_dict()

        if self.contrastive_model is not None:
            checkpoint['cl_model'] = self.contrastive_model.state_dict()
            checkpoint['cl_optimizer'] = self.cl_trainer.optimizer.state_dict()

        if self.anomaly_model is not None:
            checkpoint['ad_model'] = self.anomaly_model.state_dict()
            checkpoint['ad_optimizer'] = self.ad_trainer.optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path)

        if 'mpm_model' in checkpoint and self.masked_price_model is not None:
            self.masked_price_model.load_state_dict(checkpoint['mpm_model'])
            self.mpm_trainer.optimizer.load_state_dict(checkpoint['mpm_optimizer'])

        if 'cl_model' in checkpoint and self.contrastive_model is not None:
            self.contrastive_model.load_state_dict(checkpoint['cl_model'])
            self.cl_trainer.optimizer.load_state_dict(checkpoint['cl_optimizer'])

        if 'ad_model' in checkpoint and self.anomaly_model is not None:
            self.anomaly_model.load_state_dict(checkpoint['ad_model'])
            self.ad_trainer.optimizer.load_state_dict(checkpoint['ad_optimizer'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        logger.info(f"Checkpoint loaded: {path}")

    def save_training_history(self, path: str):
        """Save training history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def _train_epoch_mpm(self, data: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """Train one epoch for MPM"""
        total_loss = 0.0
        total_masked = 0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            metrics = self.mpm_trainer.train_step(batch)
            total_loss += metrics['loss']
            total_masked += metrics['masked_ratio'] * len(batch)
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'masked_ratio': total_masked / len(data)
        }

    def _train_epoch_cl(self, data: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """Train one epoch for Contrastive Learning"""
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            metrics = self.cl_trainer.train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1

        return {
            'loss': total_loss / num_batches
        }

    def _train_epoch_ad(self, data: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """Train one epoch for Anomaly Detection"""
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            metrics = self.ad_trainer.train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1

        return {
            'loss': total_loss / num_batches
        }