"""
Tests for Self-Supervised Pre-training Module
自己教師あり事前学習モジュールのテスト

This module contains comprehensive tests for all self-supervised learning
components including Masked Price Modeling, Contrastive Learning,
and Anomaly Detection Pre-training.
"""

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

from ztb.multimodal.pretraining import (
    MaskedPriceModel,
    MaskedPriceModelingTrainer,
    ContrastiveLearningModel,
    ContrastiveLearningTrainer,
    TimeSeriesAugmentation,
    HybridAnomalyDetector,
    AnomalyDetectionPretrainer,
    SelfSupervisedTrainer
)


class TestMaskedPriceModeling:
    """Test Masked Price Modeling components"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample financial time series data"""
        batch_size, seq_len, input_dim = 4, 50, 156
        return torch.randn(batch_size, seq_len, input_dim)

    def test_masked_price_model_initialization(self):
        """Test MaskedPriceModel initialization"""
        model = MaskedPriceModel(
            input_dim=156,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            max_seq_len=50
        )

        assert model.input_dim == 156
        assert model.hidden_dim == 256
        assert model.mask_prob == 0.15

    def test_masked_price_model_forward(self, sample_data):
        """Test MaskedPriceModel forward pass"""
        model = MaskedPriceModel(input_dim=156, hidden_dim=256, max_seq_len=50)

        predictions, mask_indices = model(sample_data)

        assert predictions.shape == sample_data.shape
        assert mask_indices.shape == (sample_data.shape[0], sample_data.shape[1])
        assert mask_indices.dtype == torch.bool

    def test_masked_price_model_loss(self, sample_data):
        """Test loss computation"""
        model = MaskedPriceModel(input_dim=156, hidden_dim=256, max_seq_len=50)

        predictions, mask_indices = model(sample_data)
        loss = model.compute_loss(predictions, sample_data, mask_indices)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss >= 0

    def test_masked_price_trainer(self, sample_data):
        """Test MaskedPriceModelingTrainer"""
        model = MaskedPriceModel(input_dim=156, hidden_dim=256, max_seq_len=50)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = MaskedPriceModelingTrainer(model, optimizer, 'cpu')

        metrics = trainer.train_step(sample_data)

        assert 'loss' in metrics
        assert 'masked_ratio' in metrics
        assert isinstance(metrics['loss'], float)
        assert 0 <= metrics['masked_ratio'] <= 1


class TestContrastiveLearning:
    """Test Contrastive Learning components"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for contrastive learning"""
        batch_size, seq_len, input_dim = 8, 50, 156
        return torch.randn(batch_size, seq_len, input_dim)

    def test_time_series_augmentation(self, sample_data):
        """Test TimeSeriesAugmentation"""
        augmentation = TimeSeriesAugmentation(
            shift_prob=1.0,  # Always apply for testing
            noise_prob=1.0,  # Always apply noise
            scale_prob=0.0,
            max_shift=10  # Larger shift to ensure change
        )

        augmented = augmentation(sample_data)

        # Shape should remain the same
        assert augmented.shape == sample_data.shape
        # Data should be different due to shifting
        assert not torch.equal(augmented, sample_data)

    def test_contrastive_model_initialization(self):
        """Test ContrastiveLearningModel initialization"""
        model = ContrastiveLearningModel(
            input_dim=156,
            hidden_dim=256,
            projection_dim=64,
            temperature=0.5
        )

        assert model.input_dim == 156
        assert model.projection_dim == 64
        assert model.temperature == 0.5

    def test_contrastive_model_forward(self, sample_data):
        """Test ContrastiveLearningModel forward pass"""
        model = ContrastiveLearningModel(input_dim=156, hidden_dim=256, projection_dim=64)

        z1, z2 = model(sample_data, sample_data)

        assert z1.shape == (sample_data.shape[0], 64)
        assert z2.shape == (sample_data.shape[0], 64)

    def test_contrastive_loss(self, sample_data):
        """Test contrastive loss computation"""
        model = ContrastiveLearningModel(input_dim=156, hidden_dim=256, projection_dim=64)

        z1, z2 = model(sample_data, sample_data)
        loss = model.compute_contrastive_loss(z1, z2)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0

    def test_contrastive_trainer(self, sample_data):
        """Test ContrastiveLearningTrainer"""
        model = ContrastiveLearningModel(input_dim=156, hidden_dim=256, projection_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        augmentation = TimeSeriesAugmentation()
        trainer = ContrastiveLearningTrainer(model, optimizer, augmentation, 'cpu')

        metrics = trainer.train_step(sample_data)

        assert 'loss' in metrics
        assert 'z1_norm' in metrics
        assert 'z2_norm' in metrics
        assert isinstance(metrics['loss'], float)


class TestAnomalyDetection:
    """Test Anomaly Detection components"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for anomaly detection"""
        batch_size, seq_len, input_dim = 4, 50, 156
        return torch.randn(batch_size, seq_len, input_dim)

    def test_hybrid_anomaly_detector_initialization(self):
        """Test HybridAnomalyDetector initialization"""
        model = HybridAnomalyDetector(
            input_dim=156,
            hidden_dims=[128, 64, 32],
            latent_dim=16,
            seq_len=50,
            alpha=0.5
        )

        assert model.alpha == 0.5

    def test_hybrid_anomaly_detector_forward(self, sample_data):
        """Test HybridAnomalyDetector forward pass"""
        model = HybridAnomalyDetector(
            input_dim=156,
            hidden_dims=[128, 64, 32],
            latent_dim=16,
            seq_len=50
        )

        reconstructed, prediction = model(sample_data)

        assert reconstructed.shape == sample_data.shape
        assert prediction.shape == (sample_data.shape[0], 156)

    def test_anomaly_score_computation(self, sample_data):
        """Test anomaly score computation"""
        model = HybridAnomalyDetector(
            input_dim=156,
            hidden_dims=[128, 64, 32],
            latent_dim=16,
            seq_len=50
        )

        scores = model.compute_anomaly_score(sample_data)

        assert scores.shape == (sample_data.shape[0],)
        assert torch.all(scores >= 0)

    def test_anomaly_trainer(self, sample_data):
        """Test AnomalyDetectionPretrainer"""
        model = HybridAnomalyDetector(
            input_dim=156,
            hidden_dims=[128, 64, 32],
            latent_dim=16,
            seq_len=50
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = AnomalyDetectionPretrainer(model, optimizer, 'cpu')

        metrics = trainer.train_step(sample_data)

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)


class TestSelfSupervisedTrainer:
    """Test integrated SelfSupervisedTrainer"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        train_size, val_size, seq_len, input_dim = 20, 10, 50, 156
        train_data = torch.randn(train_size, seq_len, input_dim)
        val_data = torch.randn(val_size, seq_len, input_dim)
        return train_data, val_data

    def test_trainer_initialization(self):
        """Test SelfSupervisedTrainer initialization"""
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')
        assert trainer.input_dim == 156
        assert trainer.device == 'cpu'

    def test_mpm_initialization(self):
        """Test MPM model initialization in trainer"""
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')
        trainer.initialize_masked_price_model(hidden_dim=256, max_seq_len=50)

        assert trainer.masked_price_model is not None
        assert trainer.mpm_trainer is not None

    def test_contrastive_initialization(self):
        """Test contrastive model initialization in trainer"""
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')
        trainer.initialize_contrastive_model(hidden_dim=256, projection_dim=64)

        assert trainer.contrastive_model is not None
        assert trainer.cl_trainer is not None

    def test_anomaly_initialization(self):
        """Test anomaly model initialization in trainer"""
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')
        trainer.initialize_anomaly_model(
            hidden_dims=[128, 64, 32],
            latent_dim=16,
            seq_len=50
        )

        assert trainer.anomaly_model is not None
        assert trainer.ad_trainer is not None

    def test_get_pretrained_encoders(self):
        """Test getting pretrained encoders"""
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')

        # Without any models
        encoders = trainer.get_pretrained_encoders()
        assert len(encoders) == 0

        # With models initialized
        trainer.initialize_contrastive_model(hidden_dim=256, projection_dim=64)
        encoders = trainer.get_pretrained_encoders()
        assert 'contrastive_encoder' in encoders

    def test_embeddings_extraction(self, sample_data):
        """Test embeddings extraction"""
        train_data, val_data = sample_data
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')
        trainer.initialize_contrastive_model(hidden_dim=256, projection_dim=64)

        embeddings = trainer.get_embeddings(train_data, method='contrastive')

        assert embeddings is not None
        assert embeddings.shape[0] == train_data.shape[0]
        assert embeddings.shape[1] == 256  # hidden_dim


class TestIntegration:
    """Integration tests for the complete pipeline"""

    def test_full_pipeline(self):
        """Test complete self-supervised learning pipeline"""
        # This is a lightweight integration test
        trainer = SelfSupervisedTrainer(input_dim=156, device='cpu')

        # Generate small dataset
        batch_size, seq_len, input_dim = 4, 20, 156
        train_data = torch.randn(batch_size, seq_len, input_dim)
        val_data = torch.randn(batch_size // 2, seq_len, input_dim)

        # Initialize models
        trainer.initialize_masked_price_model(
            hidden_dim=128, num_layers=2, num_heads=2, max_seq_len=20
        )
        trainer.initialize_contrastive_model(hidden_dim=128, projection_dim=32)
        trainer.initialize_anomaly_model(
            hidden_dims=[64, 32], latent_dim=8, seq_len=20
        )

        # Test training for a few epochs
        trainer.train_masked_price_modeling(
            train_data, val_data, epochs=2, batch_size=2, patience=5
        )
        trainer.train_contrastive_learning(
            train_data, val_data, epochs=2, batch_size=2, patience=5
        )
        trainer.train_anomaly_detection(
            train_data, val_data, epochs=2, batch_size=2, patience=5
        )

        # Test inference
        encoders = trainer.get_pretrained_encoders()
        assert len(encoders) == 3  # mpm, contrastive, anomaly

        embeddings = trainer.get_embeddings(train_data, method='contrastive')
        assert embeddings is not None

        anomaly_scores = trainer.compute_anomaly_scores(val_data)
        assert anomaly_scores is not None
        assert anomaly_scores.shape[0] == val_data.shape[0]


if __name__ == '__main__':
    pytest.main([__file__])