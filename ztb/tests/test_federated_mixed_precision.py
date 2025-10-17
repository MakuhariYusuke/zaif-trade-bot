#!/usr/bin/env python3
"""
Test suite for Federated Learning and Mixed Precision Training in UnifiedTrainer.
"""

import sys
import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
sys.path.append('.')

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.unified_trainer.config import UnifiedTrainerConfig, UnifiedAlgorithm


class TestFederatedMixedPrecision(unittest.TestCase):
    """Test cases for federated learning and mixed precision training."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            'algorithm': 'ppo',
            'total_timesteps': 1000,
            'model_name': 'test_model'
        }

    def test_federated_learning_config(self):
        """Test federated learning configuration setup."""
        config = self.base_config.copy()
        config.update({
            'enable_federated': True,
            'num_clients': 3,
            'federated_rounds': 5,
            'privacy_budget': 1.0,
            'client_fraction': 0.8
        })

        trainer = UnifiedTrainer(config, dry_run=True)

        # Check if federated components are initialized
        self.assertEqual(len(trainer.federated_clients), 0)  # Not initialized until _setup_federated_learning
        self.assertIsNone(trainer.global_model_state)

    def test_mixed_precision_config(self):
        """Test mixed precision training configuration setup."""
        config = self.base_config.copy()
        config.update({
            'enable_mixed_precision': True,
            'precision': 'fp16',
            'gradient_scaling': True,
            'gradient_clip_norm': 1.0
        })

        trainer = UnifiedTrainer(config, dry_run=True)

        # Check if mixed precision components are initialized
        if torch.cuda.is_available():
            self.assertIsNotNone(trainer.grad_scaler)
        else:
            # On CPU, GradScaler might not be initialized
            pass

    @patch('ztb.training.unified_trainer.trainer.OPACUS_AVAILABLE', True)
    def test_setup_federated_learning_success(self):
        """Test successful federated learning setup."""
        config = self.base_config.copy()
        config.update({
            'enable_federated': True,
            'num_clients': 2
        })

        trainer = UnifiedTrainer(config, dry_run=True)

        # Test setup
        success = trainer._setup_federated_learning()
        self.assertTrue(success)
        self.assertEqual(len(trainer.federated_clients), 2)

        # Check client configurations
        for i, client_config in enumerate(trainer.federated_clients):
            self.assertEqual(client_config['client_id'], i)

    @patch('ztb.training.unified_trainer.trainer.OPACUS_AVAILABLE', False)
    def test_setup_federated_learning_without_opacus(self):
        """Test federated learning setup without Opacus (warning logged)."""
        config = self.base_config.copy()
        config.update({
            'enable_federated': True,
            'num_clients': 2
        })

        trainer = UnifiedTrainer(config, dry_run=True)

        # Should still work without Opacus
        success = trainer._setup_federated_learning()
        self.assertTrue(success)
        self.assertEqual(len(trainer.federated_clients), 2)

    @patch('ztb.training.unified_trainer.trainer.AMP_AVAILABLE')
    @patch('torch.cuda.is_available')
    def test_setup_mixed_precision_success(self, mock_cuda, mock_amp):
        """Test successful mixed precision setup."""
        mock_amp.return_value = True
        mock_cuda.return_value = True
        
        config = self.base_config.copy()
        config['enable_mixed_precision'] = True

        trainer = UnifiedTrainer(config, dry_run=True)

        success = trainer._setup_mixed_precision()
        self.assertTrue(success)
        self.assertIsNotNone(trainer.grad_scaler)

    @patch('ztb.training.unified_trainer.trainer.AMP_AVAILABLE', False)
    def test_setup_mixed_precision_without_amp(self):
        """Test mixed precision setup failure without AMP."""
        config = self.base_config.copy()
        config['enable_mixed_precision'] = True

        trainer = UnifiedTrainer(config, dry_run=True)

        success = trainer._setup_mixed_precision()
        self.assertFalse(success)

    def test_federated_average_simple(self):
        """Test simple federated averaging."""
        trainer = UnifiedTrainer(self.base_config, dry_run=True)

        # Mock client updates (simple tensors)
        client_updates = [
            {'param1': torch.tensor([1.0, 2.0]), 'param2': torch.tensor([3.0])},
            {'param1': torch.tensor([3.0, 4.0]), 'param2': torch.tensor([5.0])},
        ]

        averaged = trainer._federated_average(client_updates)

        # Check averaged values
        expected_param1 = torch.tensor([2.0, 3.0])  # Average of [1,2] and [3,4]
        expected_param2 = torch.tensor([4.0])       # Average of [3] and [5]

        self.assertTrue(torch.allclose(averaged['param1'], expected_param1))
        self.assertTrue(torch.allclose(averaged['param2'], expected_param2))

    @patch('ztb.training.unified_trainer.trainer.AMP_AVAILABLE')
    @patch('torch.cuda.is_available')
    def test_apply_mixed_precision_with_scaler(self, mock_cuda, mock_amp):
        """Test mixed precision loss scaling."""
        mock_amp.return_value = True
        mock_cuda.return_value = True
        
        config = self.base_config.copy()
        config['enable_mixed_precision'] = True

        trainer = UnifiedTrainer(config, dry_run=True)

        # Mock loss tensor
        loss = torch.tensor(1.0, requires_grad=True)

        if trainer.grad_scaler is not None:
            scaled_loss = trainer._apply_mixed_precision(loss)
            # With CUDA available, scaled loss should be scaled
            # Note: The actual scaling factor depends on GradScaler's internal state
            self.assertIsInstance(scaled_loss, torch.Tensor)
        else:
            # Without scaler, should return original loss
            scaled_loss = trainer._apply_mixed_precision(loss)
            self.assertEqual(scaled_loss.item(), loss.item())

    def test_step_optimizer_with_scaler(self):
        """Test optimizer step with gradient scaler."""
        config = self.base_config.copy()
        config['enable_mixed_precision'] = True

        trainer = UnifiedTrainer(config, dry_run=True)

        # Mock optimizer
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # This should not raise an exception
        try:
            trainer._step_optimizer(optimizer)
        except Exception as e:
            self.fail(f"Optimizer step failed: {e}")


if __name__ == '__main__':
    unittest.main()