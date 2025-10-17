#!/usr/bin/env python3
"""
Tests for Model Compression Module.

This module contains comprehensive tests for all compression techniques:
- Quantization (dynamic, static, mixed precision)
- Pruning (L1/L2 unstructured, structured)
- Knowledge Distillation
- Integration with SAC models
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from ztb.optimization.model_compression import (
    QuantizationCompressor,
    PruningCompressor,
    KnowledgeDistillationCompressor,
    ModelCompressionManager,
    create_compression_pipeline,
    BaseCompressionTechnique
)


class SimpleTestModel(nn.Module):
    """Simple neural network for testing compression techniques."""

    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestQuantizationCompressor:
    """Test cases for quantization compression."""

    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = SimpleTestModel()
        compressor = QuantizationCompressor(quantization_type="dynamic")

        # Get original size
        original_size = compressor._get_model_size(model)

        # Apply compression
        compressed_model = compressor.compress(model)

        # Check that model was quantized
        assert compressed_model is not None
        assert compressor.quantized_model is not None

        # Check compression stats
        stats = compressor.get_compression_stats()
        assert stats["technique"] == "quantization"
        assert stats["type"] == "dynamic"
        assert stats["compression_ratio"] >= 0.0  # Compression ratio should be valid
        assert stats["original_size_mb"] > 0  # Original size should be calculated

    def test_static_quantization(self):
        """Test static quantization with calibration."""
        model = SimpleTestModel()
        compressor = QuantizationCompressor(quantization_type="static")

        # Create dummy calibration data
        calibration_data = torch.randn(100, 10)

        # Apply compression
        compressed_model = compressor.compress(model, calibration_data=calibration_data)

        # Check that model was quantized
        assert compressed_model is not None
        assert compressor.quantized_model is not None

        # Check compression stats
        stats = compressor.get_compression_stats()
        assert stats["technique"] == "quantization"
        assert stats["type"] == "static"

    def test_mixed_precision(self):
        """Test mixed precision compression."""
        model = SimpleTestModel()
        compressor = QuantizationCompressor(quantization_type="mixed_precision")

        # Apply compression
        compressed_model = compressor.compress(model)

        # Check that model was converted to half precision
        assert compressed_model is not None
        assert next(compressed_model.parameters()).dtype == torch.float16

    def test_invalid_quantization_type(self):
        """Test invalid quantization type raises error."""
        with pytest.raises(ValueError, match="Unsupported quantization type"):
            QuantizationCompressor(quantization_type="invalid")


class TestPruningCompressor:
    """Test cases for pruning compression."""

    def test_l1_unstructured_pruning(self):
        """Test L1 unstructured pruning."""
        model = SimpleTestModel()
        compressor = PruningCompressor(pruning_type="l1_unstructured", amount=0.5)

        # Get original sparsity
        original_sparsity = compressor._calculate_sparsity(model)
        assert original_sparsity == 0.0  # No sparsity initially

        # Apply compression
        compressed_model = compressor.compress(model)

        # Check that model was pruned
        assert compressed_model is not None

        # Check sparsity increased
        final_sparsity = compressor._calculate_sparsity(compressed_model)
        assert final_sparsity > original_sparsity

        # Check compression stats
        stats = compressor.get_compression_stats()
        assert stats["technique"] == "pruning"
        assert stats["type"] == "l1_unstructured"
        assert stats["amount"] == 0.5

    def test_l2_unstructured_pruning(self):
        """Test L2 unstructured pruning."""
        model = SimpleTestModel()
        compressor = PruningCompressor(pruning_type="l2_unstructured", amount=0.3)

        # Apply compression
        compressed_model = compressor.compress(model)

        # Check that model was pruned
        assert compressed_model is not None

        # Check compression stats
        stats = compressor.get_compression_stats()
        assert stats["technique"] == "pruning"
        assert stats["type"] == "l2_unstructured"
        assert stats["amount"] == 0.3

    def test_structured_pruning(self):
        """Test structured pruning."""
        model = SimpleTestModel()
        compressor = PruningCompressor(pruning_type="structured", amount=0.2)

        # Apply compression
        compressed_model = compressor.compress(model)

        # Check that model was pruned
        assert compressed_model is not None

        # Check compression stats
        stats = compressor.get_compression_stats()
        assert stats["technique"] == "pruning"
        assert stats["type"] == "structured"
        assert stats["amount"] == 0.2

    def test_invalid_pruning_type(self):
        """Test invalid pruning type raises error."""
        with pytest.raises(ValueError, match="Unsupported pruning type"):
            PruningCompressor(pruning_type="invalid")


class TestKnowledgeDistillationCompressor:
    """Test cases for knowledge distillation compression."""

    def test_distillation_initialization(self):
        """Test knowledge distillation compressor initialization."""
        compressor = KnowledgeDistillationCompressor(temperature=3.0, alpha=0.7)

        assert compressor.temperature == 3.0
        assert compressor.alpha == 0.7
        assert compressor.teacher_model is None
        assert compressor.student_model is None

    def test_distillation_compression(self):
        """Test knowledge distillation compression."""
        teacher_model = SimpleTestModel()
        student_model = SimpleTestModel(hidden_size=10)  # Smaller student model
        compressor = KnowledgeDistillationCompressor()

        # Apply compression
        compressed_model = compressor.compress(student_model, teacher_model=teacher_model)

        # Check that models were set
        assert compressor.teacher_model is teacher_model
        assert compressor.student_model is student_model
        assert compressed_model is student_model

    def test_distillation_without_teacher(self):
        """Test distillation fails without teacher model."""
        student_model = SimpleTestModel()
        compressor = KnowledgeDistillationCompressor()

        with pytest.raises(ValueError, match="teacher_model must be provided"):
            compressor.compress(student_model)

    def test_distillation_loss_calculation(self):
        """Test distillation loss calculation."""
        compressor = KnowledgeDistillationCompressor(temperature=2.0, alpha=0.5)

        # Create mock teacher and student models
        teacher_model = SimpleTestModel()
        student_model = SimpleTestModel()

        # Create dummy inputs and targets
        batch_size = 4
        input_size = 10
        output_size = 2

        student_logits = torch.randn(batch_size, output_size)
        teacher_logits = torch.randn(batch_size, output_size)
        targets = torch.randint(0, output_size, (batch_size,))

        # Create dummy criterion
        criterion = nn.CrossEntropyLoss()

        # Calculate distillation loss
        loss = compressor.get_distillation_loss(student_logits, teacher_logits, targets, criterion)

        # Check that loss is reasonable
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

        # Check that distillation loss history was updated
        assert len(compressor.distillation_loss_history) > 0


class TestModelCompressionManager:
    """Test cases for model compression manager."""

    def test_add_compressor(self):
        """Test adding compressors to manager."""
        manager = ModelCompressionManager()

        quant_compressor = QuantizationCompressor()
        prune_compressor = PruningCompressor()

        manager.add_compressor("quantization", quant_compressor)
        manager.add_compressor("pruning", prune_compressor)

        assert "quantization" in manager.compressors
        assert "pruning" in manager.compressors

    def test_compress_model_single_technique(self):
        """Test compressing model with single technique."""
        manager = ModelCompressionManager()
        quant_compressor = QuantizationCompressor(quantization_type="dynamic")
        manager.add_compressor("quantization", quant_compressor)

        model = SimpleTestModel()

        # Compress model
        compressed_model = manager.compress_model(model, ["quantization"])

        # Check that compression was applied
        assert compressed_model is not None
        assert len(manager.compression_stats) == 1
        assert "quantization" in manager.compression_stats

    def test_compress_model_multiple_techniques(self):
        """Test compressing model with multiple techniques."""
        manager = ModelCompressionManager()

        quant_compressor = QuantizationCompressor(quantization_type="dynamic")
        prune_compressor = PruningCompressor(amount=0.1)  # Light pruning

        manager.add_compressor("quantization", quant_compressor)
        manager.add_compressor("pruning", prune_compressor)

        model = SimpleTestModel()

        # Compress model
        compressed_model = manager.compress_model(model, ["quantization", "pruning"])

        # Check that both compressions were applied
        assert compressed_model is not None
        assert len(manager.compression_stats) == 2
        assert "quantization" in manager.compression_stats
        assert "pruning" in manager.compression_stats

    def test_compress_model_invalid_technique(self):
        """Test compressing with invalid technique."""
        manager = ModelCompressionManager()
        model = SimpleTestModel()

        # Try to compress with non-existent technique
        compressed_model = manager.compress_model(model, ["invalid_technique"])

        # Should still return model but log warning
        assert compressed_model is model
        assert len(manager.compression_stats) == 0

    def test_get_compression_report(self):
        """Test getting compression report."""
        manager = ModelCompressionManager()
        quant_compressor = QuantizationCompressor()
        manager.add_compressor("quantization", quant_compressor)

        model = SimpleTestModel()
        manager.compress_model(model, ["quantization"])

        report = manager.get_compression_report()

        assert "compression_stats" in report
        assert "total_techniques_applied" in report
        assert "techniques" in report
        assert report["total_techniques_applied"] == 1
        assert "quantization" in report["techniques"]

    @patch('torch.save')
    def test_save_compressed_model(self, mock_save):
        """Test saving compressed model."""
        manager = ModelCompressionManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            model = SimpleTestModel()
            save_path = Path(temp_dir) / "compressed_model.pth"

            manager.save_compressed_model(model, save_path)

            # Check that torch.save was called
            mock_save.assert_called_once()

    @patch('torch.load')
    def test_load_compressed_model(self, mock_load):
        """Test loading compressed model."""
        manager = ModelCompressionManager()

        # Mock the loaded checkpoint
        mock_checkpoint = {
            'model_state_dict': {'dummy': 'state'},
            'compression_stats': {'test': 'stats'}
        }
        mock_load.return_value = mock_checkpoint

        with patch.object(SimpleTestModel, 'load_state_dict') as mock_load_state:
            with tempfile.TemporaryDirectory() as temp_dir:
                load_path = Path(temp_dir) / "compressed_model.pth"

                loaded_model = manager.load_compressed_model(load_path, SimpleTestModel)

                # Check that model was created and state dict was loaded
                assert isinstance(loaded_model, SimpleTestModel)
                mock_load_state.assert_called_once_with(mock_checkpoint['model_state_dict'])

                # Check that compression stats were loaded
                assert manager.compression_stats == mock_checkpoint['compression_stats']


class TestCompressionPipeline:
    """Test cases for compression pipeline creation."""

    def test_create_quantization_pipeline(self):
        """Test creating quantization pipeline."""
        config = {
            "quantization": {
                "type": "quantization",
                "quantization_type": "dynamic"
            }
        }

        manager = create_compression_pipeline(config)

        assert isinstance(manager, ModelCompressionManager)
        assert "quantization" in manager.compressors
        assert isinstance(manager.compressors["quantization"], QuantizationCompressor)

    def test_create_pruning_pipeline(self):
        """Test creating pruning pipeline."""
        config = {
            "pruning": {
                "type": "pruning",
                "pruning_type": "l1_unstructured",
                "amount": 0.3
            }
        }

        manager = create_compression_pipeline(config)

        assert isinstance(manager, ModelCompressionManager)
        assert "pruning" in manager.compressors
        assert isinstance(manager.compressors["pruning"], PruningCompressor)

    def test_create_distillation_pipeline(self):
        """Test creating distillation pipeline."""
        config = {
            "distillation": {
                "type": "distillation",
                "temperature": 2.5,
                "alpha": 0.6
            }
        }

        manager = create_compression_pipeline(config)

        assert isinstance(manager, ModelCompressionManager)
        assert "distillation" in manager.compressors
        assert isinstance(manager.compressors["distillation"], KnowledgeDistillationCompressor)

    def test_create_mixed_pipeline(self):
        """Test creating mixed compression pipeline."""
        config = {
            "quantization": {
                "type": "quantization",
                "quantization_type": "dynamic"
            },
            "pruning": {
                "type": "pruning",
                "pruning_type": "l1_unstructured",
                "amount": 0.2
            }
        }

        manager = create_compression_pipeline(config)

        assert isinstance(manager, ModelCompressionManager)
        assert len(manager.compressors) == 2
        assert "quantization" in manager.compressors
        assert "pruning" in manager.compressors

    def test_create_pipeline_invalid_type(self):
        """Test creating pipeline with invalid compression type."""
        config = {
            "invalid": {
                "type": "invalid_type"
            }
        }

        # Should not raise error, just skip invalid types
        manager = create_compression_pipeline(config)

        assert isinstance(manager, ModelCompressionManager)
        # Invalid compressor should not be added
        assert len(manager.compressors) == 0


class TestBaseCompressionTechnique:
    """Test cases for base compression technique."""

    def test_base_class_is_abstract(self):
        """Test that BaseCompressionTechnique cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseCompressionTechnique()

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # This is more of a documentation test - the abstract methods should exist
        assert hasattr(BaseCompressionTechnique, 'compress')
        assert hasattr(BaseCompressionTechnique, 'decompress')
        assert hasattr(BaseCompressionTechnique, 'get_compression_stats')


if __name__ == "__main__":
    pytest.main([__file__])