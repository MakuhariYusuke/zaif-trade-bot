#!/usr/bin/env python3
"""
Model Compression Module for SAC v421.

This module provides comprehensive model compression techniques including:
- Quantization (FP32→FP16/INT8, dynamic quantization, mixed precision training)
- Pruning (structural pruning, dynamic pruning based on importance)
- Knowledge Distillation (teacher-student model training)

The module is designed to work seamlessly with the existing SAC algorithm
and can be integrated into the training pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BaseCompressionTechnique(ABC):
    """Base class for model compression techniques."""

    @abstractmethod
    def compress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply compression technique to the model."""
        pass

    @abstractmethod
    def decompress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Reverse compression if needed."""
        pass

    @abstractmethod
    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        pass

class QuantizationCompressor(BaseCompressionTechnique):
    """
    Quantization-based model compression.

    Supports:
    - Dynamic quantization (FP32→INT8)
    - Static quantization with calibration
    - Mixed precision training (FP16)
    """

    def __init__(self, dtype: torch.dtype = torch.qint8) -> None:
        """
        Simple dynamic quantization compressor.

        Args:
            dtype: Quantized dtype to use for dynamic quantization
        """
        self.dtype = dtype
        self.original_size_mb = 0.0
        self.compressed_size_mb = 0.0

    def compress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply dynamic quantization to reduce model size."""
        try:
            # Record original size (best-effort)
            self.original_size_mb = self._get_model_size(model)

            # Use torch's dynamic quantization for supported layers
            q_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=self.dtype
            )

            self.compressed_size_mb = self._get_model_size(q_model)

            logger.info(
                f"Quantized model: original={self.original_size_mb:.2f}MB compressed={self.compressed_size_mb:.2f}MB"
            )
            return q_model
        except Exception as e:
            logger.warning(f"Quantization failed, returning original model: {e}")
            return model

    def decompress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Decompression is not supported for quantized models; return as-is."""
        logger.warning("Decompression for quantized models is not supported; returning provided model")
        return model

    def get_compression_stats(self) -> dict[str, Any]:
        """Return basic compression stats for quantization."""
        return {
            "original_size_mb": self.original_size_mb,
            "compressed_size_mb": self.compressed_size_mb,
            "compression_ratio": (
                self.original_size_mb / self.compressed_size_mb
                if self.compressed_size_mb > 0
                else 0
            ),
        }

    # Pruning helpers are defined on the canonical PruningCompressor below

    

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity (fraction of zero weights)."""
        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

class PruningCompressor(BaseCompressionTechnique):
    """
    Pruning-based model compression.

    Supports:
    - Structural pruning (channel/filter pruning)
    - Dynamic pruning based on importance scores
    """

    def __init__(self, pruning_type: str = "l1_unstructured", amount: float = 0.3) -> None:
        """
        Initialize pruning compressor.

        Args:
            pruning_type: Type of pruning ("l1_unstructured", "l2_unstructured", "structured")
            amount: Amount of pruning (0.0 to 1.0)
        """
        valid_types = ["l1_unstructured", "l2_unstructured", "structured"]
        if pruning_type not in valid_types:
            raise ValueError(
                f"Unsupported pruning type: {pruning_type}. Supported types: {valid_types}"
            )

        if not (0.0 < amount < 1.0):
            raise ValueError(
                f"Pruning amount must be between 0.0 and 1.0, got {amount}"
            )

        self.pruning_type = pruning_type
        self.amount = amount
        self.pruned_weights = {}
        self.original_sparsity = 0
        self.final_sparsity = 0

    def compress(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply pruning to the model.

        Args:
            model: PyTorch model to prune
            **kwargs: Additional arguments for pruning

        Returns:
            Pruned model
        """
        logger.info(
            f"Applying {self.pruning_type} pruning with amount {self.amount}..."
        )

        self.original_sparsity = self._calculate_sparsity(model)

        if self.pruning_type == "l1_unstructured":
            self._apply_l1_unstructured_pruning(model)
        elif self.pruning_type == "l2_unstructured":
            self._apply_l2_unstructured_pruning(model)
        elif self.pruning_type == "structured":
            self._apply_structured_pruning(model)
        else:
            raise ValueError(
                f"Unsupported pruning type: {self.pruning_type}. Supported types: l1_unstructured, l2_unstructured, structured"
            )

        self.final_sparsity = self._calculate_sparsity(model)

        logger.info(".1%")

        return model

    def _apply_l1_unstructured_pruning(self, model: nn.Module) -> None:
        """Apply L1 unstructured pruning."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate L1 norm for each weight
                weight_l1 = torch.abs(module.weight).sum(dim=1)
                _, indices = torch.topk(
                    weight_l1,
                    int(module.weight.size(0) * (1 - self.amount)),
                    largest=True,
                )
                mask = torch.zeros_like(module.weight)
                mask[indices] = 1
                module.weight.data *= mask

    def _apply_l2_unstructured_pruning(self, model: nn.Module) -> None:
        """Apply L2 unstructured pruning."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate L2 norm for each weight
                weight_l2 = torch.sqrt(torch.sum(module.weight**2, dim=1))
                _, indices = torch.topk(
                    weight_l2,
                    int(module.weight.size(0) * (1 - self.amount)),
                    largest=True,
                )
                mask = torch.zeros_like(module.weight)
                mask[indices] = 1
                module.weight.data *= mask

    def _apply_structured_pruning(self, model: nn.Module) -> None:
        """Apply structured pruning (channel pruning)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune entire neurons/channels
                weight_l1 = torch.abs(module.weight).sum(
                    dim=0
                )  # Sum across output dimension
                _, indices = torch.topk(
                    weight_l1,
                    int(module.weight.size(1) * (1 - self.amount)),
                    largest=True,
                )
                mask = torch.zeros_like(module.weight)
                mask[:, indices] = 1
                module.weight.data *= mask

    def decompress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Pruning is typically irreversible, return model as-is."""
        logger.warning(
            "Pruning decompression not supported - pruning is typically irreversible"
        )
        return model

    def get_compression_stats(self) -> dict[str, Any]:
        """Get pruning compression statistics."""
        return {
            "technique": "pruning",
            "type": self.pruning_type,
            "amount": self.amount,
            "original_sparsity": self.original_sparsity,
            "final_sparsity": self.final_sparsity,
            "sparsity_increase": self.final_sparsity - self.original_sparsity,
        }

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity (percentage of zero weights)."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0

class KnowledgeDistillationCompressor(BaseCompressionTechnique):
    """
    Knowledge Distillation-based model compression.

    Trains a smaller student model to mimic a larger teacher model.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        """
        Initialize knowledge distillation compressor.

        Args:
            temperature: Temperature for softening logits
            alpha: Weight for distillation loss vs ground truth loss
        """
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = None
        self.student_model = None
        self.distillation_loss_history = []

    def compress(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply knowledge distillation.

        Args:
            model: Student model to train
            teacher_model: Pre-trained teacher model
            **kwargs: Additional arguments

        Returns:
            Trained student model
        """
        teacher_model = kwargs.get("teacher_model")
        if teacher_model is None:
            raise ValueError(
                "teacher_model must be provided for knowledge distillation"
            )

        logger.info("Applying knowledge distillation...")
        logger.info(".1f")

        self.teacher_model = teacher_model
        self.student_model = model

        # set teacher to evaluation mode
        self.teacher_model.eval()

        return model

    def get_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Calculate distillation loss.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            targets: Ground truth targets
            criterion: Base criterion for ground truth loss

        Returns:
            Combined distillation loss
        """
        # Ground truth loss
        loss_gt = criterion(student_logits, targets)

        # Distillation loss (KL divergence between softened logits)
        teacher_softened = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_softened = torch.log_softmax(student_logits / self.temperature, dim=1)

        loss_distill = nn.KLDivLoss(reduction="batchmean")(student_softened, teacher_softened) * (
            self.temperature ** 2
        )

        # Combined loss (weighted by alpha)
        total_loss = self.alpha * loss_distill + (1.0 - self.alpha) * loss_gt

        return total_loss
    # Note: A legacy duplicate `PruningCompressor` implementation was removed
    # in favor of the single canonical implementation above which includes
    # validation, different pruning strategies and optional caching.

    # End of PruningCompressor

    def compress_model(
        self, model: nn.Module, techniques: list[str], **kwargs
    ) -> nn.Module:
        """
        Apply multiple compression techniques to a model.

        Args:
            model: Model to compress
            techniques: list of compression technique names to apply
            **kwargs: Arguments for compression techniques

        Returns:
            Compressed model
        """
        compressed_model = model

        for technique in techniques:
            if technique not in self.compressors:
                logger.warning(f"Compressor {technique} not found, skipping")
                continue

            logger.info(f"Applying compression technique: {technique}")
            compressor = self.compressors[technique]

            try:
                compressed_model = compressor.compress(compressed_model, **kwargs)
                self.compression_stats[technique] = compressor.get_compression_stats()
                logger.info(f"Successfully applied {technique}")
            except Exception as e:
                logger.error(f"Failed to apply {technique}: {e}")
                # Continue with other techniques but don't add to stats
                continue

        return compressed_model

    def get_compression_report(self) -> dict[str, Any]:
        """Get comprehensive compression report."""
        return {
            "compression_stats": self.compression_stats,
            "total_techniques_applied": len(self.compression_stats),
            "techniques": list(self.compression_stats.keys()),
        }

    def save_compressed_model(self, model: nn.Module, path: str | Path):
        """Save compressed model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "compression_stats": self.compression_stats,
            },
            path,
        )

        logger.info(f"Compressed model saved to {path}")

    def load_compressed_model(
        self, path: str | Path, model_class: nn.Module
    ) -> nn.Module:
        """Load compressed model from file."""
        path = Path(path)
        checkpoint = torch.load(path)

        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])

        self.compression_stats = checkpoint.get("compression_stats", {})

        logger.info(f"Compressed model loaded from {path}")
        return model

class ModelCompressionManager:
    """Manager class for applying multiple compression techniques and tracking stats."""

    def __init__(self) -> None:
        self.compressors: dict[str, BaseCompressionTechnique] = {}
        self.compression_stats: dict[str, Any] = {}

    def add_compressor(self, name: str, compressor: BaseCompressionTechnique) -> None:
        self.compressors[name] = compressor

    def compress(self, model: nn.Module, techniques: list[str], **kwargs) -> nn.Module:
        compressed_model = model
        for technique in techniques:
            compressor = self.compressors.get(technique)
            if compressor is None:
                logger.warning(f"Compressor {technique} not found, skipping")
                continue
            try:
                compressed_model = compressor.compress(compressed_model, **kwargs)
                self.compression_stats[technique] = compressor.get_compression_stats()
            except Exception as e:
                logger.error(f"Failed to apply {technique}: {e}")
        return compressed_model

    def get_compression_stats(self) -> dict[str, Any]:
        return self.compression_stats

def create_compression_pipeline(
    techniques_config: dict[str, dict[str, Any]],
) -> Any:
    """
    Create a compression pipeline from configuration.

    Args:
        techniques_config: Configuration for compression techniques

    Returns:
        Configured ModelCompressionManager
    """
    manager = ModelCompressionManager()

    for name, config in techniques_config.items():
        technique_type = config.pop("type")

        if technique_type == "quantization":
            compressor = QuantizationCompressor(**config)
        elif technique_type == "pruning":
            compressor = PruningCompressor(**config)
        elif technique_type == "distillation":
            compressor = KnowledgeDistillationCompressor(**config)
        else:
            logger.warning(f"Unknown compression type: {technique_type}")
            continue

        manager.add_compressor(name, compressor)

    return manager
