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
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from ztb.utils.logging_utils import get_logger
from ztb.cache.memory_cache import default_memory_manager

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
    def get_compression_stats(self) -> Dict[str, Any]:
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

    def __init__(self, quantization_type: str = "dynamic"):
        """
        Initialize quantization compressor.

        Args:
            quantization_type: Type of quantization ("dynamic", "static", "mixed_precision")
        """
        valid_types = ["dynamic", "static", "mixed_precision"]
        if quantization_type not in valid_types:
            raise ValueError(
                f"Unsupported quantization type: {quantization_type}. Supported types: {valid_types}"
            )

        self.quantization_type = quantization_type
        self.original_model_size = 0
        self.compressed_model_size = 0
        self.quantized_model = None

    def compress(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply quantization to the model.

        Args:
            model: PyTorch model to quantize
            **kwargs: Additional arguments for quantization

        Returns:
            Quantized model
        """
        self.original_model_size = self._get_model_size(model)

        if self.quantization_type == "dynamic":
            return self._apply_dynamic_quantization(model, **kwargs)
        elif self.quantization_type == "static":
            return self._apply_static_quantization(model, **kwargs)
        elif self.quantization_type == "mixed_precision":
            return self._apply_mixed_precision(model, **kwargs)
        else:
            raise ValueError(
                f"Unsupported quantization type: {self.quantization_type}. Supported types: dynamic, static, mixed_precision"
            )

    def _apply_dynamic_quantization(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply dynamic quantization (FP32→INT8)."""
        logger.info("Applying dynamic quantization...")

        # Prepare model for quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,  # Layers to quantize
        )

        self.quantized_model = quantized_model
        self.compressed_model_size = self._get_model_size(quantized_model)

        if self.compressed_model_size > 0:
            compression_ratio = self.original_model_size / self.compressed_model_size
            logger.info(".2f")
        else:
            logger.warning(
                "Compressed model size is 0, cannot calculate compression ratio"
            )
            compression_ratio = 1.0

        return quantized_model

    def _apply_static_quantization(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        logger.info("Applying static quantization...")

        # Add quantization stubs
        model = self._add_quantization_stubs(model)

        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # Prepare for quantization
        torch.quantization.prepare(model, inplace=True)

        # Calibrate with data if provided
        if calibration_data is not None:
            self._calibrate_model(model, calibration_data)

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        self.quantized_model = model
        self.compressed_model_size = self._get_model_size(model)

        compression_ratio = self.original_model_size / self.compressed_model_size
        logger.info(".2f")

        return model

    def _apply_mixed_precision(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply mixed precision (FP16)."""
        logger.info("Applying mixed precision training...")

        # Convert model to half precision
        model.half()

        self.quantized_model = model
        self.compressed_model_size = self._get_model_size(model)

        compression_ratio = self.original_model_size / self.compressed_model_size
        logger.info(".2f")

        return model

    def _add_quantization_stubs(self, model: nn.Module) -> nn.Module:
        """Add quantization and dequantization stubs to model."""
        model.quant = QuantStub()
        model.dequant = DeQuantStub()
        return model

    def _calibrate_model(self, model: nn.Module, calibration_data: torch.Tensor):
        """Calibrate quantized model with sample data."""
        logger.info("Calibrating quantized model...")
        with torch.no_grad():
            for _ in range(100):  # Calibration steps
                _ = model(calibration_data)

    def decompress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Dequantize model back to FP32."""
        if hasattr(model, "dequant"):
            # Static quantization case
            return model.dequant(model.quant(torch.randn(1, model.quant.input_size)))
        else:
            # Dynamic quantization case - convert back to float
            return model.float()

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get quantization compression statistics."""
        return {
            "technique": "quantization",
            "type": self.quantization_type,
            "original_size_mb": self.original_model_size,
            "compressed_size_mb": self.compressed_model_size,
            "compression_ratio": self.original_model_size / self.compressed_model_size
            if self.compressed_model_size > 0
            else 0,
        }

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            total_size = param_size + buffer_size
            return total_size / 1024 / 1024 if total_size > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
            return 0.0


class PruningCompressor(BaseCompressionTechnique):
    """
    Neural network pruning for model compression.

    Supports multiple pruning techniques:
    - L1/L2 unstructured pruning
    - Structured pruning (channel-wise)
    - Dynamic pruning based on importance scores
    """

    def __init__(self, pruning_type: str = "l1_unstructured", amount: float = 0.2):
        """
        Initialize pruning compressor.

        Args:
            pruning_type: Type of pruning ('l1_unstructured', 'l2_unstructured', 'structured')
            amount: Fraction of weights to prune (0.0 to 1.0)
        """
        self.pruning_type = pruning_type
        self.amount = amount
        self.pruned_weights = {}
        self.original_sparsity = 0.0
        self.final_sparsity = 0.0

    def compress(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply pruning to the model with memory caching.

        Args:
            model: PyTorch model to prune
            **kwargs: Additional arguments for pruning

        Returns:
            Pruned model
        """
        # Create cache key for pruned model
        model_hash = hash(str(model.state_dict()))
        cache_key = f"pruned_model_{model_hash}_{self.pruning_type}_{self.amount}"

        # Check memory cache first
        cached_model = default_memory_manager.get_cached_model_state(cache_key)
        if cached_model is not None:
            logger.info("Loading pruned model from memory cache")
            model.load_state_dict(cached_model)
            return model

        logger.info(f"Applying {self.pruning_type} pruning with amount {self.amount}...")

        self.original_sparsity = self._calculate_sparsity(model)

        if self.pruning_type == "l1_unstructured":
            self._apply_l1_unstructured_pruning(model)
        elif self.pruning_type == "l2_unstructured":
            self._apply_l2_unstructured_pruning(model)
        elif self.pruning_type == "structured":
            self._apply_structured_pruning(model)
        else:
            raise ValueError(
                f"Unsupported pruning type: {self.pruning_type}. "
                "Supported types: l1_unstructured, l2_unstructured, structured"
            )

        self.final_sparsity = self._calculate_sparsity(model)

        logger.info(".1%")

        # Cache the pruned model state
        default_memory_manager.cache_model_state(cache_key, model.state_dict())

        return model

    def _apply_l1_unstructured_pruning(self, model: nn.Module):
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

    def _apply_l2_unstructured_pruning(self, model: nn.Module):
        """Apply L2 unstructured pruning."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate L2 norm for each weight
                weight_l2 = torch.sqrt(torch.sum(module.weight ** 2, dim=1))
                _, indices = torch.topk(
                    weight_l2,
                    int(module.weight.size(0) * (1 - self.amount)),
                    largest=True,
                )
                mask = torch.zeros_like(module.weight)
                mask[indices] = 1
                module.weight.data *= mask

    def _apply_structured_pruning(self, model: nn.Module):
        """Apply structured pruning (remove entire channels/filters)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate importance scores for output channels
                weight_norm = torch.norm(module.weight, p=2, dim=1)
                _, indices = torch.topk(
                    weight_norm,
                    int(module.weight.size(0) * (1 - self.amount)),
                    largest=True,
                )

                # Create mask for selected channels
                mask = torch.zeros(module.weight.size(0), dtype=torch.bool)
                mask[indices] = True

                # Apply mask
                module.weight.data = module.weight.data[mask]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]

                # Update output features
                module.out_features = len(indices)

    def decompress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Decompression not applicable for pruning - return model as-is."""
        logger.warning("Pruning decompression not supported - returning original model")
        return model

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get pruning compression statistics."""
        return {
            "technique": "pruning",
            "type": self.pruning_type,
            "pruning_amount": self.amount,
            "original_sparsity": self.original_sparsity,
            "final_sparsity": self.final_sparsity,
            "sparsity_increase": self.final_sparsity - self.original_sparsity
        }

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

    def __init__(self, pruning_type: str = "l1_unstructured", amount: float = 0.3):
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

    def _apply_l1_unstructured_pruning(self, model: nn.Module):
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

    def _apply_l2_unstructured_pruning(self, model: nn.Module):
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

    def _apply_structured_pruning(self, model: nn.Module):
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

    def get_compression_stats(self) -> Dict[str, Any]:
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

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
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

        # Set teacher to evaluation mode
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

        loss_distill = nn.KLDivLoss(reduction="batchmean")(
            student_softened, teacher_softened
        ) * (self.temperature**2)

        # Combined loss
        total_loss = self.alpha * loss_distill + (1 - self.alpha) * loss_gt

        self.distillation_loss_history.append(total_loss.item())

        return total_loss

    def decompress(self, model: nn.Module, **kwargs) -> nn.Module:
        """Knowledge distillation doesn't require decompression."""
        return model

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get distillation compression statistics."""
        return {
            "technique": "knowledge_distillation",
            "temperature": self.temperature,
            "alpha": self.alpha,
            "avg_distillation_loss": sum(self.distillation_loss_history)
            / len(self.distillation_loss_history)
            if self.distillation_loss_history
            else 0,
            "total_distillation_steps": len(self.distillation_loss_history),
        }


class ModelCompressionManager:
    """
    Manager class for applying multiple compression techniques.

    Provides a unified interface for compressing SAC models with
    quantization, pruning, and knowledge distillation.
    """

    def __init__(self):
        self.compressors = {}
        self.compression_stats = {}

    def add_compressor(self, name: str, compressor: BaseCompressionTechnique):
        """Add a compression technique."""
        self.compressors[name] = compressor
        logger.info(f"Added compressor: {name}")

    def compress_model(
        self, model: nn.Module, techniques: List[str], **kwargs
    ) -> nn.Module:
        """
        Apply multiple compression techniques to a model.

        Args:
            model: Model to compress
            techniques: List of compression technique names to apply
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

    def get_compression_report(self) -> Dict[str, Any]:
        """Get comprehensive compression report."""
        return {
            "compression_stats": self.compression_stats,
            "total_techniques_applied": len(self.compression_stats),
            "techniques": list(self.compression_stats.keys()),
        }

    def save_compressed_model(self, model: nn.Module, path: Union[str, Path]):
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
        self, path: Union[str, Path], model_class: nn.Module
    ) -> nn.Module:
        """Load compressed model from file."""
        path = Path(path)
        checkpoint = torch.load(path)

        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])

        self.compression_stats = checkpoint.get("compression_stats", {})

        logger.info(f"Compressed model loaded from {path}")
        return model


def create_compression_pipeline(
    techniques_config: Dict[str, Dict[str, Any]],
) -> ModelCompressionManager:
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
