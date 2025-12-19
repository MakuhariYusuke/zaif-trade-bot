"""
Advanced Model Compression Pipeline for SAC v421

This module provides a comprehensive compression pipeline that combines
pruning, quantization, low-rank approximation, and knowledge distillation
for optimal model compression with minimal accuracy loss.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.training.compression.compressor import LowRankApproximator, SACPruner
from ztb.training.distillation.distiller import SACDistiller
from ztb.training.quantization.quantizer import SACQuantizer

logger = logging.getLogger(__name__)


class CompressionMetrics:
    """Metrics for tracking compression performance."""

    def __init__(self):
        self.original_size = 0
        self.compressed_size = 0
        self.compression_ratio = 1.0
        self.accuracy_drop = 0.0
        self.compression_time = 0.0
        self.memory_savings = 0.0

    def calculate_metrics(self, original_model: nn.Module, compressed_model: nn.Module):
        """Calculate compression metrics."""
        self.original_size = self._calculate_model_size(original_model)
        self.compressed_size = self._calculate_model_size(compressed_model)
        self.compression_ratio = self.original_size / max(self.compressed_size, 1e-6)
        self.memory_savings = (1 - self.compressed_size / self.original_size) * 100

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / BYTES_PER_MB


class CompositeCompressor:
    """
    Composite model compression pipeline combining multiple techniques.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize composite compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or self._get_default_config()
        self.metrics = CompressionMetrics()

        # Initialize compression components
        self.quantizer = SACQuantizer(self.config.get("quantization", {}))
        self.distiller = SACDistiller(self.config.get("distillation", {}))
        self.pruner = SACPruner(self.config.get("pruning", {}))
        self.low_rank = LowRankApproximator(self.config.get("low_rank", {}))

    def _get_default_config(self) -> Dict:
        """Get default compression configuration."""
        return {
            "pipeline": ["pruning", "quantization", "low_rank", "distillation"],
            "target_compression_ratio": 0.5,
            "max_accuracy_drop": 0.05,
            "pruning": {
                "method": "l1_unstructured",
                "amount": 0.3,
                "target_modules": [nn.Linear],
            },
            "quantization": {"method": "dynamic", "dtype": torch.qint8},
            "low_rank": {"rank_ratio": 0.8, "target_modules": [nn.Linear]},
            "distillation": {"temperature": 2.0, "alpha": 0.5},
        }

    def compress_model(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Compress model using composite pipeline.

        Args:
            model: Model to compress
            teacher_model: Teacher model for distillation
            calibration_data: Data for quantization calibration

        Returns:
            Compressed model
        """
        logger.info("Starting composite model compression")
        start_time = time.time()

        original_model = self._copy_model(model)
        compressed_model = self._copy_model(model)

        # Apply compression pipeline
        pipeline = self.config["pipeline"]
        for technique in pipeline:
            logger.info(f"Applying {technique} compression")
            if technique == "pruning":
                compressed_model = self._apply_pruning(compressed_model)
            elif technique == "quantization":
                compressed_model = self._apply_quantization(
                    compressed_model, calibration_data
                )
            elif technique == "low_rank":
                compressed_model = self._apply_low_rank(compressed_model)
            elif technique == "distillation" and teacher_model is not None:
                compressed_model = self._apply_distillation(
                    compressed_model, teacher_model
                )

        # Calculate metrics
        self.metrics.calculate_metrics(original_model, compressed_model)
        self.metrics.compression_time = time.time() - start_time

        logger.info(
            f"Compression completed. Ratio: {self.metrics.compression_ratio:.2f}x, "
            f"Size reduction: {self.metrics.memory_savings:.1f}%"
        )

        return compressed_model

    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning compression."""
        try:
            res = self.pruner.apply_pruning(model)
            # Some implementations return (model, stats)
            if isinstance(res, tuple) and len(res) >= 1:
                pruned_model = res[0]
                return pruned_model
            return res
        except Exception as e:
            logger.warning(f"Pruning failed: {e}, skipping")
            return model

    def _apply_quantization(
        self, model: nn.Module, calibration_data: Optional[torch.Tensor]
    ) -> nn.Module:
        """Apply quantization compression."""
        try:
            res = self.quantizer.quantize_model(model, calibration_data)
            if isinstance(res, tuple) and len(res) >= 1:
                quantized_model = res[0]
                return quantized_model
            return res
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, skipping")
            return model

    def _apply_low_rank(self, model: nn.Module) -> nn.Module:
        """Apply low-rank approximation."""
        try:
            res = self.low_rank.apply_low_rank_approximation(model)
            if isinstance(res, tuple) and len(res) >= 1:
                low_rank_model = res[0]
                return low_rank_model
            return res
        except Exception as e:
            logger.warning(f"Low-rank approximation failed: {e}, skipping")
            return model

    def _apply_distillation(
        self, model: nn.Module, teacher_model: nn.Module
    ) -> nn.Module:
        """Apply knowledge distillation."""
        try:
            res = self.distiller.distill_model(model, teacher_model)
            if isinstance(res, tuple) and len(res) >= 1:
                distilled_model = res[0]
                return distilled_model
            return res
        except Exception as e:
            logger.warning(f"Distillation failed: {e}, skipping")
            return model

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy

        return copy.deepcopy(model)

    def get_compression_report(self) -> Dict:
        """Get compression performance report."""
        return {
            "compression_ratio": self.metrics.compression_ratio,
            "original_size_mb": self.metrics.original_size,
            "compressed_size_mb": self.metrics.compressed_size,
            "memory_savings_percent": self.metrics.memory_savings,
            "compression_time_seconds": self.metrics.compression_time,
            "accuracy_drop": self.metrics.accuracy_drop,
        }


class AdaptiveCompressor:
    """
    Adaptive compression that adjusts techniques based on model characteristics.
    """

    def __init__(self, target_compression_ratio: float = 0.5):
        """
        Initialize adaptive compressor.

        Args:
            target_compression_ratio: Target compression ratio
        """
        self.target_ratio = target_compression_ratio
        self.composite_compressor = CompositeCompressor()

    def compress_adaptively(self, model: nn.Module) -> nn.Module:
        """
        Compress model adaptively based on analysis.

        Args:
            model: Model to compress

        Returns:
            Compressed model
        """
        # Analyze model characteristics
        model_analysis = self._analyze_model(model)

        # Select optimal compression pipeline
        pipeline_config = self._select_optimal_pipeline(model_analysis)

        # Apply compression
        self.composite_compressor.config.update(pipeline_config)
        compressed_model = self.composite_compressor.compress_model(model)

        return compressed_model

    def _analyze_model(self, model: nn.Module) -> Dict:
        """Analyze model characteristics for compression."""
        analysis = {
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_layers": len(list(model.modules())),
            "layer_types": {},
            "parameter_distribution": {},
        }

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                analysis["layer_types"]["linear"] = (
                    analysis["layer_types"].get("linear", 0) + 1
                )
            elif isinstance(module, nn.Conv2d):
                analysis["layer_types"]["conv2d"] = (
                    analysis["layer_types"].get("conv2d", 0) + 1
                )
            elif isinstance(module, nn.LSTM):
                analysis["layer_types"]["lstm"] = (
                    analysis["layer_types"].get("lstm", 0) + 1
                )

        return analysis

    def _select_optimal_pipeline(self, analysis: Dict) -> Dict:
        """Select optimal compression pipeline based on analysis."""
        pipeline = []

        # Always include pruning for parameter reduction
        pipeline.append("pruning")

        # Add quantization for inference speedup
        pipeline.append("quantization")

        # Add low-rank for linear layers
        if analysis["layer_types"].get("linear", 0) > 0:
            pipeline.append("low_rank")

        # Distillation can be added later with teacher model
        # pipeline.append('distillation')

        return {"pipeline": pipeline, "target_compression_ratio": self.target_ratio}


# Utility functions
def compress_model_pipeline(
    model: nn.Module,
    compression_ratio: float = 0.5,
    techniques: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function for model compression.

    Args:
        model: Model to compress
        compression_ratio: Target compression ratio
        techniques: List of compression techniques

    Returns:
        Tuple of (compressed_model, compression_report)
    """
    if techniques:
        config = {"pipeline": techniques, "target_compression_ratio": compression_ratio}
        compressor = CompositeCompressor(config)
    else:
        compressor = AdaptiveCompressor(compression_ratio)

    if isinstance(compressor, AdaptiveCompressor):
        compressed_model = compressor.compress_adaptively(model)
    else:
        compressed_model = compressor.compress_model(model)

    report = compressor.get_compression_report()
    return compressed_model, report


def benchmark_compression(
    original_model: nn.Module,
    compressed_model: nn.Module,
    test_data: torch.Tensor,
    num_runs: int = 10,
) -> Dict:
    """
    Benchmark compression performance.

    Args:
        original_model: Original model
        compressed_model: Compressed model
        test_data: Test data for inference
        num_runs: Number of benchmark runs

    Returns:
        Benchmark results
    """
    device = next(original_model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = original_model(test_data)
            _ = compressed_model(test_data)

    # Benchmark original model (use perf_counter for higher resolution)
    original_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = original_model(test_data)
            torch.cuda.synchronize() if device.type == "cuda" else None
            original_times.append(time.perf_counter() - start)

    # Benchmark compressed model
    compressed_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = compressed_model(test_data)
            torch.cuda.synchronize() if device.type == "cuda" else None
            compressed_times.append(time.perf_counter() - start)

    return {
        "original_avg_time": np.mean(original_times),
        "compressed_avg_time": np.mean(compressed_times),
        "speedup_ratio": np.mean(original_times) / np.mean(compressed_times),
        "original_std": np.std(original_times),
        "compressed_std": np.std(compressed_times),
    }
