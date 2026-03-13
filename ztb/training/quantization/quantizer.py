"""
SAC v421 Model Quantization Module

This module provides dynamic quantization capabilities for SAC models,
enabling model compression and inference speed optimization while maintaining accuracy.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ztb.trading.environment.constants import BYTES_PER_MB

logger = logging.getLogger(__name__)

class SACQuantizer:
    """
    Dynamic quantizer for SAC models with precision vs speed trade-off analysis.
    """

    def __init__(self, quantization_config: dict | None = None):
        """
        Initialize the SAC quantizer.

        Args:
            quantization_config: Configuration dictionary for quantization settings
        """
        default = self._get_default_config()
        if quantization_config:
            # デフォルトを上書き
            merged = default.copy()
            merged.update(quantization_config)
            self.config = merged
        else:
            self.config = default
        self.quantized_models = {}
        self.quantization_stats = {}

    def _get_default_config(self) -> dict:
        """Get default quantization configuration."""
        return {
            "dtype": torch.qint8,  # Default quantization dtype
            "quantizable_modules": [
                nn.Linear,
                nn.LSTM,
                nn.LSTMCell,
                nn.GRU,
                nn.GRUCell,
                nn.Conv1d,
                nn.Conv2d,
                nn.Embedding,
            ],
            "skip_modules": [],  # Modules to skip quantization
            "accuracy_tolerance": 0.01,  # 1% accuracy drop tolerance
            "performance_target": "balanced",  # 'speed', 'size', 'balanced'
            "enable_fusion": True,  # Enable operator fusion
            "calibration_samples": 1000,  # Number of samples for calibration
        }

    def analyze_model(self, model: nn.Module) -> dict:
        """
        Analyze model for quantization compatibility and estimate compression.

        Args:
            model: PyTorch model to analyze

        Returns:
            Analysis results dictionary
        """
        analysis = {
            "total_parameters": 0,
            "quantizable_parameters": 0,
            "quantizable_modules": [],
            "estimated_compression_ratio": 0.0,
            "estimated_speedup": 0.0,
            "risk_assessment": "low",
        }

        for name, module in model.named_modules():
            if not list(module.children()):  # Leaf module
                param_count = sum(p.numel() for p in module.parameters())
                analysis["total_parameters"] += param_count

                if self._is_quantizable_module(module):
                    analysis["quantizable_parameters"] += param_count
                    analysis["quantizable_modules"].append(
                        {
                            "name": name,
                            "type": type(module).__name__,
                            "parameters": param_count,
                        }
                    )

        if analysis["total_parameters"] > 0:
            compression_ratio = (
                analysis["quantizable_parameters"] / analysis["total_parameters"]
            )
            analysis["estimated_compression_ratio"] = compression_ratio
            analysis["estimated_speedup"] = self._estimate_speedup(compression_ratio)

            # Risk assessment based on compression ratio
            if compression_ratio > 0.8:
                analysis["risk_assessment"] = "high"
            elif compression_ratio > 0.5:
                analysis["risk_assessment"] = "medium"
            else:
                analysis["risk_assessment"] = "low"

        return analysis

    def _is_quantizable_module(self, module: nn.Module) -> bool:
        """Check if a module is quantizable."""
        return (
            type(module) in self.config["quantizable_modules"]
            and type(module) not in self.config["skip_modules"]
        )

    def _estimate_speedup(self, compression_ratio: float) -> float:
        """Estimate inference speedup based on compression ratio."""
        # Empirical estimation: higher compression generally means higher speedup
        base_speedup = 1.5
        ratio_bonus = compression_ratio * 2.0
        return base_speedup + ratio_bonus

    def quantize_model(
        self, model: nn.Module, calibration_data: torch.Tensor | None = None
    ) -> tuple[nn.Module, dict]:
        """
        Quantize the model using dynamic quantization.

        Args:
            model: Model to quantize
            calibration_data: Optional calibration data for static quantization

        Returns:
            tuple of (quantized_model, quantization_stats)
        """
        logger.info("Starting model quantization...")

        # Analyze model first
        analysis = self.analyze_model(model)
        logger.info(
            f"Model analysis: {analysis['estimated_compression_ratio']:.2%} quantizable"
        )

        if analysis["quantizable_parameters"] == 0:
            logger.warning("No quantizable modules found, returning original model")
            return model, {"status": "skipped", "reason": "no_quantizable_modules"}

        # Apply dynamic quantization
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                dict.fromkeys(self.config["quantizable_modules"], self.config["dtype"]),
                inplace=False,
            )

            # Collect quantization statistics
            stats = self._collect_quantization_stats(model, quantized_model, analysis)

            logger.info(
                f"Quantization completed. Compression: {stats['compression_ratio']:.2%}"
            )
            return quantized_model, stats

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model, {"status": "failed", "error": str(e)}

    def _collect_quantization_stats(
        self, original_model: nn.Module, quantized_model: nn.Module, analysis: dict
    ) -> dict:
        """Collect comprehensive quantization statistics."""
        stats = {
            "status": "success",
            "original_size_mb": self._get_model_size_mb(original_model),
            "quantized_size_mb": self._get_model_size_mb(quantized_model),
            "compression_ratio": 0.0,
            "quantized_modules": len(analysis["quantizable_modules"]),
            "quantization_dtype": str(self.config["dtype"]),
            "performance_target": self.config["performance_target"],
        }

        if stats["original_size_mb"] > 0:
            stats["compression_ratio"] = 1.0 - (
                stats["quantized_size_mb"] / stats["original_size_mb"]
            )

        return stats

    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / BYTES_PER_MB

    def validate_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        validation_data: torch.Tensor,
        validation_labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Validate quantization quality by comparing outputs.

        Args:
            original_model: Original model
            quantized_model: Quantized model
            validation_data: Validation input data
            validation_labels: Optional validation labels for accuracy comparison

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "output_similarity": 0.0,
            "accuracy_drop": 0.0,
            "max_output_diff": 0.0,
            "within_tolerance": True,
        }

        try:
            with torch.no_grad():
                original_output = original_model(validation_data)
                quantized_output = quantized_model(validation_data)

                # Calculate output similarity (cosine similarity)
                original_flat = original_output.view(-1)
                quantized_flat = quantized_output.view(-1)

                similarity = torch.cosine_similarity(
                    original_flat, quantized_flat, dim=0
                )
                validation_results["output_similarity"] = similarity.item()

                # Calculate maximum absolute difference
                max_diff = torch.max(torch.abs(original_output - quantized_output))
                validation_results["max_output_diff"] = max_diff.item()

                # Check if within tolerance
                tolerance = self.config["accuracy_tolerance"]
                validation_results["within_tolerance"] = (
                    1.0 - similarity.item()
                ) <= tolerance

                logger.info(
                    f"Validation results: similarity={similarity:.4f}, "
                    f"max_diff={max_diff:.6f}, within_tolerance={validation_results['within_tolerance']}"
                )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["error"] = str(e)

        return validation_results

    def save_quantized_model(
        self, model: nn.Module, path: str | Path, metadata: dict | None = None
    ) -> bool:
        """
        Save quantized model with metadata.

        Args:
            model: Quantized model to save
            path: Save path
            metadata: Optional metadata to save

        Returns:
            Success status
        """
        try:
            save_dict = {
                "model_state_dict": model.state_dict(),
                "model_config": self.config,
                "quantization_metadata": metadata or {},
                "quantization_type": "dynamic",
            }

            torch.save(save_dict, path)
            logger.info(f"Quantized model saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save quantized model: {e}")
            return False

    def load_quantized_model(
        self, path: str | Path
    ) -> tuple[nn.Module | None, dict]:
        """
        Load quantized model with metadata.

        Args:
            path: Model path

        Returns:
            tuple of (model, metadata)
        """
        try:
            checkpoint = torch.load(path, map_location="cpu")
            metadata = checkpoint.get("quantization_metadata", {})

            # Note: Need model architecture to load state dict
            # This would need to be implemented based on specific model type
            logger.info(f"Quantized model metadata loaded from {path}")
            return (
                None,
                metadata,
            )  # Return None for model, implement based on model type

        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            return None, {}

class QuantizationPipeline:
    """
    End-to-end quantization pipeline for SAC models.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.quantizer = SACQuantizer(self.config.get("quantization", {}))

    def run_pipeline(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor | None = None,
        validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """
        Run complete quantization pipeline.

        Args:
            model: Model to quantize
            calibration_data: Optional calibration data
            validation_data: Optional (input, label) validation data

        Returns:
            Pipeline results dictionary
        """
        results = {
            "success": False,
            "analysis": {},
            "quantization_stats": {},
            "validation_results": {},
            "recommendations": [],
        }

        try:
            # Step 1: Analyze model
            logger.info("Step 1: Analyzing model...")
            results["analysis"] = self.quantizer.analyze_model(model)

            # Step 2: Quantize model
            logger.info("Step 2: Quantizing model...")
            quantized_model, stats = self.quantizer.quantize_model(
                model, calibration_data
            )
            results["quantization_stats"] = stats

            if stats.get("status") != "success":
                results["recommendations"].append(
                    "Quantization failed, check model compatibility"
                )
                return results

            # Step 3: Validate quantization (if validation data provided)
            if validation_data:
                logger.info("Step 3: Validating quantization...")
                val_input, val_labels = validation_data
                results["validation_results"] = self.quantizer.validate_quantization(
                    model, quantized_model, val_input, val_labels
                )

                if not results["validation_results"].get("within_tolerance", True):
                    results["recommendations"].append(
                        "Quantization accuracy drop exceeds tolerance, consider reducing quantization aggressiveness"
                    )

            results["quantized_model"] = quantized_model
            results["success"] = True
            logger.info("Quantization pipeline completed successfully")

        except Exception as e:
            logger.error(f"Quantization pipeline failed: {e}")
            results["error"] = str(e)

        return results
