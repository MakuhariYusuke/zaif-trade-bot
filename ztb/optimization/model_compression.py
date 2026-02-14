"""Lightweight model compression utilities used by optimization-layer tests.

This module intentionally keeps implementations simple and robust while exposing
the same public API names expected by tests and legacy callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import io
from pathlib import Path

import torch


def _iter_model_parameters(model: object) -> list[object]:
    """Best-effort parameter iterator for torch-like models."""
    params_fn = getattr(model, "parameters", None)
    if not callable(params_fn):
        return []
    try:
        return list(params_fn())
    except Exception:
        return []


def _model_state_dict(model: object) -> dict[str, object]:
    """Best-effort state_dict fetch for torch-like models."""
    state_dict_fn = getattr(model, "state_dict", None)
    if not callable(state_dict_fn):
        return {}
    try:
        state = state_dict_fn()
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


class ModelCompressor:
    """Minimal compressor used for tests."""

    def __init__(self, **kwargs: object) -> None:
        self.options = dict(kwargs)

    def compress(self, model: object) -> object:
        """Return model unchanged (no-op)."""
        return model


def compress_model(model: object, **kwargs: object) -> object:
    """Compatibility helper used by tests and older call sites."""
    return ModelCompressor(**kwargs).compress(model)


class BaseCompressionTechnique(ABC):
    """Abstract base class for compression techniques used in tests."""

    @abstractmethod
    def compress(self, model: object, *args: object, **kwargs: object) -> object:
        """Compress the provided model and return the compressed model."""

    @abstractmethod
    def decompress(self, model: object) -> object:
        """Restore a compressed model back to original form."""

    @abstractmethod
    def get_compression_stats(self) -> dict[str, object]:
        """Return lightweight stats about the compression."""


class LowRankApproximator(BaseCompressionTechnique):
    def compress(self, model: object, *args: object, **kwargs: object) -> object:
        return model

    def decompress(self, model: object) -> object:
        return model

    def get_compression_stats(self) -> dict[str, object]:
        return {"technique": "low_rank", "compression_ratio": 0.0}


class SACPruner(BaseCompressionTechnique):
    def compress(self, model: object, *args: object, **kwargs: object) -> object:
        return model

    def decompress(self, model: object) -> object:
        return model

    def get_compression_stats(self) -> dict[str, object]:
        return {"technique": "pruning", "compression_ratio": 0.0}


class QuantizationCompressor(BaseCompressionTechnique):
    """Simple quantization stub used in tests."""

    def __init__(self, quantization_type: str = "dynamic") -> None:
        if quantization_type not in ("dynamic", "static", "mixed_precision"):
            raise ValueError("Unsupported quantization type")
        self.quantization_type = quantization_type
        self.quantized_model: object | None = None

    def _get_model_size(self, model: object) -> float:
        if model is None:
            return 0.0
        state_dict = _model_state_dict(model)
        if not state_dict:
            return 0.0
        try:
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            return buf.getbuffer().nbytes / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def compress(self, model: object, calibration_data: object | None = None) -> object:
        # For static/mixed precision tests, perform lightweight conversions.
        if self.quantization_type == "mixed_precision":
            for p in _iter_model_parameters(model):
                try:
                    dtype = getattr(p, "dtype", None)
                    data = getattr(p, "data", None)
                    if dtype == torch.float32 and data is not None and hasattr(data, "half"):
                        p.data = data.half()
                except Exception:
                    continue
        self.quantized_model = model
        return model

    def decompress(self, model: object) -> object:
        return model

    def get_compression_stats(self) -> dict[str, object]:
        return {
            "technique": "quantization",
            "type": self.quantization_type,
            "compression_ratio": 0.0,
            "original_size_mb": self._get_model_size(self.quantized_model),
        }


class PruningCompressor(BaseCompressionTechnique):
    def __init__(self, pruning_type: str = "l1_unstructured", amount: float = 0.2) -> None:
        supported = ("l1_unstructured", "l2_unstructured", "structured")
        if pruning_type not in supported:
            raise ValueError("Unsupported pruning type")
        self.pruning_type = pruning_type
        self.amount = float(amount)

    def _calculate_sparsity(self, model: object) -> float:
        total = 0
        zeros = 0
        for p in _iter_model_parameters(model):
            try:
                arr = p.detach().cpu().numpy()
                total += int(arr.size)
                zeros += int((arr == 0).sum())
            except Exception:
                continue
        return float(zeros) / float(total) if total > 0 else 0.0

    def compress(self, model: object, *args: object, **kwargs: object) -> object:
        # Apply a naive pruning: zero out smallest-magnitude weights across params.
        for p in _iter_model_parameters(model):
            try:
                data = getattr(p, "data", None)
                if data is None or not hasattr(data, "view") or not hasattr(data, "numel"):
                    continue
                flat = data.view(-1)
                numel = int(flat.numel())
                if numel <= 0:
                    continue
                k = max(1, int(self.amount * numel))
                if k >= numel:
                    continue
                _, idx = torch.topk(flat.abs(), k, largest=False)
                flat[idx] = 0
            except Exception:
                continue
        return model

    def decompress(self, model: object) -> object:
        return model

    def get_compression_stats(self) -> dict[str, object]:
        return {"technique": "pruning", "type": self.pruning_type, "amount": self.amount}


class KnowledgeDistillationCompressor(BaseCompressionTechnique):
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5) -> None:
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher_model: object | None = None
        self.student_model: object | None = None
        self.distillation_loss_history: list[float] = []

    def compress(self, student_model: object, teacher_model: object | None = None) -> object:
        if teacher_model is None:
            raise ValueError("teacher_model must be provided")
        self.student_model = student_model
        self.teacher_model = teacher_model
        return student_model

    def decompress(self, model: object) -> object:
        return model

    def get_compression_stats(self) -> dict[str, object]:
        return {
            "technique": "distillation",
            "temperature": self.temperature,
            "alpha": self.alpha,
        }

    def get_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        criterion: object,
    ) -> torch.Tensor:
        """Compute a simple distillation loss."""
        t = self.temperature
        student_soft = torch.log_softmax(student_logits / t, dim=1)
        teacher_soft = torch.softmax(teacher_logits / t, dim=1)
        kd_loss = torch.nn.functional.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (t * t)
        if not callable(criterion):
            raise ValueError("criterion must be callable")
        ce_loss = criterion(student_logits, targets)
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        self.distillation_loss_history.append(float(loss.detach().cpu().numpy()))
        return loss


class ModelCompressionManager:
    def __init__(self) -> None:
        self.compressors: dict[str, BaseCompressionTechnique] = {}
        self.compression_stats: dict[str, object] = {}

    def add_compressor(self, name: str, compressor: BaseCompressionTechnique) -> None:
        self.compressors[name] = compressor

    def compress_model(self, model: object, techniques: list[str]) -> object:
        for technique in techniques:
            compressor = self.compressors.get(technique)
            if compressor is None:
                continue
            model = compressor.compress(model)
            self.compression_stats[technique] = compressor.get_compression_stats()
        return model

    def get_compression_report(self) -> dict[str, object]:
        return {
            "compression_stats": self.compression_stats,
            "total_techniques_applied": len(self.compression_stats),
            "techniques": list(self.compression_stats.keys()),
        }

    def save_compressed_model(self, model: object, path: str | Path) -> None:
        torch.save(
            {
                "model_state_dict": _model_state_dict(model),
                "compression_stats": self.compression_stats,
            },
            path,
        )

    def load_compressed_model(self, path: str | Path, model_class: object) -> object:
        checkpoint = torch.load(path)
        if not callable(model_class):
            raise ValueError("model_class must be callable")
        model = model_class()
        load_fn = getattr(model, "load_state_dict", None)
        if callable(load_fn):
            load_fn(checkpoint.get("model_state_dict", {}))
        self.compression_stats = checkpoint.get("compression_stats", {})
        return model


def create_compression_pipeline(
    config: dict[str, dict[str, object]] | None,
) -> ModelCompressionManager:
    manager = ModelCompressionManager()
    for name, cfg in (config or {}).items():
        technique_type = cfg.get("type")
        if technique_type == "quantization":
            manager.add_compressor(
                name,
                QuantizationCompressor(str(cfg.get("quantization_type", "dynamic"))),
            )
        elif technique_type == "pruning":
            manager.add_compressor(
                name,
                PruningCompressor(
                    str(cfg.get("pruning_type", "l1_unstructured")),
                    float(cfg.get("amount", 0.1)),
                ),
            )
        elif technique_type == "distillation":
            manager.add_compressor(
                name,
                KnowledgeDistillationCompressor(
                    float(cfg.get("temperature", 1.0)),
                    float(cfg.get("alpha", 0.5)),
                ),
            )
    return manager


__all__ = [
    "ModelCompressor",
    "compress_model",
    "BaseCompressionTechnique",
    "LowRankApproximator",
    "SACPruner",
    "QuantizationCompressor",
    "PruningCompressor",
    "KnowledgeDistillationCompressor",
    "ModelCompressionManager",
    "create_compression_pipeline",
]
