"""Minimal model compression shim for tests.

This module provides a tiny API surface so tests importing
`ztb.optimization.model_compression` don't fail during collection.
"""
from typing import Any
from abc import ABC, abstractmethod


class ModelCompressor:
    """Minimal compressor used for tests."""


    def compress(self, model: Any) -> Any:
        """Return model unchanged (no-op).

        Real implementations live under `ztb.training.compression` and friends.
        """
        return model


def compress_model(model: Any, **kwargs) -> Any:
    return ModelCompressor(**kwargs).compress(model)

class BaseCompressionTechnique(ABC):
    """Abstract base class for compression techniques used in tests.

    Defines the minimal abstract API expected by unit tests.
    """

    @abstractmethod
    def compress(self, model: Any, *args, **kwargs) -> Any:
        """Compress the provided model and return the compressed model."""

    @abstractmethod
    def decompress(self, model: Any) -> Any:
        """Restore a compressed model back to original form."""

    @abstractmethod
    def get_compression_stats(self) -> dict:
        """Return lightweight stats about the compression (for reporting)."""



class LowRankApproximator(BaseCompressionTechnique):
    def compress(self, model: Any, *a, **k) -> Any:
        return model

    def decompress(self, model: Any) -> Any:
        return model

    def get_compression_stats(self) -> dict:
        return {"technique": "low_rank", "compression_ratio": 0.0}


class SACPruner(BaseCompressionTechnique):
    def compress(self, model: Any, *a, **k) -> Any:
        return model

    def decompress(self, model: Any) -> Any:
        return model

    def get_compression_stats(self) -> dict:
        return {"technique": "pruning", "compression_ratio": 0.0}


"""Minimal model compression shim for tests.

This module provides a tiny API surface so tests importing
`ztb.optimization.model_compression` don't fail during collection.
"""
from typing import Any
import io
import torch

class KnowledgeDistillationCompressor(BaseCompressionTechnique):
    def compress(self, model: Any, *a, **k) -> Any:
        return model

    def decompress(self, model: Any) -> Any:
        return model

    def get_compression_stats(self) -> dict:
        return {"technique": "distillation", "distilled": False}


class QuantizationCompressor(BaseCompressionTechnique):
    def __init__(self, quantization_type: str = "dynamic"):
        if quantization_type not in ("dynamic", "static", "mixed_precision"):
            raise ValueError("Unsupported quantization type")
        self.quantization_type = quantization_type
        self.quantized_model = None

    def _get_model_size(self, model: Any) -> float:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        size_bytes = buf.getbuffer().nbytes
        return size_bytes / (1024.0 * 1024.0)

    def compress(self, model: Any, calibration_data: Any = None) -> Any:
        # For static/mixed precision tests, perform lightweight conversions
        if self.quantization_type == "mixed_precision":
            for p in model.parameters():
                if p.dtype == torch.float32:
                    p.data = p.data.half()
        # store quantized model placeholder
        self.quantized_model = model
        return model

    def decompress(self, model: Any) -> Any:
        # No-op for stub: return model as-is
        return model

    def get_compression_stats(self) -> dict:
        return {
            "technique": "quantization",
            "type": self.quantization_type,
            "compression_ratio": 0.0,
            "original_size_mb": self._get_model_size(self.quantized_model)
            if self.quantized_model is not None
            else 0.0,
        }


class PruningCompressor(BaseCompressionTechnique):
    def __init__(self, pruning_type: str = "l1_unstructured", amount: float = 0.2):
        supported = ("l1_unstructured", "l2_unstructured", "structured")
        if pruning_type not in supported:
            raise ValueError("Unsupported pruning type")

        self.pruning_type = pruning_type
        self.amount = float(amount)

    def _calculate_sparsity(self, model: Any) -> float:
        total = 0
        zeros = 0
        for p in model.parameters():
            arr = p.detach().cpu().numpy()
            total += arr.size
            zeros += (arr == 0).sum()
        return float(zeros) / float(total) if total > 0 else 0.0

    def compress(self, model: Any) -> Any:
        # Apply a naive pruning: zero out smallest-magnitude weights across params
        for p in model.parameters():
            flat = p.data.view(-1)
            k = max(1, int(self.amount * flat.numel()))
            if k < flat.numel():
                vals, idx = torch.topk(flat.abs(), k, largest=False)
                flat[idx] = 0
        return model

    def decompress(self, model: Any) -> Any:
        # No-op in tests
        return model

    def get_compression_stats(self) -> dict:
        return {"technique": "pruning", "type": self.pruning_type, "amount": self.amount}


class KnowledgeDistillationCompressor(BaseCompressionTechnique):
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher_model = None
        self.student_model = None
        self.distillation_loss_history = []

    def compress(self, student_model: Any, teacher_model: Any = None) -> Any:
        if teacher_model is None:
            raise ValueError("teacher_model must be provided")
        self.student_model = student_model
        self.teacher_model = teacher_model
        return student_model

    def decompress(self, model: Any) -> Any:
        # No-op for stub: return model
        return model

    def get_compression_stats(self) -> dict:
        return {"technique": "distillation", "temperature": self.temperature, "alpha": self.alpha}

    def get_distillation_loss(self, student_logits, teacher_logits, targets, criterion):
        # Simple KD loss: combination of soft targets and student cross-entropy
        t = self.temperature
        student_soft = torch.log_softmax(student_logits / t, dim=1)
        teacher_soft = torch.softmax(teacher_logits / t, dim=1)
        kd_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction="batchmean") * (t * t)
        ce_loss = criterion(student_logits, targets)
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        self.distillation_loss_history.append(float(loss.detach().cpu().numpy()))
        return loss


class ModelCompressionManager:
    def __init__(self):
        self.compressors = {}
        self.compression_stats = {}

    def add_compressor(self, name: str, compressor: BaseCompressionTechnique) -> None:
        self.compressors[name] = compressor

    def compress_model(self, model: Any, techniques: list[str]) -> Any:
        applied = 0
        for t in techniques:
            comp = self.compressors.get(t)
            if comp is None:
                continue
            model = comp.compress(model)
            self.compression_stats[t] = getattr(comp, "get_compression_stats", lambda: {} )()
            applied += 1
        return model if applied > 0 else model

    def get_compression_report(self) -> dict:
        return {
            "compression_stats": self.compression_stats,
            "total_techniques_applied": len(self.compression_stats),
            "techniques": list(self.compression_stats.keys()),
        }

    def save_compressed_model(self, model: Any, path):
        torch.save({"model_state_dict": model.state_dict(), "compression_stats": self.compression_stats}, path)

    def load_compressed_model(self, path, model_class: Any) -> Any:
        ckpt = torch.load(path)
        model = model_class()
        model.load_state_dict(ckpt.get("model_state_dict", {}))
        self.compression_stats = ckpt.get("compression_stats", {})
        return model


def create_compression_pipeline(config: dict) -> ModelCompressionManager:
    manager = ModelCompressionManager()
    for name, cfg in (config or {}).items():
        t = cfg.get("type")
        if t == "quantization":
            manager.add_compressor(name, QuantizationCompressor(cfg.get("quantization_type", "dynamic")))
        elif t == "pruning":
            manager.add_compressor(name, PruningCompressor(cfg.get("pruning_type", "l1_unstructured"), cfg.get("amount", 0.1)))
        elif t == "distillation":
            manager.add_compressor(name, KnowledgeDistillationCompressor(cfg.get("temperature", 1.0), cfg.get("alpha", 0.5)))
    return manager


__all__ = [
    "ModelCompressor",
    "compress_model",
    "BaseCompressionTechnique",
    "LowRankApproximator",
    "SACPruner",
]
