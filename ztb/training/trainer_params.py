"""Minimal trainer params for legacy scripts/tests."""
from dataclasses import dataclass


@dataclass
class SELLMitigationParams:
    enabled: bool = False
    threshold: float = 0.1


__all__ = ["SELLMitigationParams"]
