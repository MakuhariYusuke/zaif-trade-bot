"""Minimal torch layers used by tests (features extractor shims)."""
from typing import Any, Dict, Optional
import torch.nn as nn


class BaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: Any, features_dim: int = 1):
        super().__init__()
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Any, features_dim: int = 1):
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations: Any) -> Any:
        # Minimal flatten implementation (tests use shape info only)
        import torch

        if isinstance(observations, torch.Tensor):
            return observations.view(observations.size(0), -1)
        return observations


__all__ = ["BaseFeaturesExtractor", "FlattenExtractor"]
