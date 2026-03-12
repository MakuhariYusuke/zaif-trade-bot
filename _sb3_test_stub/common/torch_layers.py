"""Minimal torch layer shims for SB3 imports."""
from __future__ import annotations

try:
    import torch.nn as nn
except Exception:  # pragma: no cover
    class _NNModule:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class _NN:  # pragma: no cover
        Module = _NNModule

    nn = _NN()  # type: ignore[assignment]


class BaseFeaturesExtractor(nn.Module):  # type: ignore[misc]
    def __init__(self, observation_space: object, features_dim: int = 1) -> None:
        super().__init__()
        self._features_dim = int(features_dim)

    @property
    def features_dim(self) -> int:
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    def forward(self, observations: object) -> object:  # pragma: no cover
        return observations
