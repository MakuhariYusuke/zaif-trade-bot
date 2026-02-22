"""v460 ML model protocol definitions.

Any 型を使わず、学習器/変換器の最小インターフェースを共有する。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ProbabilisticEstimator(Protocol):
    """分類器の最小契約: fit + predict_proba."""

    def fit(self, X: object, y: object) -> object: ...
    def predict_proba(self, X: object) -> np.ndarray: ...


class RegressorEstimator(Protocol):
    """回帰器の最小契約: fit + predict."""

    def fit(self, X: object, y: object) -> object: ...
    def predict(self, X: object) -> np.ndarray: ...


class FeatureTransformer(Protocol):
    """前処理器の最小契約: transform."""

    def transform(self, X: object) -> np.ndarray: ...


PipelineStep = tuple[str, object]
