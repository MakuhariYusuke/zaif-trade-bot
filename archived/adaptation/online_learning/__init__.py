"""
Online Learning Pipeline Module
ストリーミングデータでのインクリメンタル学習システム

Features:
- Incremental learning: ミニバッチ/オンライン更新
- Streaming data processing: Kafka/Redis統合
- Model versioning: 自動バージョン管理
- Memory management: GPU/CPUメモリ最適化
"""

from typing import TYPE_CHECKING

from .config import OnlineLearningConfig

if TYPE_CHECKING:
    # Import runtime-heavy modules only for type checking; avoid importing torch during package import
    from .pipeline import DriftDetector, OnlineLearningPipeline, ResourceMonitor
    from .types import DataBatch, LearningState, UpdateResult
else:
    # Provide lightweight placeholders in runtime to avoid import-time heavy dependency loads
    DriftDetector = None  # type: ignore
    OnlineLearningPipeline = None  # type: ignore
    ResourceMonitor = None  # type: ignore
    DataBatch = None  # type: ignore
    LearningState = None  # type: ignore
    UpdateResult = None  # type: ignore

__all__ = [
    "OnlineLearningPipeline",
    "DriftDetector",
    "ResourceMonitor",
    "OnlineLearningConfig",
    "LearningState",
    "UpdateResult",
    "DataBatch",
]
