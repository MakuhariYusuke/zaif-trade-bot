"""
Online Learning Pipeline Module
ストリーミングデータでのインクリメンタル学習システム

Features:
- Incremental learning: ミニバッチ/オンライン更新
- Streaming data processing: Kafka/Redis統合
- Model versioning: 自動バージョン管理
- Memory management: GPU/CPUメモリ最適化
"""

from .pipeline import OnlineLearningPipeline, DriftDetector, ResourceMonitor
from .config import OnlineLearningConfig
from .types import LearningState, UpdateResult, DataBatch

__all__ = [
    "OnlineLearningPipeline",
    "DriftDetector",
    "ResourceMonitor",
    "OnlineLearningConfig",
    "LearningState",
    "UpdateResult",
    "DataBatch",
]