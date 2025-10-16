"""
データ処理モジュール

金融時系列データに対する包括的なデータ処理機能を提供：
- データ拡張（Data Augmentation）
- 異常値検出・処理（Outlier Detection & Handling）
- データバリデーション（Data Validation）
- データ処理パイプライン（Data Processing Pipeline）
- ストリーミングデータ処理（Streaming Data Processing）
"""

from .coin_gecko_stream import CoinGeckoStream, MarketDataBatch, StreamConfig
from .data_loader import (
    analyze_feature_distributions,
    detect_outliers_iqr,
    detect_outliers_zscore,
)
from .stream_buffer import BufferStats, StreamBuffer
from .streaming_pipeline import PipelineStats, StreamingPipeline

# 新しいデータ処理モジュール
from .data_augmentation import DataAugmentation
from .outlier_detection import OutlierDetector, OutlierHandler
from .data_validation import DataValidator, DataIntegrityChecker, ValidationResult, DataQualityMetrics
from .data_processing_pipeline import (
    DataProcessingPipeline,
    PipelineResult,
    create_financial_data_pipeline
)

__all__ = [
    # 既存のストリーミング機能
    "analyze_feature_distributions",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "StreamBuffer",
    "BufferStats",
    "CoinGeckoStream",
    "StreamConfig",
    "MarketDataBatch",
    "StreamingPipeline",
    "PipelineStats",

    # 新しいデータ処理機能
    'DataAugmentation',
    'OutlierDetector',
    'OutlierHandler',
    'DataValidator',
    'DataIntegrityChecker',
    'ValidationResult',
    'DataQualityMetrics',
    'DataProcessingPipeline',
    'PipelineResult',
    'create_financial_data_pipeline'
]

__version__ = "1.0.0"
