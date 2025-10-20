"""
データ処理モジュール

金融時系列データに対する包括的なデータ処理機能を提供：
- データ拡張（Data Augmentation）
- 異常値検出・処理（Outlier Detection & Handling）
- データバリデーション（Data Validation）
- データ処理パイプライン（Data Processing Pipeline）
- ストリーミングデータ処理（Streaming Data Processing）
"""

from .btc_data_augmentation import BTCBiasDetector, BTCDataAugmentor
from .coin_gecko_stream import CoinGeckoStream, MarketDataBatch, StreamConfig

# 新しいデータ処理モジュール
from .data_augmentation import DataAugmentation
from .data_loader import (
    analyze_feature_distributions,
    detect_outliers_iqr,
    detect_outliers_zscore,
)
from .data_processing_pipeline import (
    DataProcessingPipeline,
    PipelineResult,
    create_financial_data_pipeline,
)
from .data_validation import (
    DataIntegrityChecker,
    DataQualityMetrics,
    DataValidator,
    ValidationResult,
)
from .outlier_detection import OutlierDetector, OutlierHandler
from .stream_buffer import BufferStats, StreamBuffer

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
    # BTCデータ拡張機能
    "BTCDataAugmentor",
    "BTCBiasDetector",
    # 新しいデータ処理機能
    "DataAugmentation",
    "OutlierDetector",
    "OutlierHandler",
    "DataValidator",
    "DataIntegrityChecker",
    "ValidationResult",
    "DataQualityMetrics",
    "DataProcessingPipeline",
    "PipelineResult",
    "create_financial_data_pipeline",
]

__version__ = "1.0.0"
