"""Data processing package with lazy export loading."""

from __future__ import annotations

from importlib import import_module

_LAZY_MODULE_ATTRS: dict[str, tuple[str, str]] = {
    "analyze_feature_distributions": ("ztb.data.data_loader", "analyze_feature_distributions"),
    "detect_outliers_iqr": ("ztb.data.data_loader", "detect_outliers_iqr"),
    "detect_outliers_zscore": ("ztb.data.data_loader", "detect_outliers_zscore"),
    "StreamBuffer": ("ztb.data.stream_buffer", "StreamBuffer"),
    "BufferStats": ("ztb.data.stream_buffer", "BufferStats"),
    "CoinGeckoStream": ("ztb.data.coin_gecko_stream", "CoinGeckoStream"),
    "StreamConfig": ("ztb.data.coin_gecko_stream", "StreamConfig"),
    "MarketDataBatch": ("ztb.data.coin_gecko_stream", "MarketDataBatch"),
    "StreamingPipeline": ("ztb.data.streaming_pipeline", "StreamingPipeline"),
    "PipelineStats": ("ztb.data.streaming_pipeline", "PipelineStats"),
    "BTCDataAugmentor": ("ztb.data.btc_data_augmentation", "BTCDataAugmentor"),
    "BTCBiasDetector": ("ztb.data.btc_data_augmentation", "BTCBiasDetector"),
    "DataAugmentation": ("ztb.data.data_augmentation", "DataAugmentation"),
    "OutlierDetector": ("ztb.data.outlier_detection", "OutlierDetector"),
    "OutlierHandler": ("ztb.data.outlier_detection", "OutlierHandler"),
    "DataValidator": ("ztb.data.data_validation", "DataValidator"),
    "DataIntegrityChecker": ("ztb.data.data_validation", "DataIntegrityChecker"),
    "ValidationResult": ("ztb.data.data_validation", "ValidationResult"),
    "DataQualityMetrics": ("ztb.data.data_validation", "DataQualityMetrics"),
    "DataProcessingPipeline": ("ztb.data.data_processing_pipeline", "DataProcessingPipeline"),
    "PipelineResult": ("ztb.data.data_processing_pipeline", "PipelineResult"),
    "create_financial_data_pipeline": ("ztb.data.data_processing_pipeline", "create_financial_data_pipeline"),
}

__all__ = list(_LAZY_MODULE_ATTRS)

__version__ = "1.0.0"


def __getattr__(name: str) -> object:
    target = _LAZY_MODULE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
