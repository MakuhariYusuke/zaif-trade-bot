"""
Streaming Processor Implementation for Action Signal Guide.

This module implements real-time data processing and streaming analytics
for adaptive signal generation and performance monitoring.
"""

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..config.asg_adaptation_config import (
    RealTimeAdaptationConfig,
    StreamingProcessorConfig,
)
from ..interfaces.adaptation_interfaces import (
    IStreamingProcessor,
    ProcessingResult,
    StreamingDataPoint,
    StreamingDataType,
)
from ztb.types.common import ObjectMap, ObjectRecords
from ztb.utils.safety import ensure_dict, safe_to_float

logger = logging.getLogger(__name__)


@dataclass
class StreamProcessingResult:
    """Result of streaming data processing."""

    processed_data: ObjectRecords = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    anomalies_detected: ObjectRecords = field(default_factory=list)
    processing_time: float = 0.0
    data_quality_score: float = 1.0
    throughput: float = 0.0


def _as_object_map(value: object) -> ObjectMap:
    """Safely convert arbitrary payloads into a mutable object map."""
    return ensure_dict(value)


def _as_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion used by feature/meter calculations."""
    return safe_to_float(value, default)


class BaseStreamingProcessor(IStreamingProcessor):
    """Base implementation of streaming processor."""

    def __init__(self, config: StreamingProcessorConfig):
        self.config = config
        self.data_queue: queue.Queue[StreamingDataPoint] = queue.Queue(
            maxsize=config.max_queue_size
        )
        self.processed_data: deque[StreamingDataPoint] = deque(
            maxlen=max(1, config.max_queue_size)
        )
        self.anomaly_detector: ObjectMap | None = None
        self.quality_monitor: ObjectMap | None = None
        self.is_running = False
        self.executor: ThreadPoolExecutor | None = None
        self.processing_thread: threading.Thread | None = None
        self.metrics_collector: ObjectMap | None = None
        self.data_handlers: dict[
            StreamingDataType, list[Callable[[StreamingDataPoint], None]]
        ] = {}
        self.processed_data_cache: dict[str, ObjectMap] = {}
        self.max_cache_size = max(config.max_queue_size, config.buffer_size)

    def start_processing(self) -> bool:
        """Start the streaming data processing."""
        try:
            if self.is_running:
                logger.warning("Streaming processor already running")
                return True

            self.is_running = True

            # Initialize components
            self._initialize_components()

            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # Start executor if parallel processing enabled
            if self.config.enable_parallel_processing:
                self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

            logger.info("Streaming processor started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming processor: {e}")
            self.is_running = False
            return False

    def stop_processing(self) -> bool:
        """Stop the streaming data processing."""
        try:
            self.is_running = False

            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)

            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)

            logger.info("Streaming processor stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop streaming processor: {e}")
            return False

    def add_data_point(self, data_point: StreamingDataPoint) -> bool:
        """Add a data point to the processing queue."""
        try:
            if not self.is_running:
                logger.warning("Streaming processor not running")
                return False

            # Validate data point
            if not self._validate_data_point(data_point):
                logger.warning("Invalid data point rejected")
                return False

            # Add to queue with timeout
            self.data_queue.put(data_point, timeout=1.0)
            return True

        except queue.Full:
            logger.warning("Processing queue is full, dropping data point")
            return False
        except Exception as e:
            logger.error(f"Failed to add data point: {e}")
            return False

    def process_batch(self, data_points: list[StreamingDataPoint]) -> ProcessingResult:
        """Process a batch of data points."""
        start_time = time.time()

        try:
            if not data_points:
                return ProcessingResult(
                    success=True,
                    processed_count=0,
                    processing_time=0.0,
                    quality_score=1.0,
                    metadata={},
                )

            # Process data points
            processed_data: ObjectRecords = []
            anomalies: ObjectRecords = []

            for data_point in data_points:
                if self.config.enable_parallel_processing and self.executor:
                    # Parallel processing
                    future = self.executor.submit(
                        self._process_single_point, data_point
                    )
                    result = future.result(timeout=5.0)
                else:
                    # Sequential processing
                    result = self._process_single_point(data_point)

                if result:
                    processed_data.append(result)
                    self.processed_data.append(data_point)
                    self._store_processed_data(data_point.data_type, result)
                    self._trigger_handlers(data_point)

                    # Check for anomalies
                    if self._is_anomaly(result):
                        anomalies.append(
                            {
                                "data_point": data_point.__dict__,
                                "processed_data": result,
                                "timestamp": time.time(),
                            }
                        )

            # Calculate metrics
            processing_time = time.time() - start_time
            throughput = (
                len(data_points) / processing_time if processing_time > 0 else 0
            )
            quality_score = self._calculate_data_quality(data_points)

            result = StreamProcessingResult(
                processed_data=processed_data,
                performance_metrics={
                    "processing_time": processing_time,
                    "throughput": throughput,
                    "queue_size": self.data_queue.qsize(),
                    "processed_count": len(processed_data),
                },
                anomalies_detected=anomalies,
                processing_time=processing_time,
                data_quality_score=quality_score,
                throughput=throughput,
            )

            return ProcessingResult(
                success=True,
                processed_count=len(processed_data),
                processing_time=processing_time,
                quality_score=quality_score,
                metadata={
                    "anomalies_count": len(anomalies),
                    "throughput": throughput,
                    "result": result.__dict__,
                },
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return ProcessingResult(
                success=False,
                processed_count=0,
                processing_time=time.time() - start_time,
                quality_score=0.0,
                metadata={"error": str(e)},
            )

    def get_processing_status(self) -> ObjectMap:
        """Get current processing status."""
        return {
            "is_running": self.is_running,
            "queue_size": self.data_queue.qsize(),
            "max_queue_size": self.config.max_queue_size,
            "processed_count": len(self.processed_data),
            "buffer_size": self.config.buffer_size,
            "processing_interval": self.config.processing_interval,
            "parallel_processing": self.config.enable_parallel_processing,
            "max_workers": self.config.max_workers,
        }

    def get_recent_data(self, limit: int = 100) -> list[StreamingDataPoint]:
        """Get recently processed data points."""
        if not self.processed_data:
            return []
        data = list(self.processed_data)
        return data[-limit:]

    def clear_buffer(self) -> bool:
        """Clear the processing buffer."""
        try:
            self.processed_data.clear()
            self.processed_data_cache.clear()
            # Clear queue
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
            return True
        except Exception as e:
            logger.error(f"Failed to clear buffer: {e}")
            return False

    def process_streaming_data(self, data_queue: Queue[StreamingDataPoint]) -> None:
        """Process streaming data from queue."""
        try:
            while not data_queue.empty():
                data_point = data_queue.get_nowait()
                if not self._validate_data_point(data_point):
                    continue

                processed = self._process_single_point(data_point)
                if processed is None:
                    continue

                self.processed_data.append(data_point)
                self._store_processed_data(data_point.data_type, processed)
                self._trigger_handlers(data_point)
        except Exception as e:
            logger.error(f"Streaming data processing failed: {e}")

    def register_data_handler(
        self,
        data_type: StreamingDataType,
        handler: Callable[[StreamingDataPoint], None],
    ) -> None:
        """Register handler for specific data type."""
        self.data_handlers.setdefault(data_type, []).append(handler)
        logger.info(f"Registered handler for data type: {data_type}")

    def get_processed_data(
        self, data_type: StreamingDataType, lookback_period: int = 100
    ) -> pd.DataFrame:
        """Get processed data for specified type and period."""
        try:
            records = [
                payload
                for payload in self.processed_data_cache.values()
                if str(payload.get("data_type", "")) == data_type.value
            ]

            # Fallback to raw stream points if cache is empty.
            if not records:
                for data_point in self.processed_data:
                    if data_point.data_type != data_type:
                        continue
                    payload = _as_object_map(data_point.data)
                    payload["timestamp"] = data_point.timestamp
                    payload["data_type"] = data_point.data_type.value
                    records.append(payload)

            if not records:
                return pd.DataFrame()

            sorted_records = sorted(
                records,
                key=lambda item: _as_float(
                    item.get("processing_timestamp", item.get("timestamp", 0.0)), 0.0
                ),
            )
            return pd.DataFrame(sorted_records[-lookback_period:])
        except Exception as e:
            logger.error(f"Failed to get processed data for {data_type}: {e}")
            return pd.DataFrame()

    def _trigger_handlers(self, data_point: StreamingDataPoint) -> None:
        """Trigger registered handlers for a data point."""
        handlers = self.data_handlers.get(data_point.data_type, [])
        for handler in handlers:
            try:
                handler(data_point)
            except Exception as e:
                logger.error(f"Handler execution failed: {e}")

    def _store_processed_data(
        self, data_type: StreamingDataType, processed_data: ObjectMap
    ) -> None:
        """Store processed data in an in-memory bounded cache."""
        timestamp = _as_float(
            processed_data.get("processing_timestamp", processed_data.get("timestamp", 0.0)),
            time.time(),
        )
        storage_key = f"{data_type.value}_{timestamp:.6f}_{time.time_ns()}"

        payload = dict(processed_data)
        payload.setdefault("processing_timestamp", timestamp)
        payload["data_type"] = data_type.value
        self.processed_data_cache[storage_key] = payload

        if len(self.processed_data_cache) > self.max_cache_size:
            oldest_keys = sorted(self.processed_data_cache.keys())[
                : len(self.processed_data_cache) - self.max_cache_size
            ]
            for key in oldest_keys:
                del self.processed_data_cache[key]

    def _initialize_components(self) -> None:
        """Initialize processing components."""
        # Initialize anomaly detector
        self.anomaly_detector = self._create_anomaly_detector()

        # Initialize quality monitor
        self.quality_monitor = self._create_quality_monitor()

        # Initialize metrics collector
        self.metrics_collector = self._create_metrics_collector()

    def _processing_loop(self) -> None:
        """Main processing loop for continuous streaming."""
        buffer: list[StreamingDataPoint] = []
        last_process_time = time.time()

        while self.is_running:
            try:
                # Collect data points up to buffer size
                while len(buffer) < self.config.buffer_size and self.is_running:
                    try:
                        data_point = self.data_queue.get(timeout=0.1)
                        buffer.append(data_point)
                    except queue.Empty:
                        break

                # Process buffer if full or time interval reached
                current_time = time.time()
                if (
                    len(buffer) >= self.config.buffer_size
                    or current_time - last_process_time
                    >= self.config.processing_interval
                ):
                    if buffer:
                        result = self.process_batch(buffer)
                        if result.success:
                            logger.debug(
                                f"Processed {result.processed_count} data points"
                            )
                        else:
                            logger.warning(
                                f"Batch processing failed: {result.metadata.get('error', 'Unknown error')}"
                            )

                        buffer.clear()
                        last_process_time = current_time

                # Small sleep to prevent busy waiting
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)  # Wait before retrying

    def _validate_data_point(self, data_point: StreamingDataPoint) -> bool:
        """Validate a data point."""
        if not isinstance(data_point, StreamingDataPoint):
            return False

        # Check required fields
        if not hasattr(data_point, "timestamp") or data_point.timestamp <= 0:
            return False

        if not hasattr(data_point, "data") or not isinstance(data_point.data, dict):
            return False

        # Check data quality
        if len(data_point.data) == 0:
            return False

        return True

    def _process_single_point(
        self, data_point: StreamingDataPoint
    ) -> ObjectMap | None:
        """Process a single data point."""
        try:
            # Extract features
            features = self._extract_features(data_point)

            # Apply transformations
            transformed_features = self._apply_transformations(features)

            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(transformed_features)

            # Combine results
            processed_data: ObjectMap = {
                "original_data": data_point.data,
                "features": features,
                "transformed_features": transformed_features,
                "derived_metrics": derived_metrics,
                "processing_timestamp": time.time(),
                "data_quality": self._assess_data_quality(data_point),
            }

            return processed_data

        except Exception as e:
            logger.error(f"Failed to process data point: {e}")
            return None

    def _extract_features(self, data_point: StreamingDataPoint) -> ObjectMap:
        """Extract features from data point."""
        features: ObjectMap = {}

        # Basic feature extraction
        data = _as_object_map(data_point.data)

        # Price-based features
        if "price" in data:
            features["price"] = data["price"]
            features["price_change"] = data.get("price_change", 0)
            features["price_volatility"] = data.get("price_volatility", 0)

        # Volume-based features
        if "volume" in data:
            features["volume"] = data["volume"]
            features["volume_change"] = data.get("volume_change", 0)

        # Technical indicators
        if "indicators" in data:
            indicators = _as_object_map(data["indicators"])
            features.update(
                {
                    "rsi": _as_float(indicators.get("rsi", 50.0), 50.0),
                    "macd": _as_float(indicators.get("macd", 0.0), 0.0),
                    "bollinger_upper": _as_float(
                        indicators.get("bollinger_upper", 0.0), 0.0
                    ),
                    "bollinger_lower": _as_float(
                        indicators.get("bollinger_lower", 0.0), 0.0
                    ),
                    "sma_20": _as_float(indicators.get("sma_20", 0.0), 0.0),
                    "ema_12": _as_float(indicators.get("ema_12", 0.0), 0.0),
                }
            )

        # Market data
        if "market_data" in data:
            market = _as_object_map(data["market_data"])
            features.update(
                {
                    "market_volatility": _as_float(market.get("volatility", 0.0), 0.0),
                    "market_trend": _as_float(market.get("trend", 0.0), 0.0),
                    "market_volume": _as_float(market.get("volume", 0.0), 0.0),
                }
            )

        return features

    def _apply_transformations(self, features: ObjectMap) -> ObjectMap:
        """Apply data transformations."""
        transformed: ObjectMap = dict(features)

        # Normalization (simple z-score)
        numeric_features = [
            "price",
            "volume",
            "rsi",
            "macd",
            "price_change",
            "volume_change",
        ]
        for feature in numeric_features:
            value = transformed.get(feature)
            if value is not None:
                # In practice, you'd use rolling statistics
                transformed[f"{feature}_normalized"] = _as_float(value, 0.0)

        # Log transformations for skewed data
        volume_value = _as_float(transformed.get("volume"), 0.0)
        if volume_value > 0:
            transformed["volume_log"] = float(np.log(volume_value))

        # Difference transformations
        if "price" in transformed and "sma_20" in transformed:
            transformed["price_sma_diff"] = _as_float(
                transformed.get("price"), 0.0
            ) - _as_float(transformed.get("sma_20"), 0.0)

        return transformed

    def _calculate_derived_metrics(self, features: ObjectMap) -> ObjectMap:
        """Calculate derived metrics."""
        metrics: ObjectMap = {}

        # Momentum indicators
        if "price_change" in features:
            metrics["momentum"] = features["price_change"]

        # Volatility measures
        if "price_volatility" in features:
            metrics["volatility_ratio"] = features["price_volatility"]

        # Volume indicators
        if "volume" in features and "volume_change" in features:
            metrics["volume_momentum"] = features["volume_change"]

        # Composite indicators
        if "rsi" in features and "macd" in features:
            rsi_value = _as_float(features.get("rsi"), 50.0)
            macd_value = _as_float(features.get("macd"), 0.0)
            metrics["composite_signal"] = (rsi_value - 50.0) * 0.01 + macd_value * 0.1

        # Market regime indicators
        if "market_volatility" in features and "market_trend" in features:
            volatility = _as_float(features.get("market_volatility"), 0.0)
            trend = _as_float(features.get("market_trend"), 0.0)
            metrics["regime_strength"] = abs(trend) / (
                volatility + 0.001
            )

        return metrics

    def _assess_data_quality(self, data_point: StreamingDataPoint) -> float:
        """Assess data quality score."""
        score = 1.0
        data = _as_object_map(data_point.data)

        # Check for missing values
        total_fields = len(data)
        missing_fields = sum(
            1
            for v in data.values()
            if v is None or (isinstance(v, float) and np.isnan(v))
        )
        if total_fields > 0:
            score *= 1 - missing_fields / total_fields

        # Check for outliers (simple range check)
        price_value = _as_float(data.get("price"), 0.0)
        if price_value > 0 and not (0.1 < price_value < 1000000):  # Reasonable price range
                score *= 0.5

        # Check timestamp freshness
        if hasattr(data_point, "timestamp"):
            age = time.time() - data_point.timestamp
            if age > 300:  # 5 minutes old
                score *= max(0.1, 1 - age / 3600)  # Degrade over time

        return max(0.0, min(1.0, score))

    def _is_anomaly(self, processed_data: ObjectMap) -> bool:
        """Check if processed data indicates an anomaly."""
        if not self.anomaly_detector:
            return False

        try:
            # Simple anomaly detection based on thresholds
            metrics = _as_object_map(processed_data.get("derived_metrics", {}))

            # Check for extreme values
            if "momentum" in metrics and abs(_as_float(metrics["momentum"], 0.0)) > 0.05:
                return True

            if (
                "volatility_ratio" in metrics
                and _as_float(metrics["volatility_ratio"], 0.0) > 0.1
            ):  # High volatility
                return True

            return False

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False

    def _calculate_data_quality(self, data_points: list[StreamingDataPoint]) -> float:
        """Calculate overall data quality score."""
        if not data_points:
            return 1.0

        quality_scores = [self._assess_data_quality(dp) for dp in data_points]
        return float(np.mean(quality_scores))

    def _create_anomaly_detector(self) -> ObjectMap:
        """Create anomaly detection component."""
        # Placeholder for anomaly detector implementation
        return {}

    def _create_quality_monitor(self) -> ObjectMap:
        """Create data quality monitoring component."""
        # Placeholder for quality monitor implementation
        return {}

    def _create_metrics_collector(self) -> ObjectMap:
        """Create metrics collection component."""
        # Placeholder for metrics collector implementation
        return {}


class AdvancedStreamingProcessor(BaseStreamingProcessor):
    """Advanced streaming processor with enhanced features."""

    def __init__(self, config: StreamingProcessorConfig):
        super().__init__(config)
        self.feature_buffer: deque[ObjectMap] = deque(maxlen=1000)
        self.pattern_detector: ObjectMap | None = None
        self.predictive_model: ObjectMap | None = None

    def _initialize_components(self) -> None:
        """Initialize advanced processing components."""
        super()._initialize_components()

        # Initialize pattern detector
        self.pattern_detector = self._create_pattern_detector()

        # Initialize predictive model
        self.predictive_model = self._create_predictive_model()

    def _process_single_point(
        self, data_point: StreamingDataPoint
    ) -> ObjectMap | None:
        """Enhanced single point processing."""
        # Get base processing
        result = super()._process_single_point(data_point)

        if result:
            # Add pattern detection
            patterns = self._detect_patterns(result)
            result["detected_patterns"] = patterns

            # Add predictions
            predictions = self._generate_predictions(result)
            result["predictions"] = predictions

            # Update feature buffer
            self.feature_buffer.append(result)

        return result

    def _detect_patterns(self, processed_data: ObjectMap) -> ObjectRecords:
        """Detect patterns in the processed data."""
        patterns: ObjectRecords = []

        try:
            features = _as_object_map(processed_data.get("features", {}))

            # Trend pattern detection
            if "price_change" in features:
                price_change = _as_float(features["price_change"], 0.0)
                if price_change > 0.02:  # 2% increase
                    patterns.append(
                        {
                            "type": "uptrend",
                            "strength": price_change,
                            "confidence": 0.8,
                        }
                    )
                elif price_change < -0.02:  # 2% decrease
                    patterns.append(
                        {
                            "type": "downtrend",
                            "strength": abs(price_change),
                            "confidence": 0.8,
                        }
                    )

            # Volatility pattern
            if (
                "price_volatility" in features
                and _as_float(features["price_volatility"], 0.0) > 0.05
            ):
                patterns.append(
                    {
                        "type": "high_volatility",
                        "strength": _as_float(features["price_volatility"], 0.0),
                        "confidence": 0.9,
                    }
                )

            # Volume pattern
            if "volume_change" in features and abs(
                _as_float(features["volume_change"], 0.0)
            ) > 0.5:
                patterns.append(
                    {
                        "type": "volume_spike",
                        "strength": _as_float(features["volume_change"], 0.0),
                        "confidence": 0.7,
                    }
                )

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")

        return patterns

    def _generate_predictions(self, processed_data: ObjectMap) -> ObjectMap:
        """Generate predictions based on processed data."""
        predictions: ObjectMap = {}

        try:
            # Simple momentum-based predictions
            metrics = _as_object_map(processed_data.get("derived_metrics", {}))

            if "momentum" in metrics:
                # Predict next momentum direction
                predictions["next_momentum"] = _as_float(metrics["momentum"], 0.0) * 0.8
                predictions["momentum_confidence"] = 0.6

            if "composite_signal" in metrics:
                # Predict signal strength
                predictions["signal_strength"] = _as_float(
                    metrics["composite_signal"], 0.0
                )
                predictions["signal_confidence"] = 0.7

        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")

        return predictions

    def _create_pattern_detector(self) -> ObjectMap:
        """Create advanced pattern detection component."""
        # Placeholder for advanced pattern detector
        return {}

    def _create_predictive_model(self) -> ObjectMap:
        """Create predictive modeling component."""
        # Placeholder for predictive model
        return {}

    def get_processed_data(
        self, data_type: StreamingDataType, lookback_period: int = 100
    ) -> pd.DataFrame:
        """Get processed data for specified type and period with advanced features."""
        try:
            # Get base data
            data = super().get_processed_data(data_type, lookback_period)

            if data.empty:
                return data

            # Add advanced features
            enhanced_data = self._enhance_processed_data(data, data_type)

            return enhanced_data

        except Exception as e:
            logger.error(f"Failed to get processed data for {data_type}: {e}")
            return pd.DataFrame()

    def _enhance_processed_data(
        self, data: pd.DataFrame, data_type: StreamingDataType
    ) -> pd.DataFrame:
        """Enhance processed data with advanced features."""
        try:
            enhanced = data.copy()
            _ = data_type  # Reserved for future type-specific feature branches.

            composite_signal = (
                enhanced["composite_signal"]
                if "composite_signal" in enhanced.columns
                and is_numeric_dtype(enhanced["composite_signal"])
                else pd.Series(0.0, index=enhanced.index)
            )
            price_volatility = (
                enhanced["price_volatility"]
                if "price_volatility" in enhanced.columns
                and is_numeric_dtype(enhanced["price_volatility"])
                else pd.Series(0.0, index=enhanced.index)
            )
            price_change = (
                enhanced["price_change"]
                if "price_change" in enhanced.columns
                and is_numeric_dtype(enhanced["price_change"])
                else pd.Series(0.0, index=enhanced.index)
            )

            # Add pattern-based features
            if len(enhanced) > 10:
                # Rolling pattern detection
                enhanced["pattern_strength"] = composite_signal.rolling(10).mean()
                enhanced["volatility_regime"] = price_volatility.rolling(20).std()

            # Add predictive features
            if len(enhanced) > 5:
                # Simple trend prediction
                enhanced["predicted_trend"] = price_change.shift(-1).fillna(0.0)

            return enhanced

        except Exception as e:
            logger.warning(f"Data enhancement failed: {e}")
            return data


def create_streaming_processor(config: RealTimeAdaptationConfig) -> IStreamingProcessor:
    """Factory function to create streaming processor."""
    if config.streaming_processor.enable_parallel_processing:
        return AdvancedStreamingProcessor(config.streaming_processor)
    else:
        return BaseStreamingProcessor(config.streaming_processor)
