"""
Streaming Processor Implementation for Action Signal Guide.

This module implements real-time data processing and streaming analytics
for adaptive signal generation and performance monitoring.
"""

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

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

logger = logging.getLogger(__name__)


@dataclass
class StreamProcessingResult:
    """Result of streaming data processing."""

    processed_data: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    data_quality_score: float = 1.0
    throughput: float = 0.0


class BaseStreamingProcessor(IStreamingProcessor):
    """Base implementation of streaming processor."""

    def __init__(self, config: StreamingProcessorConfig):
        self.config = config
        self.data_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self.processed_data: List[StreamingDataPoint] = []
        self.anomaly_detector = None
        self.quality_monitor = None
        self.is_running = False
        self.executor = None
        self.processing_thread = None
        self.metrics_collector = None

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

    def process_batch(self, data_points: List[StreamingDataPoint]) -> ProcessingResult:
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
            processed_data = []
            anomalies = []

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

    def get_processing_status(self) -> Dict[str, Any]:
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

    def get_recent_data(self, limit: int = 100) -> List[StreamingDataPoint]:
        """Get recently processed data points."""
        return self.processed_data[-limit:] if self.processed_data else []

    def clear_buffer(self) -> bool:
        """Clear the processing buffer."""
        try:
            self.processed_data.clear()
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

    def _initialize_components(self):
        """Initialize processing components."""
        # Initialize anomaly detector
        self.anomaly_detector = self._create_anomaly_detector()

        # Initialize quality monitor
        self.quality_monitor = self._create_quality_monitor()

        # Initialize metrics collector
        self.metrics_collector = self._create_metrics_collector()

    def _processing_loop(self):
        """Main processing loop for continuous streaming."""
        buffer = []
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
    ) -> Optional[Dict[str, Any]]:
        """Process a single data point."""
        try:
            # Extract features
            features = self._extract_features(data_point)

            # Apply transformations
            transformed_features = self._apply_transformations(features)

            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(transformed_features)

            # Combine results
            processed_data = {
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

    def _extract_features(self, data_point: StreamingDataPoint) -> Dict[str, Any]:
        """Extract features from data point."""
        features = {}

        # Basic feature extraction
        data = data_point.data

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
            indicators = data["indicators"]
            features.update(
                {
                    "rsi": indicators.get("rsi", 50),
                    "macd": indicators.get("macd", 0),
                    "bollinger_upper": indicators.get("bollinger_upper", 0),
                    "bollinger_lower": indicators.get("bollinger_lower", 0),
                    "sma_20": indicators.get("sma_20", 0),
                    "ema_12": indicators.get("ema_12", 0),
                }
            )

        # Market data
        if "market_data" in data:
            market = data["market_data"]
            features.update(
                {
                    "market_volatility": market.get("volatility", 0),
                    "market_trend": market.get("trend", 0),
                    "market_volume": market.get("volume", 0),
                }
            )

        return features

    def _apply_transformations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data transformations."""
        transformed = features.copy()

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
            if feature in transformed and transformed[feature] is not None:
                # In practice, you'd use rolling statistics
                transformed[f"{feature}_normalized"] = transformed[
                    feature
                ]  # Placeholder

        # Log transformations for skewed data
        if "volume" in transformed and transformed["volume"] > 0:
            transformed["volume_log"] = np.log(transformed["volume"])

        # Difference transformations
        if "price" in transformed and "sma_20" in transformed:
            transformed["price_sma_diff"] = transformed["price"] - transformed["sma_20"]

        return transformed

    def _calculate_derived_metrics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived metrics."""
        metrics = {}

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
            metrics["composite_signal"] = (features["rsi"] - 50) * 0.01 + features[
                "macd"
            ] * 0.1

        # Market regime indicators
        if "market_volatility" in features and "market_trend" in features:
            metrics["regime_strength"] = abs(features["market_trend"]) / (
                features["market_volatility"] + 0.001
            )

        return metrics

    def _assess_data_quality(self, data_point: StreamingDataPoint) -> float:
        """Assess data quality score."""
        score = 1.0
        data = data_point.data

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
        if "price" in data and isinstance(data["price"], (int, float)):
            if not (0.1 < data["price"] < 1000000):  # Reasonable price range
                score *= 0.5

        # Check timestamp freshness
        if hasattr(data_point, "timestamp"):
            age = time.time() - data_point.timestamp
            if age > 300:  # 5 minutes old
                score *= max(0.1, 1 - age / 3600)  # Degrade over time

        return max(0.0, min(1.0, score))

    def _is_anomaly(self, processed_data: Dict[str, Any]) -> bool:
        """Check if processed data indicates an anomaly."""
        if not self.anomaly_detector:
            return False

        try:
            # Simple anomaly detection based on thresholds
            metrics = processed_data.get("derived_metrics", {})

            # Check for extreme values
            if "momentum" in metrics and abs(metrics["momentum"]) > 0.05:  # 5% change
                return True

            if (
                "volatility_ratio" in metrics and metrics["volatility_ratio"] > 0.1
            ):  # High volatility
                return True

            return False

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False

    def _calculate_data_quality(self, data_points: List[StreamingDataPoint]) -> float:
        """Calculate overall data quality score."""
        if not data_points:
            return 1.0

        quality_scores = [self._assess_data_quality(dp) for dp in data_points]
        return np.mean(quality_scores)

    def _create_anomaly_detector(self) -> Any:
        """Create anomaly detection component."""
        # Placeholder for anomaly detector implementation
        return {}

    def _create_quality_monitor(self) -> Any:
        """Create data quality monitoring component."""
        # Placeholder for quality monitor implementation
        return {}

    def _create_metrics_collector(self) -> Any:
        """Create metrics collection component."""
        # Placeholder for metrics collector implementation
        return {}


class AdvancedStreamingProcessor(BaseStreamingProcessor):
    """Advanced streaming processor with enhanced features."""

    def __init__(self, config: StreamingProcessorConfig):
        super().__init__(config)
        self.feature_buffer: List[Dict[str, Any]] = []
        self.pattern_detector = None
        self.predictive_model = None

    def _initialize_components(self):
        """Initialize advanced processing components."""
        super()._initialize_components()

        # Initialize pattern detector
        self.pattern_detector = self._create_pattern_detector()

        # Initialize predictive model
        self.predictive_model = self._create_predictive_model()

    def _process_single_point(
        self, data_point: StreamingDataPoint
    ) -> Optional[Dict[str, Any]]:
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
            if len(self.feature_buffer) > 1000:  # Keep last 1000 points
                self.feature_buffer.pop(0)

        return result

    def _detect_patterns(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns in the processed data."""
        patterns = []

        try:
            features = processed_data.get("features", {})

            # Trend pattern detection
            if "price_change" in features:
                if features["price_change"] > 0.02:  # 2% increase
                    patterns.append(
                        {
                            "type": "uptrend",
                            "strength": features["price_change"],
                            "confidence": 0.8,
                        }
                    )
                elif features["price_change"] < -0.02:  # 2% decrease
                    patterns.append(
                        {
                            "type": "downtrend",
                            "strength": abs(features["price_change"]),
                            "confidence": 0.8,
                        }
                    )

            # Volatility pattern
            if "price_volatility" in features and features["price_volatility"] > 0.05:
                patterns.append(
                    {
                        "type": "high_volatility",
                        "strength": features["price_volatility"],
                        "confidence": 0.9,
                    }
                )

            # Volume pattern
            if "volume_change" in features and abs(features["volume_change"]) > 0.5:
                patterns.append(
                    {
                        "type": "volume_spike",
                        "strength": features["volume_change"],
                        "confidence": 0.7,
                    }
                )

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")

        return patterns

    def _generate_predictions(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on processed data."""
        predictions = {}

        try:
            # Simple momentum-based predictions
            metrics = processed_data.get("derived_metrics", {})

            if "momentum" in metrics:
                # Predict next momentum direction
                predictions["next_momentum"] = metrics["momentum"] * 0.8  # Dampened
                predictions["momentum_confidence"] = 0.6

            if "composite_signal" in metrics:
                # Predict signal strength
                predictions["signal_strength"] = metrics["composite_signal"]
                predictions["signal_confidence"] = 0.7

        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")

        return predictions

    def _create_pattern_detector(self) -> Any:
        """Create advanced pattern detection component."""
        # Placeholder for advanced pattern detector
        return {}

    def _create_predictive_model(self) -> Any:
        """Create predictive modeling component."""
        # Placeholder for predictive model
        return {}

    def process_streaming_data(self, data_queue: Queue) -> None:
        """Process streaming data from queue with advanced features."""
        try:
            while not data_queue.empty():
                data_point = data_queue.get_nowait()
                processed = self._process_single_point(data_point)

                if processed:
                    # Store processed data
                    self._store_processed_data(data_point.data_type, processed)

                    # Trigger registered handlers
                    self._trigger_handlers(data_point)

        except Exception as e:
            logger.error(f"Streaming data processing failed: {e}")

    def register_data_handler(
        self,
        data_type: StreamingDataType,
        handler: Callable[[StreamingDataPoint], None],
    ) -> None:
        """Register handler for specific data type."""
        if data_type not in self.data_handlers:
            self.data_handlers[data_type] = []

        self.data_handlers[data_type].append(handler)
        logger.info(f"Registered handler for data type: {data_type}")

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

            # Add pattern-based features
            if len(enhanced) > 10:
                # Rolling pattern detection
                enhanced["pattern_strength"] = (
                    enhanced.get("composite_signal", 0).rolling(10).mean()
                )
                enhanced["volatility_regime"] = (
                    enhanced.get("price_volatility", 0).rolling(20).std()
                )

            # Add predictive features
            if len(enhanced) > 5:
                # Simple trend prediction
                enhanced["predicted_trend"] = (
                    enhanced.get("price_change", 0).shift(-1).fillna(0)
                )

            return enhanced

        except Exception as e:
            logger.warning(f"Data enhancement failed: {e}")
            return data

    def _trigger_handlers(self, data_point: StreamingDataPoint) -> None:
        """Trigger registered handlers for data point."""
        handlers = self.data_handlers.get(data_point.data_type, [])

        for handler in handlers:
            try:
                handler(data_point)
            except Exception as e:
                logger.error(f"Handler execution failed: {e}")

    def _store_processed_data(
        self, data_type: StreamingDataType, processed_data: Dict[str, Any]
    ) -> None:
        """Store processed data with advanced indexing."""
        try:
            # Enhanced storage with metadata
            storage_key = f"{data_type.value}_{processed_data.get('timestamp', 0)}"
            self.processed_data_cache[storage_key] = processed_data

            # Maintain cache size
            if len(self.processed_data_cache) > self.config.max_cache_size:
                # Remove oldest entries
                oldest_keys = sorted(self.processed_data_cache.keys())[:100]
                for key in oldest_keys:
                    del self.processed_data_cache[key]

        except Exception as e:
            logger.error(f"Data storage failed: {e}")


def create_streaming_processor(config: RealTimeAdaptationConfig) -> IStreamingProcessor:
    """Factory function to create streaming processor."""
    if config.streaming_processor.enable_parallel_processing:
        return AdvancedStreamingProcessor(config.streaming_processor)
    else:
        return BaseStreamingProcessor(config.streaming_processor)
