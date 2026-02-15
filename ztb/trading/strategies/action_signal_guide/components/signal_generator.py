"""
SignalGenerator Component.

This component is responsible for generating trading signals from observations.
Follows Single Responsibility Principle by focusing only on signal generation.
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

import pandas as pd

from ztb.utils.errors import ValidationError
from ztb.utils.logging_utils import get_logger

from .interfaces import ISignalGenerator

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal, ActionSignalGuideConfig, GuidanceLevel
    from ..pattern_recognition.base import SignalResult
    from ..pattern_recognition.multi_timeframe_analyzer import MultiTimeframeAnalyzer
    from ..types import SignalList
    from .signal_quality_filter import SignalQualityFilter, SignalQualityEvaluator

from ..pattern_recognition.base import PatternRecognizer
from .interfaces import IPatternStatistics, IPerformanceTracker
from .market_regime import (
    MarketConditionAnalyzer,
    MarketRegimeDetector,
    RegimeAdaptiveSignalProcessor,
)
from .performance_tracker import PerformanceTracker

# Import adaptive components
from .sac_integration import (
    SACDecisionIntegrator,
    SACPerformanceMonitor,
    SACSignalValidator,
)
from .validation import DataSanitizer, SignalValidator


def _get_action_signal_class() -> Type["ActionSignal"]:
    """Lazy import to avoid circular imports."""
    from ..action_signal_guide import ActionSignal

    return ActionSignal


def _get_guidance_level_enum() -> type["GuidanceLevel"]:
    """Lazy import to avoid circular imports."""
    from ..action_signal_guide import GuidanceLevel

    return GuidanceLevel


class SignalGenerator(ISignalGenerator):
    """
    Generates trading signals from market observations.

    This class encapsulates signal generation logic including:
    - Pattern recognizer initialization
    - Signal aggregation from multiple patterns
    - Guidance level filtering
    """

    def __init__(
        self,
        config: "ActionSignalGuideConfig",
        performance_tracker: Optional["IPerformanceTracker"] = None,
        pattern_statistics: Optional["IPatternStatistics"] = None,
    ) -> None:
        """
        Initialize SignalGenerator.

        Args:
            config: ActionSignalGuide configuration
            performance_tracker: Optional performance tracker
            pattern_statistics: Optional pattern statistics tracker
        """
        self.config = config
        self.guidance_level = config.guidance_level
        self.logger = get_logger("ztb.trading.strategies.signal_generator")

        self.all_recognizers: List[PatternRecognizer] = []
        self.performance_tracker = performance_tracker
        self.pattern_statistics = pattern_statistics
        # Internal error tracking to avoid log spam for repeated identical errors
        self._error_counts: Dict[str, int] = {}

        # Adaptive algorithm components
        self.adaptive_weights: Dict[str, float] = {}
        self.pattern_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.market_regime_detector: Optional["MarketRegimeDetector"] = None
        self.regime_processor: Optional["RegimeAdaptiveSignalProcessor"] = None
        self.market_regime_adapter: Optional["RegimeAdaptiveSignalProcessor"] = None
        self.market_analyzer: Optional["MarketConditionAnalyzer"] = None
        self.sac_validator: Optional["SACSignalValidator"] = None
        self.sac_integrator: Optional["SACDecisionIntegrator"] = None
        self.sac_monitor: Optional["SACPerformanceMonitor"] = None
        self.signal_validator: Optional["SignalValidator"] = None
        self.data_sanitizer: Optional["DataSanitizer"] = None
        self.performance_tracker: Optional["IPerformanceTracker"] = performance_tracker
        self.multi_timeframe_analyzer: Optional["MultiTimeframeAnalyzer"] = None
        self.signal_quality_filter: Optional["SignalQualityFilter"] = None
        self.signal_quality_evaluator: Optional["SignalQualityEvaluator"] = None

        # Runtime context for adaptive filtering
        self._current_data = pd.DataFrame()
        self._current_multi_timeframe_data: Optional[dict[str, object]] = None
        self._current_market_regime: object | None = None
        self._current_sac_decision: object | None = None

        # Initialize recognizers first
        self.initialize_recognizers()

        # Initialize adaptive components
        self._initialize_adaptive_components()

        # Parallel processing settings (after recognizers are initialized)
        self.enable_parallel = getattr(config, "enable_parallel_processing", False)
        self.max_workers = (
            min(4, len(self.all_recognizers) // 2) if self.enable_parallel else 1
        )

    def _initialize_adaptive_components(self) -> None:
        """Initialize adaptive algorithm components."""
        # Initialize market regime detection and processing
        if getattr(self.config, "enable_adaptive_algorithms", True):
            self.market_regime_detector = MarketRegimeDetector()
            self.regime_processor = RegimeAdaptiveSignalProcessor()
            self.market_regime_adapter = self.regime_processor

        # Initialize market condition analysis
        if getattr(self.config, "enable_market_analysis", True):
            self.market_analyzer = MarketConditionAnalyzer()

        # Initialize validation and utility components
        if getattr(self.config, "enable_signal_validation", True):
            self.signal_validator = SignalValidator()

        if getattr(self.config, "enable_data_sanitization", True):
            self.data_sanitizer = DataSanitizer()

        if getattr(self.config, "enable_performance_tracking", True):
            if self.performance_tracker is None:  # Only create if not provided
                self.performance_tracker = PerformanceTracker()

        # Initialize weights from currently active recognizers.
        self._initialize_adaptive_weights()

        self.logger.info("Adaptive components initialized successfully")

    def initialize_recognizers(self) -> None:
        """Initialize all pattern recognition systems."""
        start_time = time.time()

        try:
            # Import all recognizer classes
            from ..pattern_recognition.adx_patterns import ADXRecognizer
            from ..pattern_recognition.bollinger_patterns import (
                BollingerBandsRecognizer,
            )
            from ..pattern_recognition.candlestick_patterns import (
                BearishEngulfingRecognizer,
                BullishEngulfingRecognizer,
                EveningStarRecognizer,
                HammerRecognizer,
                HangingManRecognizer,
                MorningStarRecognizer,
                PiercingPatternRecognizer,
                RisingThreeMethodsRecognizer,
                SakataFiveMethodsRecognizer,
                ThreeBlackCrowsRecognizer,
                ThreeWhiteSoldiersRecognizer,
            )
            from ..pattern_recognition.dow_theory import DowTheoryRecognizer
            from ..pattern_recognition.fibonacci_patterns import (
                FibonacciExtensionRecognizer,
                FibonacciProjectionRecognizer,
                FibonacciRetracementRecognizer,
            )
            from ..pattern_recognition.gann_analysis import (
                GannAngleRecognizer,
                GannSquareRecognizer,
                GannTimeClusterRecognizer,
            )
            from ..pattern_recognition.granville_law import GranvilleLawRecognizer
            from ..pattern_recognition.harmonic_patterns import (
                BatRecognizer,
                ButterflyRecognizer,
                CrabRecognizer,
                GartleyRecognizer,
            )
            from ..pattern_recognition.heikin_ashi import HeikinAshiRecognizer
            from ..pattern_recognition.macd import MACDPatternRecognizer
            from ..pattern_recognition.oscillator_patterns import (
                CCIRecognizer,
                MFIRecognizer,
                StochasticRecognizer,
                WilliamsRRecognizer,
            )
            from ..pattern_recognition.rsi import RSIPatternRecognizer
            from ..pattern_recognition.volume_patterns import ChaikinADRecognizer
            from ..pattern_recognition.wave_counting import (
                CorrectiveWaveRecognizer,
                ImpulseWaveRecognizer,
                WaveExtensionRecognizer,
                WaveIRecognizer,
                WaveNRecognizer,
                WavePRecognizer,
                WaveSRecognizer,
                WaveVRecognizer,
                WaveYRecognizer,
            )

            # Initialize all recognizers based on config flags
            self.all_recognizers = []

            # Candlestick patterns
            if getattr(self.config, "enable_candlestick_patterns", True):
                self.all_recognizers.extend(
                    [
                        BearishEngulfingRecognizer(),
                        BullishEngulfingRecognizer(),
                        EveningStarRecognizer(),
                        HammerRecognizer(),
                        HangingManRecognizer(),
                        MorningStarRecognizer(),
                        PiercingPatternRecognizer(),
                        RisingThreeMethodsRecognizer(),
                        SakataFiveMethodsRecognizer(),
                        ThreeBlackCrowsRecognizer(),
                        ThreeWhiteSoldiersRecognizer(),
                    ]
                )

            # Fibonacci patterns
            if getattr(self.config, "enable_fibonacci_patterns", True):
                self.all_recognizers.extend(
                    [
                        FibonacciExtensionRecognizer(),
                        FibonacciProjectionRecognizer(),
                        FibonacciRetracementRecognizer(),
                    ]
                )

            # Gann analysis
            if getattr(self.config, "enable_gann_patterns", True):
                self.all_recognizers.extend(
                    [
                        GannAngleRecognizer(),
                        GannSquareRecognizer(),
                        GannTimeClusterRecognizer(),
                    ]
                )

            # Granville law
            if getattr(self.config, "enable_granville_patterns", True):
                self.all_recognizers.append(GranvilleLawRecognizer())

            # Harmonic patterns
            if getattr(self.config, "enable_harmonic_patterns", True):
                self.all_recognizers.extend(
                    [
                        BatRecognizer(),
                        ButterflyRecognizer(),
                        CrabRecognizer(),
                        GartleyRecognizer(),
                    ]
                )

            # Wave counting
            if getattr(self.config, "enable_wave_patterns", True):
                self.all_recognizers.extend(
                    [
                        ImpulseWaveRecognizer(),
                        CorrectiveWaveRecognizer(),
                        WaveExtensionRecognizer(),
                        WaveIRecognizer(),
                        WaveVRecognizer(),
                        WaveYRecognizer(),
                        WavePRecognizer(),
                        WaveNRecognizer(),
                        WaveSRecognizer(),
                    ]
                )

            # Oscillator patterns
            if getattr(self.config, "enable_oscillator_patterns", True):
                self.all_recognizers.extend(
                    [
                        CCIRecognizer(),
                        StochasticRecognizer(),
                        WilliamsRRecognizer(),
                        MFIRecognizer(),
                        RSIPatternRecognizer(),
                        MACDPatternRecognizer(),
                    ]
                )

            # Volume patterns
            if getattr(self.config, "enable_volume_patterns", True):
                self.all_recognizers.append(ChaikinADRecognizer())

            # Bollinger patterns
            if getattr(self.config, "enable_bollinger_patterns", True):
                self.all_recognizers.append(BollingerBandsRecognizer())

            # ADX patterns
            if getattr(self.config, "enable_adx_patterns", True):
                self.all_recognizers.append(ADXRecognizer())

            # Heikin-Ashi patterns
            if getattr(self.config, "enable_heikin_ashi_patterns", True):
                self.all_recognizers.append(HeikinAshiRecognizer())

            # Dow Theory
            if getattr(self.config, "enable_dow_theory_patterns", True):
                self.all_recognizers.append(DowTheoryRecognizer())

            initialization_time = time.time() - start_time
            self.logger.info(
                f"Initialized {len(self.all_recognizers)} pattern recognizers in {initialization_time:.3f}s"
            )

            if self.performance_tracker:
                self.performance_tracker.record_signal_generation(initialization_time)

        except Exception as e:
            self.logger.error(f"Failed to initialize recognizers: {e}")
            self.all_recognizers = []

    def initialize_adaptive_components(
        self,
        enable_adaptive_weights: bool = True,
        enable_regime_adaptation: bool = True,
        enable_quality_filtering: bool = True,
        enable_quality_evaluation: bool = True,
        enable_multi_timeframe_analysis: bool = True,
    ) -> None:
        """
        Initialize adaptive algorithm components.

        Args:
            enable_adaptive_weights: Enable adaptive pattern weighting
            enable_regime_adaptation: Enable market regime-based adaptation
            enable_quality_filtering: Enable signal quality filtering
            enable_quality_evaluation: Enable comprehensive signal quality evaluation
            enable_multi_timeframe_analysis: Enable multi-timeframe signal analysis
        """
        if enable_adaptive_weights:
            self._initialize_adaptive_weights()

        if enable_regime_adaptation:
            # Use RegimeAdaptiveSignalProcessor for regime-aware adaptation
            self.market_regime_adapter = RegimeAdaptiveSignalProcessor()
            self.regime_processor = self.market_regime_adapter

        if enable_quality_filtering or enable_quality_evaluation:
            from .signal_quality_filter import SignalQualityEvaluator, SignalQualityFilter

        if enable_quality_filtering:
            self.signal_quality_filter = SignalQualityFilter(
                getattr(self.config, "quality_filter", {})
            )

        if enable_quality_evaluation:
            self.signal_quality_evaluator = SignalQualityEvaluator(
                getattr(self.config, "quality_filter", {})
            )

        if enable_multi_timeframe_analysis and self.multi_timeframe_analyzer is None:
            from ..pattern_recognition.multi_timeframe_analyzer import (
                MultiTimeframeAnalyzer,
            )

            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()

        self.logger.info("Adaptive components initialized successfully")

    def _initialize_adaptive_weights(self) -> None:
        """Initialize adaptive weights for pattern recognizers."""
        # Start with equal weights for all active recognizer pattern types.
        pattern_types = {
            self._pattern_type_from_recognizer(recognizer)
            for recognizer in self.all_recognizers
        }

        if not pattern_types:
            self.adaptive_weights = {}
            return

        initial_weight = 1.0 / len(pattern_types)
        self.adaptive_weights = {
            pattern_type: initial_weight for pattern_type in pattern_types
        }

    def adapt_to_market_conditions(
        self,
        current_data: pd.DataFrame,
        recent_performance: Dict[str, float],
        market_regime: Optional[str] = None,
    ) -> None:
        """
        Adapt signal generation based on market conditions and recent performance.

        Args:
            current_data: Current market data
            recent_performance: Recent performance metrics by pattern
            market_regime: Current market regime
        """
        # Update pattern weights based on performance
        self._update_adaptive_weights(recent_performance)

        # Update runtime context for regime-aware filtering.
        if market_regime:
            self._current_market_regime = market_regime

        # Update signal quality thresholds
        if self.signal_quality_filter:
            self.signal_quality_filter.update_thresholds(
                current_data, recent_performance
            )

    def _update_adaptive_weights(self, recent_performance: Dict[str, float]) -> None:
        """
        Update adaptive weights based on recent pattern performance.

        Args:
            recent_performance: Performance scores by pattern
        """
        if not recent_performance:
            return

        # Exponential moving average for weight updates
        alpha = 0.1  # Learning rate

        for pattern, performance in recent_performance.items():
            if pattern in self.adaptive_weights:
                # Store performance history
                self.pattern_performance_history[pattern].append(performance)

                # Keep only recent history
                if len(self.pattern_performance_history[pattern]) > 100:
                    self.pattern_performance_history[pattern].pop(0)

                # Update weight using exponential moving average
                current_weight = self.adaptive_weights[pattern]
                avg_performance = sum(self.pattern_performance_history[pattern]) / len(
                    self.pattern_performance_history[pattern]
                )

                # Higher performance -> higher weight
                target_weight = max(0.1, min(2.0, avg_performance))
                new_weight = current_weight * (1 - alpha) + target_weight * alpha

                self.adaptive_weights[pattern] = new_weight

        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            self.adaptive_weights = {
                k: v / total_weight for k, v in self.adaptive_weights.items()
            }

    def apply_adaptive_filtering(
        self,
        signals: List["ActionSignal"],
        current_data: pd.DataFrame,
        market_context: Optional[dict[str, object]] = None,
        multi_timeframe_data: Optional[dict[str, object]] = None,
    ) -> List["ActionSignal"]:
        """
        Apply adaptive filtering to generated signals.

        Args:
            signals: Raw generated signals
            current_data: Current market data
            market_context: Additional market context
            multi_timeframe_data: Multi-timeframe market data

        Returns:
            Filtered signals
        """
        if not signals:
            return []

        filtered_signals = signals.copy()
        context = market_context or {}

        # Apply multi-timeframe alignment first.
        timeframe_alignment = self._extract_timeframe_alignment(multi_timeframe_data)
        if timeframe_alignment is not None:
            alignment_factor = max(0.5, min(1.0, timeframe_alignment))
            for signal in filtered_signals:
                signal.confidence = max(
                    0.0, min(1.0, signal.confidence * alignment_factor)
                )
                signal.metadata["timeframe_alignment"] = timeframe_alignment

        # Apply regime-based filtering
        if self.market_regime_adapter and not current_data.empty:
            filtered_signals = self.market_regime_adapter.process_signals_for_regime(
                filtered_signals, current_data
            )

        # Apply quality filtering
        if self.signal_quality_filter:
            filtered_signals = self.signal_quality_filter.filter_by_quality(
                filtered_signals, current_data
            )

        # Apply quality evaluation and ranking
        if self.signal_quality_evaluator:
            filtered_signals = self._apply_quality_evaluation(
                filtered_signals, current_data, context
            )

        # Apply adaptive weighting
        filtered_signals = self._apply_adaptive_weighting(filtered_signals)

        return filtered_signals

    def _apply_adaptive_weighting(
        self, signals: List["ActionSignal"]
    ) -> List["ActionSignal"]:
        """
        Apply adaptive weighting to signals.

        Args:
            signals: Input signals

        Returns:
            Weighted signals
        """
        for signal in signals:
            pattern_name = (
                getattr(signal, "signal_type", None)
                or getattr(signal, "pattern_type", None)
                or "unknown"
            )
            if pattern_name in self.adaptive_weights:
                # Adjust confidence based on adaptive weight
                weight = self.adaptive_weights[pattern_name]
                signal.confidence = max(0.0, min(1.0, signal.confidence * weight))

                # Add metadata about weighting
                if not hasattr(signal, "metadata"):
                    signal.metadata = {}
                signal.metadata["adaptive_weight"] = weight
                signal.metadata["adjusted_confidence"] = signal.confidence

        return signals

    def get_adaptive_statistics(self) -> Dict[str, Union[float, dict, list]]:
        """
        Get statistics about adaptive algorithm performance.

        Returns:
            Adaptive algorithm statistics
        """
        return {
            "adaptive_weights": self.adaptive_weights.copy(),
            "pattern_performance_history": dict(self.pattern_performance_history),
            "total_patterns_tracked": len(self.adaptive_weights),
            "regime_adapter_active": self.market_regime_adapter is not None,
            "quality_filter_active": self.signal_quality_filter is not None,
        }

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_index: int,
        multi_timeframe_data: Optional[dict[str, object]] = None,
    ) -> "ActionSignal":
        """
        Generate trading signal from market data.

        Args:
            data: OHLCV DataFrame
            current_index: Current bar index
            multi_timeframe_data: Optional multi-timeframe data

        Returns:
            Generated action signal
        """
        ActionSignal = _get_action_signal_class()
        start_time = time.time()

        # Early return if insufficient data for pattern recognition
        # Most demanding patterns require at least 25 data points
        min_required_data = 25
        if len(data) < min_required_data:
            self.logger.debug(
                f"Insufficient data for pattern recognition: {len(data)} < {min_required_data} at index {current_index}"
            )
            processing_time = time.time() - start_time
            if self.performance_tracker and hasattr(
                self.performance_tracker, "record_signal_generation"
            ):
                self.performance_tracker.record_signal_generation(processing_time)
            # Return neutral signal on insufficient data
            return ActionSignal.neutral()

        # Store current context for adaptive filtering
        self._current_data = data
        self._current_multi_timeframe_data = multi_timeframe_data

        try:
            if self.market_regime_detector is not None:
                self._current_market_regime = self.market_regime_detector.detect_regime(
                    data
                )

            # Generate signals from all recognizers
            all_signals: list["ActionSignal"] = []
            pattern_signals: dict[str, list["ActionSignal"]] = {}

            # Respect debug/short mode: only run a subset of recognizers when enabled
            recognizers_to_run = self.all_recognizers
            if getattr(self.config, "debug_short_mode", False):
                limit = getattr(self.config, "short_mode_recognizer_limit", 8)
                # Only log once per SignalGenerator instance to avoid spam
                if not hasattr(self, "_short_mode_logged"):
                    self.logger.info(
                        f"Debug short mode enabled: limiting recognizers to first {limit}"
                    )
                    self._short_mode_logged = True
                recognizers_to_run = self.all_recognizers[:limit]

            # Generate signals using parallel or sequential processing
            if self.enable_parallel and len(recognizers_to_run) > 3:
                all_signals, pattern_signals = self._generate_signals_parallel(
                    recognizers_to_run,
                    data,
                    current_index,
                    multi_timeframe_data,
                    ActionSignal,
                )
            else:
                all_signals, pattern_signals = self._generate_signals_sequential(
                    recognizers_to_run,
                    data,
                    current_index,
                    multi_timeframe_data,
                    ActionSignal,
                )

            # Aggregate signals based on guidance level
            final_signal = self._aggregate_signals(all_signals, pattern_signals)

            processing_time = time.time() - start_time

            if self.performance_tracker and hasattr(
                self.performance_tracker, "record_signal_generation"
            ):
                self.performance_tracker.record_signal_generation(processing_time)

            return final_signal

        except Exception as e:
            # Enhanced error handling with classification and recovery
            error_type = type(e).__name__
            error_msg = str(e)

            # Classify error for appropriate handling and logging
            if "insufficient" in error_msg.lower() or "length" in error_msg.lower():
                error_category = "data_insufficient"
                self.logger.debug(
                    f"Data insufficient for signal generation at index {current_index}: {error_msg}"
                )
            elif "memory" in error_msg.lower():
                error_category = "memory_error"
                self.logger.error(f"Memory error in signal generation: {error_msg}")
            elif "timeout" in error_msg.lower():
                error_category = "timeout_error"
                self.logger.warning(f"Signal generation timeout: {error_msg}")
            elif "validation" in error_msg.lower():
                error_category = "validation_error"
                self.logger.warning(f"Signal validation error: {error_msg}")
            elif "parallel" in error_msg.lower():
                error_category = "parallel_processing_error"
                self.logger.warning(f"Parallel processing error: {error_msg}")
            else:
                error_category = "signal_generation_error"
                self.logger.error(
                    f"Signal generation failed ({error_type}): {error_msg}"
                )

            processing_time = time.time() - start_time

            if self.performance_tracker and hasattr(
                self.performance_tracker, "record_signal_generation"
            ):
                self.performance_tracker.record_signal_generation(processing_time)

            # Record error for monitoring
            self._error_counts[error_category] = (
                self._error_counts.get(error_category, 0) + 1
            )

            # Return neutral signal on failure (graceful degradation)
            return ActionSignal.neutral()

    def _record_error(self, message: str, error_category: str = "general") -> None:
        """Record and selectively log errors to prevent log spam from repeated identical messages.

        Enhanced version that considers error categories for better monitoring.

        Args:
            message: Error message to record
            error_category: Category of the error for classification
        """
        try:
            threshold = getattr(self.config, "error_suppression_threshold", 3)
        except Exception:
            threshold = 3

        # Use category-specific counting
        error_key = f"{error_category}:{message}"
        count = self._error_counts.get(error_key, 0) + 1
        self._error_counts[error_key] = count

        # Log based on error category and count
        if count <= threshold:
            if error_category in ["memory_error", "signal_generation_error"]:
                self.logger.error(f"[{error_category}] {message}")
            elif error_category in ["timeout_error", "parallel_processing_error"]:
                self.logger.warning(f"[{error_category}] {message}")
            else:
                self.logger.info(f"[{error_category}] {message}")

            if count == threshold:
                # Warn user that future identical messages will be suppressed
                self.logger.info(
                    f"Further identical errors for category '{error_category}' will be suppressed: '{message}'"
                )
        else:
            # Suppress additional identical warnings to reduce spam
            # Do nothing here; counts are still tracked for diagnostics
            pass
            # Do nothing here; counts are still tracked for diagnostics
            pass

    def _aggregate_signals(
        self,
        all_signals: "SignalList",
        pattern_signals: dict[str, list["ActionSignal"]],
    ) -> "ActionSignal":
        """
        Aggregate signals from multiple patterns based on guidance level.

        Args:
            all_signals: All detected signals
            pattern_signals: Signals grouped by pattern type

        Returns:
            Aggregated final signal
        """
        ActionSignal = _get_action_signal_class()

        if not all_signals:
            return ActionSignal.neutral()

        # Filter signals based on guidance level
        filtered_signals = self._filter_by_guidance_level(all_signals)

        if not filtered_signals:
            return ActionSignal.neutral()

        # Apply adaptive filtering if enabled
        if (
            self.market_regime_adapter is not None
            or self.signal_quality_filter is not None
            or self.signal_quality_evaluator is not None
            or self.multi_timeframe_analyzer is not None
        ):
            # Create market context from available data
            market_context: dict[str, object] = {}
            if self._current_market_regime is not None:
                market_context["regime"] = self._current_market_regime
            if self._current_sac_decision is not None:
                market_context["sac_decision"] = self._current_sac_decision

            # Apply adaptive filtering
            current_data = self._current_data
            filtered_signals = self.apply_adaptive_filtering(
                filtered_signals,
                current_data,
                market_context,
                self._current_multi_timeframe_data,
            )

        if not filtered_signals:
            return ActionSignal.neutral()

        # Calculate weighted direction (continuous value from -1.0 to 1.0)
        total_weight = sum(s.strength * s.confidence for s in filtered_signals)
        if total_weight == 0:
            return ActionSignal.neutral()

        # Weighted average of directions
        weighted_direction = (
            sum(s.direction * s.strength * s.confidence for s in filtered_signals)
            / total_weight
        )

        # Clamp to [-1.0, 1.0] range
        direction = max(-1.0, min(1.0, weighted_direction))

        # Calculate overall strength as the magnitude of the weighted direction
        strength = abs(direction)

        # Calculate average confidence
        confidence = sum(s.confidence for s in filtered_signals) / len(filtered_signals)

        # Create metadata
        # Count signals by direction range
        buy_signals = [
            s for s in filtered_signals if s.direction > 0.1
        ]  # Positive direction
        sell_signals = [
            s for s in filtered_signals if s.direction < -0.1
        ]  # Negative direction
        hold_signals = [
            s for s in filtered_signals if abs(s.direction) <= 0.1
        ]  # Near neutral

        metadata = {
            "total_signals": len(filtered_signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "hold_signals": len(hold_signals),
            "pattern_types": list(pattern_signals.keys()),
            "guidance_level": self.guidance_level.value,
            "confidence": confidence,
            "weighted_direction": weighted_direction,
        }

        return ActionSignal(
            timestamp=pd.Timestamp.now(),
            direction=direction,
            strength=strength,
            confidence=confidence,
            signal_type="aggregated",
            description=f"Multi-pattern aggregate signal ({len(filtered_signals)} patterns)",
            metadata=metadata,
            source_patterns=list(pattern_signals.keys()),
        )

    def _filter_by_guidance_level(self, signals: "SignalList") -> "SignalList":
        """
        Filter signals based on guidance level.

        Args:
            signals: All detected signals

        Returns:
            Filtered signals based on guidance level
        """
        GuidanceLevel = _get_guidance_level_enum()

        if self.guidance_level == GuidanceLevel.STRONG:
            # Only very strong signals
            return [s for s in signals if s.strength >= 0.8 and s.confidence >= 0.8]
        elif self.guidance_level == GuidanceLevel.MODERATE:
            # Moderate to strong signals
            return [s for s in signals if s.strength >= 0.6 and s.confidence >= 0.6]
        elif self.guidance_level == GuidanceLevel.WEAK:
            # Lenient mode for testing and exploratory runs: accept weaker signals.
            # Lower thresholds help surface recognizers that produce low-confidence
            # or synthetic signals during debugging and integration testing.
            return [s for s in signals if s.strength >= 0.05 and s.confidence >= 0.05]
        else:
            # Default to moderate
            return [s for s in signals if s.strength >= 0.6 and s.confidence >= 0.6]

    def _generate_signals_parallel(
        self,
        recognizers: list[PatternRecognizer],
        data: pd.DataFrame,
        current_index: int,
        multi_timeframe_data: Optional[dict[str, object]],
        action_signal_cls: type["ActionSignal"],
    ) -> tuple[list["ActionSignal"], dict[str, list["ActionSignal"]]]:
        """Generate signals using parallel processing with ThreadPoolExecutor."""
        all_signals: list["ActionSignal"] = []
        pattern_signals: dict[str, list["ActionSignal"]] = defaultdict(list)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._process_recognizer,
                    recognizer,
                    data,
                    current_index,
                    multi_timeframe_data,
                    action_signal_cls,
                )
                for recognizer in recognizers
            ]

            for future in futures:
                action_signal, pattern_type = future.result()
                self._record_generated_signal(
                    all_signals, pattern_signals, action_signal, pattern_type
                )

        return all_signals, dict(pattern_signals)

    def _generate_signals_sequential(
        self,
        recognizers: list[PatternRecognizer],
        data: pd.DataFrame,
        current_index: int,
        multi_timeframe_data: Optional[dict[str, object]],
        action_signal_cls: type["ActionSignal"],
    ) -> tuple[list["ActionSignal"], dict[str, list["ActionSignal"]]]:
        """Generate signals using sequential processing (fallback method)."""
        all_signals: list["ActionSignal"] = []
        pattern_signals: dict[str, list["ActionSignal"]] = defaultdict(list)

        for recognizer in recognizers:
            action_signal, pattern_type = self._process_recognizer(
                recognizer,
                data,
                current_index,
                multi_timeframe_data,
                action_signal_cls,
            )
            self._record_generated_signal(
                all_signals, pattern_signals, action_signal, pattern_type
            )

        return all_signals, dict(pattern_signals)

    def _process_recognizer(
        self,
        recognizer: PatternRecognizer,
        data: pd.DataFrame,
        current_index: int,
        multi_timeframe_data: Optional[dict[str, object]],
        action_signal_cls: type["ActionSignal"],
    ) -> tuple[Optional["ActionSignal"], Optional[str]]:
        """Run a single recognizer and map its result to ActionSignal."""
        try:
            lookback_period = getattr(recognizer, "get_lookback_period", lambda: 20)()
            if current_index < lookback_period:
                return None, None

            recognizer_name = getattr(recognizer, "name", recognizer.__class__.__name__)
            should_trace = self._should_trace_recognizer(recognizer_name)
            if should_trace:
                self.logger.info(
                    f"Running recognizer: {recognizer_name} at index {current_index}"
                )

            try:
                signal_result = recognizer.recognize(
                    data, current_index, multi_timeframe_data
                )
            except ValidationError:
                return None, None
            except Exception as exc:
                self.logger.warning(
                    f"Unexpected error in recognizer {recognizer_name}: {exc}"
                )
                return None, None

            if should_trace:
                self.logger.info(
                    f"Recognizer {recognizer_name} returned: {signal_result is not None}"
                )

            if signal_result is None:
                return None, None

            return self._build_action_signal(
                recognizer, signal_result, action_signal_cls
            )
        except Exception as exc:
            recognizer_name = getattr(recognizer, "name", recognizer.__class__.__name__)
            self._record_error(f"Recognizer {recognizer_name} failed: {exc}")
            return None, None

    def _build_action_signal(
        self,
        recognizer: PatternRecognizer,
        signal_result: "SignalResult",
        action_signal_cls: type["ActionSignal"],
    ) -> tuple["ActionSignal", str]:
        """Convert recognizer result to ActionSignal and normalized pattern type."""
        pattern_type = self._pattern_type_from_recognizer(recognizer)
        confidence = (
            signal_result.confidence
            if signal_result.confidence is not None
            else signal_result.strength
        )
        metadata = dict(signal_result.metadata)
        metadata.update(
            {
                "confidence": confidence,
                "risk_level": signal_result.risk_level,
                "validity_period": signal_result.validity_period,
            }
        )
        recognizer_name = getattr(recognizer, "name", recognizer.__class__.__name__)

        return (
            action_signal_cls(
                timestamp=pd.Timestamp.now(),
                direction=signal_result.direction,
                strength=signal_result.strength,
                confidence=confidence,
                signal_type=pattern_type,
                description=f"{recognizer_name}: {signal_result.description}",
                metadata=metadata,
                source_patterns=[recognizer_name],
            ),
            pattern_type,
        )

    def _record_generated_signal(
        self,
        all_signals: list["ActionSignal"],
        pattern_signals: dict[str, list["ActionSignal"]],
        action_signal: Optional["ActionSignal"],
        pattern_type: Optional[str],
    ) -> None:
        """Store generated signal and update pattern statistics."""
        if action_signal is None or pattern_type is None:
            return

        all_signals.append(action_signal)
        pattern_signals[pattern_type].append(action_signal)

        if self.pattern_statistics:
            self.pattern_statistics.record_pattern_signal(pattern_type, action_signal)

    @staticmethod
    def _pattern_type_from_recognizer(recognizer: PatternRecognizer) -> str:
        """Normalize recognizer class name to pattern type key."""
        return recognizer.__class__.__name__.lower().replace("recognizer", "")

    @staticmethod
    def _should_trace_recognizer(recognizer_name: str) -> bool:
        recognizer_name = recognizer_name.lower()
        return "harmonic" in recognizer_name or "dow" in recognizer_name

    def _extract_timeframe_alignment(
        self, multi_timeframe_data: Optional[dict[str, object]]
    ) -> Optional[float]:
        """
        Extract alignment score from multi-timeframe payload.

        Supports both:
        - feature payloads containing `timeframe_alignment`
        - nested timeframe payloads compatible with `MultiTimeframeAnalyzer`
        """
        if not multi_timeframe_data:
            return None

        raw_alignment = multi_timeframe_data.get("timeframe_alignment")
        if isinstance(raw_alignment, (int, float)):
            return max(0.0, min(1.0, float(raw_alignment)))

        if self.multi_timeframe_analyzer is None:
            return None

        nested_payload = {
            key: value
            for key, value in multi_timeframe_data.items()
            if isinstance(value, dict) and "data" in value
        }
        if not nested_payload:
            return None

        try:
            analysis = self.multi_timeframe_analyzer.analyze_multi_timeframe_alignment(
                nested_payload
            )
            return self.multi_timeframe_analyzer.calculate_overall_alignment_score(
                analysis
            )
        except Exception as exc:
            self.logger.debug(f"Failed to evaluate multi-timeframe alignment: {exc}")
            return None

    def _apply_quality_evaluation(
        self,
        signals: List["ActionSignal"],
        current_data: pd.DataFrame,
        market_context: dict[str, object],
    ) -> List["ActionSignal"]:
        """
        Apply quality evaluation to signals and filter based on quality scores.

        Args:
            signals: Signals to evaluate
            current_data: Current market data
            market_context: Market context information

        Returns:
            Quality-evaluated and filtered signals
        """
        if not self.signal_quality_evaluator:
            return signals

        evaluated_signals: list["ActionSignal"] = []

        for signal in signals:
            # Evaluate signal quality
            quality_scores = self.signal_quality_evaluator.evaluate_signal_quality(
                signal=signal,
                market_data=current_data,
                sac_decision=market_context.get("sac_decision"),
                market_regime=market_context.get("regime"),
            )

            # Calculate overall quality score
            overall_quality = self.signal_quality_evaluator.get_overall_quality_score(
                quality_scores
            )

            # Add quality information to signal
            signal.metadata["quality_scores"] = quality_scores
            signal.metadata["overall_quality"] = overall_quality
            signal.quality_scores = quality_scores
            signal.overall_quality = overall_quality

            # Filter based on minimum quality threshold
            if overall_quality >= 0.4:  # Minimum quality threshold
                evaluated_signals.append(signal)

        # Sort by overall quality (highest first)
        evaluated_signals.sort(
            key=lambda s: getattr(s, "overall_quality", 0), reverse=True
        )

        return evaluated_signals
