"""
SignalGenerator Component.

This component is responsible for generating trading signals from observations.
Follows Single Responsibility Principle by focusing only on signal generation.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from ztb.utils.logging_utils import get_logger

from .interfaces import ISignalGenerator

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal, ActionSignalGuideConfig
    from ..types import SignalList

from ..pattern_recognition.base import PatternRecognizer
from .interfaces import IPatternStatistics, IPerformanceTracker


def _get_action_signal_class():
    """Lazy import to avoid circular imports."""
    from ..action_signal_guide import ActionSignal

    return ActionSignal


def _get_guidance_level_enum():
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

        # Initialize recognizers
        self.initialize_recognizers()

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
            from ..pattern_recognition.oscillator_patterns import (
                CCIRecognizer,
                MFIRecognizer,
                StochasticRecognizer,
                WilliamsRRecognizer,
            )
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

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_index: int,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Generate trading signal from market data.

        Args:
            data: OHLCV DataFrame
            current_index: Current bar index
            multi_timeframe_data: Optional multi-timeframe data

        Returns:
            Generated action signal
        """
        start_time = time.time()

        try:
            # Generate signals from all recognizers
            all_signals = []
            pattern_signals: Dict[str, List[Any]] = {}

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

            ActionSignal = _get_action_signal_class()

            for recognizer in recognizers_to_run:
                try:
                    # Check if we have enough data for this recognizer
                    lookback_period = getattr(
                        recognizer, "get_lookback_period", lambda: 20
                    )()
                    if current_index < lookback_period:
                        continue  # Skip this recognizer if not enough data

                    signal_result = recognizer.recognize(
                        data,
                        index=current_index,
                        multi_timeframe_data=multi_timeframe_data,
                    )

                    if signal_result is not None:
                        # Create ActionSignal from pattern result
                        import pandas as pd

                        action_signal = ActionSignal(
                            timestamp=pd.Timestamp.now(),
                            direction=signal_result.direction,
                            strength=signal_result.strength,
                            confidence=signal_result.confidence
                            or signal_result.strength,
                            signal_type=recognizer.pattern_type,
                            description=f"{recognizer.name}: {signal_result.description}",
                            metadata={
                                **signal_result.metadata,
                                "confidence": signal_result.confidence
                                or signal_result.strength,
                                "risk_level": signal_result.risk_level,
                                "validity_period": signal_result.validity_period,
                            },
                            source_patterns=[recognizer.name],
                        )

                        all_signals.append(action_signal)
                        pattern_type = recognizer.pattern_type

                        if pattern_type not in pattern_signals:
                            pattern_signals[pattern_type] = []
                        pattern_signals[pattern_type].append(action_signal)

                        # Record pattern statistics
                        if self.pattern_statistics:
                            self.pattern_statistics.record_pattern_signal(
                                pattern_type, action_signal
                            )

                except Exception as e:
                    # Avoid spamming the same error message repeatedly
                    msg = f"Recognizer {recognizer.name} failed: {e}"
                    self._record_error(msg)
                    continue

            # Aggregate signals based on guidance level
            final_signal = self._aggregate_signals(all_signals, pattern_signals)

            processing_time = time.time() - start_time

            if self.performance_tracker:
                self.performance_tracker.record_signal_generation(processing_time)

            return final_signal

        except Exception as e:
            # Use internal error recording to avoid noisy repeated logs
            self._record_error(f"Signal generation failed: {e}")
            processing_time = time.time() - start_time

            if self.performance_tracker:
                self.performance_tracker.record_signal_generation(processing_time)

            # Return neutral signal on failure
            return ActionSignal.neutral()

    def _record_error(self, message: str) -> None:
        """Record and selectively log errors to prevent log spam from repeated identical messages.

        Logs the first `error_suppression_threshold` occurrences of the same message as warnings.
        Further occurrences are suppressed; when suppression first happens an info-level note
        is logged explaining that further identical messages will be suppressed.
        """
        try:
            threshold = getattr(self.config, "error_suppression_threshold", 3)
        except Exception:
            threshold = 3

        count = self._error_counts.get(message, 0) + 1
        self._error_counts[message] = count

        if count <= threshold:
            # Log the warning normally
            self.logger.warning(message)
            if count == threshold:
                # Warn user that future identical messages will be suppressed
                self.logger.info(
                    f"Further identical errors for message will be suppressed: '{message}'"
                )
        else:
            # Suppress additional identical warnings to reduce spam
            # Do nothing here; counts are still tracked for diagnostics
            pass

    def _aggregate_signals(
        self, all_signals: "SignalList", pattern_signals: dict
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

        # Aggregate signals using weighted average of directions
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

        # Create final ActionSignal
        import pandas as pd

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
            # Any detectable signals
            return [s for s in signals if s.strength >= 0.3 and s.confidence >= 0.3]
        else:
            # Default to moderate
            return [s for s in signals if s.strength >= 0.6 and s.confidence >= 0.6]
