"""
SignalGenerator Component.

This component is responsible for generating trading signals from observations.
Follows Single Responsibility Principle by focusing only on signal generation.
"""

import time
from typing import List, Optional, TYPE_CHECKING, Any

import numpy as np

from ztb.utils.logging_utils import get_logger

from .interfaces import ISignalGenerator

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal, GuidanceLevel, ActionSignalGuideConfig
    from ..types import SignalList

from ..pattern_recognition.base import PatternRecognizer
from .interfaces import IPerformanceTracker, IPatternStatistics


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
        config: 'ActionSignalGuideConfig',
        performance_tracker: Optional['IPerformanceTracker'] = None,
        pattern_statistics: Optional['IPatternStatistics'] = None,
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

        # Initialize recognizers
        self.initialize_recognizers()

    def initialize_recognizers(self) -> None:
        """Initialize all pattern recognition systems."""
        start_time = time.time()

        try:
            # Import all recognizer classes
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
            from ..pattern_recognition.wave_counting import (
                ImpulseWaveRecognizer,
                CorrectiveWaveRecognizer,
                WaveExtensionRecognizer,
                WaveIRecognizer,
                WaveVRecognizer,
                WaveYRecognizer,
                WavePRecognizer,
                WaveNRecognizer,
                WaveSRecognizer,
            )
            from ..pattern_recognition.oscillator_patterns import (
                CCIRecognizer,
                StochasticRecognizer,
                WilliamsRRecognizer,
                MFIRecognizer,
            )
            from ..pattern_recognition.volume_patterns import ChaikinADRecognizer
            from ..pattern_recognition.bollinger_patterns import BollingerBandsRecognizer
            from ..pattern_recognition.adx_patterns import ADXRecognizer
            from ..pattern_recognition.heikin_ashi_patterns import HeikinAshiRecognizer
            from ..pattern_recognition.dow_theory import DowTheoryRecognizer

            # Initialize all recognizers
            self.all_recognizers = [
                # Candlestick patterns
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

                # Fibonacci patterns
                FibonacciExtensionRecognizer(),
                FibonacciProjectionRecognizer(),
                FibonacciRetracementRecognizer(),

                # Gann analysis
                GannAngleRecognizer(),
                GannSquareRecognizer(),
                GannTimeClusterRecognizer(),

                # Granville law
                GranvilleLawRecognizer(),

                # Harmonic patterns
                BatRecognizer(),
                ButterflyRecognizer(),
                CrabRecognizer(),
                GartleyRecognizer(),

                # Wave counting
                ImpulseWaveRecognizer(),
                CorrectiveWaveRecognizer(),
                WaveExtensionRecognizer(),
                WaveIRecognizer(),
                WaveVRecognizer(),
                WaveYRecognizer(),
                WavePRecognizer(),
                WaveNRecognizer(),
                WaveSRecognizer(),

                # Oscillator patterns
                CCIRecognizer(),
                StochasticRecognizer(),
                WilliamsRRecognizer(),
                MFIRecognizer(),

                # Volume patterns
                ChaikinADRecognizer(),

                # Bollinger patterns
                BollingerBandsRecognizer(),

                # ADX patterns
                ADXRecognizer(),

                # Heikin-Ashi patterns
                HeikinAshiRecognizer(),

                # Dow Theory
                DowTheoryRecognizer(),
            ]

            initialization_time = time.time() - start_time
            self.logger.info(
                f"Initialized {len(self.all_recognizers)} pattern recognizers in {initialization_time:.3f}s"
            )

            if self.performance_tracker:
                self.performance_tracker.record_signal_generation(initialization_time)

        except Exception as e:
            self.logger.error(f"Failed to initialize recognizers: {e}")
            self.all_recognizers = []

    def generate_signal(self, observation: np.ndarray, step: int, multi_timeframe_data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generate trading signal from observation.

        Args:
            observation: Current market observation
            step: Current step number

        Returns:
            Generated action signal
        """
        start_time = time.time()

        try:
            # Generate signals from all recognizers
            all_signals = []
            pattern_signals: Dict[str, List[Any]] = {}

            ActionSignal = _get_action_signal_class()

            for recognizer in self.all_recognizers:
                try:
                    signal_result = recognizer.recognize(observation, multi_timeframe_data=multi_timeframe_data)

                    if signal_result.detected:
                        action_signal = ActionSignal(
                            action=signal_result.action,
                            strength=signal_result.strength,
                            confidence=signal_result.confidence,
                            pattern_type=recognizer.pattern_type,
                            pattern_name=recognizer.name,
                            metadata=signal_result.metadata
                        )

                        all_signals.append(action_signal)
                        pattern_type = recognizer.pattern_type

                        if pattern_type not in pattern_signals:
                            pattern_signals[pattern_type] = []
                        pattern_signals[pattern_type].append(action_signal)

                        # Record pattern statistics
                        if self.pattern_statistics:
                            self.pattern_statistics.record_pattern_signal(pattern_type, action_signal)

                except Exception as e:
                    self.logger.warning(f"Recognizer {recognizer.name} failed: {e}")
                    continue

            # Aggregate signals based on guidance level
            final_signal = self._aggregate_signals(all_signals, pattern_signals)

            processing_time = time.time() - start_time

            if self.performance_tracker:
                self.performance_tracker.record_signal_generation(processing_time)

            return final_signal

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            processing_time = time.time() - start_time

            if self.performance_tracker:
                self.performance_tracker.record_signal_generation(processing_time)

            # Return neutral signal on failure
            return ActionSignal.neutral()

    def _aggregate_signals(
        self,
        all_signals: 'SignalList',
        pattern_signals: dict
    ) -> 'ActionSignal':
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

        # Aggregate by action type
        buy_signals = [s for s in filtered_signals if s.action == 1]  # BUY
        sell_signals = [s for s in filtered_signals if s.action == 2]  # SELL
        hold_signals = [s for s in filtered_signals if s.action == 0]  # HOLD

        # Calculate weighted strengths
        buy_strength = sum(s.strength * s.confidence for s in buy_signals) / len(buy_signals) if buy_signals else 0
        sell_strength = sum(s.strength * s.confidence for s in sell_signals) / len(sell_signals) if sell_signals else 0
        hold_strength = sum(s.strength * s.confidence for s in hold_signals) / len(hold_signals) if hold_signals else 0

        # Determine final action
        max_strength = max(buy_strength, sell_strength, hold_strength)

        if max_strength == 0:
            return ActionSignal.neutral()

        if buy_strength == max_strength:
            action = 1  # BUY
            strength = buy_strength
            confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
        elif sell_strength == max_strength:
            action = 2  # SELL
            strength = sell_strength
            confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
        else:
            action = 0  # HOLD
            strength = hold_strength
            confidence = sum(s.confidence for s in hold_signals) / len(hold_signals)

        # Create metadata
        metadata = {
            'total_signals': len(filtered_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'hold_signals': len(hold_signals),
            'pattern_types': list(pattern_signals.keys()),
            'guidance_level': self.guidance_level.value
        }

        return ActionSignal(
            action=action,
            strength=strength,
            confidence=confidence,
            pattern_type='aggregated',
            pattern_name='multi_pattern_aggregate',
            metadata=metadata
        )

    def _filter_by_guidance_level(self, signals: 'SignalList') -> 'SignalList':
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