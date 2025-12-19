"""
Recognizer Factory

This module provides a factory for creating pattern recognizers.
"""

from typing import Any, Callable, Dict, List

from .pattern_recognition.adx_patterns import ADXRecognizer
from .pattern_recognition.atr import ATRPatternRecognizer
from .pattern_recognition.base import PatternRecognizer
from .pattern_recognition.bollinger_patterns import BollingerBandsRecognizer
from .pattern_recognition.candlestick_patterns import (
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
from .pattern_recognition.dow_theory import DowTheoryRecognizer
from .pattern_recognition.fibonacci_patterns import (
    FibonacciExtensionRecognizer,
    FibonacciProjectionRecognizer,
    FibonacciRetracementRecognizer,
)
from .pattern_recognition.gann_analysis import (
    GannAngleRecognizer,
    GannSquareRecognizer,
    GannTimeClusterRecognizer,
)
from .pattern_recognition.granville_law import GranvilleLawRecognizer
from .pattern_recognition.harmonic_patterns import (
    BatRecognizer,
    ButterflyRecognizer,
    CrabRecognizer,
    GartleyRecognizer,
)
from .pattern_recognition.heikin_ashi import HeikinAshiRecognizer
from .pattern_recognition.macd import MACDPatternRecognizer
from .pattern_recognition.oscillator_patterns import (
    CCIRecognizer,
    MFIRecognizer,
    StochasticRecognizer,
    WilliamsRRecognizer,
)
from .pattern_recognition.volume_patterns import ChaikinADRecognizer
from .pattern_recognition.rsi import RSIPatternRecognizer
from .pattern_recognition.trend_analyzer import HierarchicalTrendAnalyzer
from .pattern_recognition.wave_counting import (
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


class RecognizerFactory:
    """Factory for creating pattern recognizers."""

    def __init__(self) -> None:
        self._factory_map: Dict[str, Callable[[Any], PatternRecognizer]] = {
            # Candlestick patterns
            "sakata_five_methods": lambda config: SakataFiveMethodsRecognizer(config),
            "morning_star": lambda config: MorningStarRecognizer(config),
            "evening_star": lambda config: EveningStarRecognizer(config),
            "hammer": lambda config: HammerRecognizer(config),
            "hanging_man": lambda config: HangingManRecognizer(config),
            "three_black_crows": lambda config: ThreeBlackCrowsRecognizer(config),
            "three_white_soldiers": lambda config: ThreeWhiteSoldiersRecognizer(config),
            "rising_three_methods": lambda config: RisingThreeMethodsRecognizer(config),
            "bullish_engulfing": lambda config: BullishEngulfingRecognizer(config),
            "bearish_engulfing": lambda config: BearishEngulfingRecognizer(config),
            "piercing_pattern": lambda config: PiercingPatternRecognizer(config),
            # Fibonacci patterns
            "fibonacci_retracement": lambda config: FibonacciRetracementRecognizer(
                config
            ),
            "fibonacci_extension": lambda config: FibonacciExtensionRecognizer(config),
            "fibonacci_projection": lambda config: FibonacciProjectionRecognizer(
                config
            ),
            # Gann patterns
            "gann_angle": lambda config: GannAngleRecognizer(config),
            "gann_square": lambda config: GannSquareRecognizer(config),
            "gann_time_cluster": lambda config: GannTimeClusterRecognizer(config),
            # Wave patterns
            "impulse_wave": lambda config: ImpulseWaveRecognizer(config),
            "corrective_wave": lambda config: CorrectiveWaveRecognizer(config),
            "wave_extension": lambda config: WaveExtensionRecognizer(config),
            "wave_i": lambda config: WaveIRecognizer(config),
            "wave_v": lambda config: WaveVRecognizer(config),
            "wave_y": lambda config: WaveYRecognizer(config),
            "wave_p": lambda config: WavePRecognizer(config),
            "wave_n": lambda config: WaveNRecognizer(config),
            "wave_s": lambda config: WaveSRecognizer(config),
            # Harmonic patterns
            "gartley": lambda config: GartleyRecognizer(config),
            "butterfly": lambda config: ButterflyRecognizer(config),
            "bat": lambda config: BatRecognizer(config),
            "crab": lambda config: CrabRecognizer(config),
            # Oscillator patterns
            "cci": lambda config: CCIRecognizer(config),
            "stochastic": lambda config: StochasticRecognizer(config),
            "williams_r": lambda config: WilliamsRRecognizer(config),
            "mfi": lambda config: MFIRecognizer(config),
            "atr": lambda config: ATRPatternRecognizer(config),
            "rsi": lambda config: RSIPatternRecognizer(config),
            "macd": lambda config: MACDPatternRecognizer(config),
            # Volume patterns
            "chaikin_ad": lambda config: ChaikinADRecognizer(config),
            # Bollinger Bands patterns
            "bollinger_bands": lambda config: BollingerBandsRecognizer(config),
            # ADX patterns
            "adx": lambda config: ADXRecognizer(config),
            # Hierarchical trend analyzer (統合トレンド分析)
            "hierarchical_trend": lambda config: HierarchicalTrendAnalyzer(config),
            # Other patterns
            "granville_law": lambda config: GranvilleLawRecognizer(config),
            "heikin_ashi": lambda config: HeikinAshiRecognizer(config),
            "dow_theory": lambda config: DowTheoryRecognizer(config),
        }

    def create_recognizer(self, name: str, config: Any) -> PatternRecognizer:
        """Create a recognizer by name."""
        if name not in self._factory_map:
            raise ValueError(f"Unknown recognizer: {name}")
        return self._factory_map[name](config)

    def get_available_recognizers(self) -> List[str]:
        """Get list of available recognizer names."""
        return list(self._factory_map.keys())
