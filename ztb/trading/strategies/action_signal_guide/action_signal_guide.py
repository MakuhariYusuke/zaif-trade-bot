"""
Action Signal Guide - Main Implementation

This module provides the main ActionSignalGuide class that integrates all pattern
recognition systems for classical technical analysis signals in the SAC RL system.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from .components.cache_manager import CacheManager

# Import components
from .components.cache_manager import CacheManager
from .components.pattern_statistics import PatternStatistics
from .components.performance_tracker import PerformanceTracker
from .components.signal_generator import SignalGenerator
from .analysis.signal_performance_analyzer import SignalPerformanceAnalyzer
from .pattern_recognition.base import PatternRecognizer
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

# Import types
from .types import (
    CacheStats,
    ConfigInput,
    GuidanceInput,
    PatternConfig,
    PatternStats,
    PerformanceStats,
    RecognizerStatus,
    SignalHistory,
    SignalList,
    SignalMetadata,
)


def _get_action_signal_guide_config() -> Any:
    """Lazy import to avoid circular imports."""
    from .action_signal_guide import ActionSignalGuideConfig

    return ActionSignalGuideConfig


def _get_guidance_level_enum() -> Any:
    """Lazy import to avoid circular imports."""
    from .action_signal_guide import GuidanceLevel

    return GuidanceLevel


def _get_action_signal_class() -> Any:
    """Lazy import to avoid circular imports."""
    from .action_signal_guide import ActionSignal

    return ActionSignal


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


class GuidanceLevel(Enum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    FULL = "full"


@dataclass
class RecognizerConfig:
    """Configuration for a pattern recognizer."""

    name: str
    enabled: bool = True
    weight: float = 1.0
    config: Optional[PatternConfig] = None
    group: str = "default"


@dataclass
class ActionSignalGuideConfig:
    """Configuration for ActionSignalGuide."""

    guidance_level: GuidanceLevel = GuidanceLevel.STRONG
    max_signals_per_bar: int = 3
    enable_parallel_processing: bool = False
    enable_caching: bool = True
    cache_size: int = 1000
    lazy_loading: bool = False
    feature_names: Optional[List[str]] = None
    value: Optional[Any] = None  # For backward compatibility

    # Short / debug mode for fast unit tests and debugging
    debug_short_mode: bool = False
    short_mode_recognizer_limit: int = 8
    # How many identical errors to allow before suppressing further identical log lines
    error_suppression_threshold: int = 3

    # Pattern group enable/disable flags for validation
    enable_candlestick_patterns: bool = True
    enable_fibonacci_patterns: bool = True
    enable_gann_patterns: bool = True
    enable_wave_patterns: bool = True
    enable_harmonic_patterns: bool = True
    enable_oscillator_patterns: bool = True
    enable_volume_patterns: bool = True
    enable_bollinger_patterns: bool = True
    enable_adx_patterns: bool = True
    enable_granville_patterns: bool = True
    enable_heikin_ashi_patterns: bool = True
    enable_dow_theory_patterns: bool = True

    # Recognizer configurations
    candlestick_patterns: Optional[List[RecognizerConfig]] = None
    fibonacci_patterns: Optional[List[RecognizerConfig]] = None
    gann_patterns: Optional[List[RecognizerConfig]] = None
    wave_patterns: Optional[List[RecognizerConfig]] = None
    harmonic_patterns: Optional[List[RecognizerConfig]] = None
    oscillator_patterns: Optional[List[RecognizerConfig]] = None
    volume_patterns: Optional[List[RecognizerConfig]] = None
    bollinger_patterns: Optional[List[RecognizerConfig]] = None
    adx_patterns: Optional[List[RecognizerConfig]] = None
    granville_patterns: Optional[List[RecognizerConfig]] = None
    heikin_ashi_patterns: Optional[List[RecognizerConfig]] = None
    dow_theory_patterns: Optional[List[RecognizerConfig]] = None

    def __post_init__(self) -> None:
        """Initialize default configurations if not provided."""
        if self.candlestick_patterns is None:
            self.candlestick_patterns = [
                RecognizerConfig("sakata_five_methods", group="candlestick"),
                RecognizerConfig("morning_star", group="candlestick"),
                RecognizerConfig("evening_star", group="candlestick"),
                RecognizerConfig("hammer", group="candlestick"),
                RecognizerConfig("hanging_man", group="candlestick"),
                RecognizerConfig("three_black_crows", group="candlestick"),
                RecognizerConfig("three_white_soldiers", group="candlestick"),
                RecognizerConfig("rising_three_methods", group="candlestick"),
                RecognizerConfig("bullish_engulfing", group="candlestick"),
                RecognizerConfig("bearish_engulfing", group="candlestick"),
                RecognizerConfig("piercing_pattern", group="candlestick"),
            ]

        if self.fibonacci_patterns is None:
            self.fibonacci_patterns = [
                RecognizerConfig("fibonacci_retracement", group="fibonacci"),
                RecognizerConfig("fibonacci_extension", group="fibonacci"),
                RecognizerConfig("fibonacci_projection", group="fibonacci"),
            ]

        if self.gann_patterns is None:
            self.gann_patterns = [
                RecognizerConfig("gann_angle", group="gann"),
                RecognizerConfig("gann_square", group="gann"),
                RecognizerConfig("gann_time_cluster", group="gann"),
            ]

        if self.wave_patterns is None:
            self.wave_patterns = [
                RecognizerConfig("impulse_wave", group="wave"),
                RecognizerConfig("corrective_wave", group="wave"),
                RecognizerConfig("wave_extension", group="wave"),
                RecognizerConfig("wave_i", group="wave"),
                RecognizerConfig("wave_v", group="wave"),
                RecognizerConfig("wave_y", group="wave"),
                RecognizerConfig("wave_p", group="wave"),
                RecognizerConfig("wave_n", group="wave"),
                RecognizerConfig("wave_s", group="wave"),
            ]

        if self.harmonic_patterns is None:
            self.harmonic_patterns = [
                RecognizerConfig("gartley", group="harmonic"),
                RecognizerConfig("butterfly", group="harmonic"),
                RecognizerConfig("bat", group="harmonic"),
                RecognizerConfig("crab", group="harmonic"),
            ]

        if self.oscillator_patterns is None:
            self.oscillator_patterns = [
                RecognizerConfig("cci", group="oscillator"),
                RecognizerConfig("stochastic", group="oscillator"),
                RecognizerConfig("williams_r", group="oscillator"),
                RecognizerConfig("mfi", group="oscillator"),
            ]

        if self.volume_patterns is None:
            self.volume_patterns = [
                RecognizerConfig("chaikin_ad", group="volume"),
            ]

        if self.granville_patterns is None:
            self.granville_patterns = [
                RecognizerConfig("granville_law", group="granville"),
            ]

        if self.heikin_ashi_patterns is None:
            self.heikin_ashi_patterns = [
                RecognizerConfig("heikin_ashi", group="heikin_ashi"),
            ]

        if self.dow_theory_patterns is None:
            self.dow_theory_patterns = [
                RecognizerConfig("dow_theory", group="dow_theory"),
            ]

        if self.bollinger_patterns is None:
            self.bollinger_patterns = [
                RecognizerConfig("bollinger_bands", group="bollinger"),
            ]

        if self.adx_patterns is None:
            self.adx_patterns = [
                RecognizerConfig(
                    "adx",
                    group="adx",
                    config={
                        "enable_multi_timeframe": True,
                        "mtf_weight": 0.3,
                        "regime_aware": True,
                        "period": 14,
                        "strong_trend_threshold": 25,
                        "weak_trend_threshold": 20,
                        "di_cross_threshold": 1.0,
                    }
                ),
            ]


@dataclass
class ActionSignal:
    """Represents a complete action signal with all relevant information."""

    timestamp: pd.Timestamp
    direction: float  # Continuous value from -1.0 (strong sell) to 1.0 (strong buy), 0.0 for hold
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signal_type: str
    description: str
    metadata: SignalMetadata
    source_patterns: List[str]  # List of pattern names that contributed

    @classmethod
    def neutral(cls) -> "ActionSignal":
        """Create a neutral (hold) signal."""
        return cls(
            timestamp=pd.Timestamp.now(),
            direction=0,
            strength=0.0,
            confidence=0.0,
            signal_type="neutral",
            description="No clear signal detected",
            metadata={},
            source_patterns=[],
        )


class ActionSignalGuide:
    """
    Main class for generating classical technical analysis signals.

    This class integrates multiple pattern recognition systems to provide
    comprehensive technical analysis signals for the SAC RL training system.

    Now refactored to use component-based architecture following SOLID principles.
    """

    def __init__(
        self, guidance_level: GuidanceInput = None, config: ConfigInput = None
    ) -> None:
        # Lazy import to avoid circular imports
        ActionSignalGuideConfig = _get_action_signal_guide_config()
        GuidanceLevel = _get_guidance_level_enum()

        if guidance_level is None:
            guidance_level = GuidanceLevel.STRONG

        self.config = config or ActionSignalGuideConfig(guidance_level=guidance_level)
        self.guidance_level = self.config.guidance_level

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize components following SOLID principles
        self.cache_manager = CacheManager(
            max_cache_size=self.config.cache_size, cache_ttl=300
        )  # 5 minutes TTL

        self.performance_tracker = PerformanceTracker(enable_detailed_tracking=True)

        self.pattern_statistics = PatternStatistics(max_history_size=10000)

        # Initialize signal performance analyzer for SAC learning correlation analysis
        self.signal_performance_analyzer = SignalPerformanceAnalyzer(
            performance_tracker=self.performance_tracker,
            pattern_statistics=self.pattern_statistics,
        )

        # Initialize signal generator with dependencies
        self.signal_generator = SignalGenerator(
            config=cast(ActionSignalGuideConfig, self.config),
            performance_tracker=self.performance_tracker,
            pattern_statistics=self.pattern_statistics,
        )

        # Initialize multi-timeframe feature system for enhanced pattern validation
        try:
            from ztb.features.multi_timeframe import create_multi_timeframe_system

            self.multi_timeframe_system = create_multi_timeframe_system()
            self.use_multi_timeframe = True
            self.logger.info("Multi-timeframe feature system initialized")
        except ImportError:
            self.multi_timeframe_system = None
            self.use_multi_timeframe = False
            self.logger.warning("Multi-timeframe feature system not available")

        # Signal history for context
        self.signal_history: SignalHistory = []

        # Feature names for observation conversion
        self.feature_names: Optional[List[str]] = None

        # Initialize all pattern recognizers
        self._initialize_recognizers()

        self.logger.info(
            "ActionSignalGuide initialized with component-based architecture"
        )

    def _initialize_recognizers(self) -> None:
        """Initialize all pattern recognition systems using configuration."""
        from .pattern_recognition.adx_patterns import ADXRecognizer
        from .pattern_recognition.bollinger_patterns import BollingerBandsRecognizer
        from .pattern_recognition.heikin_ashi import HeikinAshiRecognizer
        from .pattern_recognition.oscillator_patterns import (
            CCIRecognizer,
            MFIRecognizer,
            StochasticRecognizer,
            WilliamsRRecognizer,
        )
        from .pattern_recognition.atr import ATRPatternRecognizer
        from .pattern_recognition.rsi import RSIPatternRecognizer
        from .pattern_recognition.macd import MACDPatternRecognizer
        from .pattern_recognition.volume_patterns import ChaikinADRecognizer

        # Recognizer factory mapping
        self._recognizer_factory = {
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
            # Other patterns
            "granville_law": lambda config: GranvilleLawRecognizer(config),
            "heikin_ashi": lambda config: HeikinAshiRecognizer(config),
            "dow_theory": lambda config: DowTheoryRecognizer(config),
        }

        if self.config.lazy_loading:
            # Lazy loading: store configurations but don't initialize recognizers yet
            self._recognizer_configs = {
                "candlestick": self.config.candlestick_patterns,
                "fibonacci": self.config.fibonacci_patterns,
                "gann": self.config.gann_patterns,
                "wave": self.config.wave_patterns,
                "harmonic": self.config.harmonic_patterns,
                "oscillator": self.config.oscillator_patterns,
                "volume": self.config.volume_patterns,
                "bollinger": self.config.bollinger_patterns,
                "adx": self.config.adx_patterns,
                "granville": self.config.granville_patterns,
                "heikin_ashi": self.config.heikin_ashi_patterns,
                "dow_theory": self.config.dow_theory_patterns,
            }

            # Initialize empty lists for lazy-loaded recognizers
            self.candlestick_recognizers = []
            self.fibonacci_recognizers = []
            self.gann_recognizers = []
            self.wave_recognizers = []
            self.harmonic_recognizers = []
            self.oscillator_recognizers = []
            self.volume_recognizers = []
            self.bollinger_recognizers = []
            self.adx_recognizers = []
            self.granville_recognizers = []
            self.heikin_ashi_recognizers = []
            self.dow_theory_recognizers = []
            # all_recognizers will be set below
        else:
            # Eager loading: initialize all recognizers immediately
            self.candlestick_recognizers = (
                self._create_recognizers_from_config(self.config.candlestick_patterns)
                if self.config.enable_candlestick_patterns
                else []
            )
            self.fibonacci_recognizers = (
                self._create_recognizers_from_config(self.config.fibonacci_patterns)
                if self.config.enable_fibonacci_patterns
                else []
            )
            self.gann_recognizers = (
                self._create_recognizers_from_config(self.config.gann_patterns)
                if self.config.enable_gann_patterns
                else []
            )
            self.wave_recognizers = (
                self._create_recognizers_from_config(self.config.wave_patterns)
                if self.config.enable_wave_patterns
                else []
            )
            self.harmonic_recognizers = (
                self._create_recognizers_from_config(self.config.harmonic_patterns)
                if self.config.enable_harmonic_patterns
                else []
            )
            self.oscillator_recognizers = (
                self._create_recognizers_from_config(self.config.oscillator_patterns)
                if self.config.enable_oscillator_patterns
                else []
            )
            self.volume_recognizers = (
                self._create_recognizers_from_config(self.config.volume_patterns)
                if self.config.enable_volume_patterns
                else []
            )
            self.bollinger_recognizers = (
                self._create_recognizers_from_config(self.config.bollinger_patterns)
                if self.config.enable_bollinger_patterns
                else []
            )
            self.adx_recognizers = (
                self._create_recognizers_from_config(self.config.adx_patterns)
                if self.config.enable_adx_patterns
                else []
            )
            self.granville_recognizers = (
                self._create_recognizers_from_config(self.config.granville_patterns)
                if self.config.enable_granville_patterns
                else []
            )
            self.heikin_ashi_recognizers = (
                self._create_recognizers_from_config(self.config.heikin_ashi_patterns)
                if self.config.enable_heikin_ashi_patterns
                else []
            )
            self.dow_theory_recognizers = (
                self._create_recognizers_from_config(self.config.dow_theory_patterns)
                if self.config.enable_dow_theory_patterns
                else []
            )

            # Combine all recognizers
            self.all_recognizers: List[PatternRecognizer] = cast(
                List[PatternRecognizer],
                list(
                    self.candlestick_recognizers
                    + self.fibonacci_recognizers
                    + self.gann_recognizers
                    + self.wave_recognizers
                    + self.harmonic_recognizers
                    + self.oscillator_recognizers
                    + self.volume_recognizers
                    + self.bollinger_recognizers
                    + self.adx_recognizers
                    + self.granville_recognizers
                    + self.heikin_ashi_recognizers
                    + self.dow_theory_recognizers
                ),
            )

        # Initialize caching (always initialize, but only use if enabled)
        self._signal_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def _ensure_recognizers_loaded(self) -> None:
        """Ensure all recognizers are loaded for lazy loading mode."""
        if not self.config.lazy_loading or not hasattr(self, "_recognizer_configs"):
            return

        # Load all recognizer groups if not already loaded
        if not self.candlestick_recognizers and self._recognizer_configs["candlestick"]:
            self.candlestick_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["candlestick"]
            )

        if not self.fibonacci_recognizers and self._recognizer_configs["fibonacci"]:
            self.fibonacci_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["fibonacci"]
            )

        if not self.gann_recognizers and self._recognizer_configs["gann"]:
            self.gann_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["gann"]
            )

        if not self.wave_recognizers and self._recognizer_configs["wave"]:
            self.wave_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["wave"]
            )

        if not self.harmonic_recognizers and self._recognizer_configs["harmonic"]:
            self.harmonic_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["harmonic"]
            )

        if not self.oscillator_recognizers and self._recognizer_configs["oscillator"]:
            self.oscillator_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["oscillator"]
            )

        if not self.volume_recognizers and self._recognizer_configs["volume"]:
            self.volume_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["volume"]
            )

        if not self.bollinger_recognizers and self._recognizer_configs["bollinger"]:
            self.bollinger_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["bollinger"]
            )

        if not self.adx_recognizers and self._recognizer_configs["adx"]:
            self.adx_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["adx"]
            )

        if not self.granville_recognizers and self._recognizer_configs["granville"]:
            self.granville_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["granville"]
            )

        if not self.heikin_ashi_recognizers and self._recognizer_configs["heikin_ashi"]:
            self.heikin_ashi_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["heikin_ashi"]
            )

        if not self.dow_theory_recognizers and self._recognizer_configs["dow_theory"]:
            self.dow_theory_recognizers = self._create_recognizers_from_config(
                self._recognizer_configs["dow_theory"]
            )

        # Update all_recognizers list
        self.all_recognizers = cast(
            List[PatternRecognizer],
            list(
                self.candlestick_recognizers
                + self.fibonacci_recognizers
                + self.gann_recognizers
                + self.wave_recognizers
                + self.harmonic_recognizers
                + self.oscillator_recognizers
                + self.volume_recognizers
                + self.bollinger_recognizers
                + self.adx_recognizers
                + self.granville_recognizers
                + self.heikin_ashi_recognizers
                + self.dow_theory_recognizers
            ),
        )

        # Clear configs after loading
        if hasattr(self, "_recognizer_configs"):
            delattr(self, "_recognizer_configs")

    def _create_recognizers_from_config(
        self, configs: Optional[List[RecognizerConfig]]
    ) -> List[PatternRecognizer]:
        """Create recognizers from configuration."""
        if not configs:
            return []

        recognizers = []
        for config in configs:
            if config.enabled and config.name in self._recognizer_factory:
                try:
                    recognizer = self._recognizer_factory[config.name](config.config)
                    recognizers.append(recognizer)
                    self.logger.debug(f"Successfully created recognizer: {config.name}")
                except Exception as e:
                    self.logger.error(f"Failed to create recognizer {config.name}: {e}")
                    # Continue with other recognizers instead of failing completely
                    continue
            elif not config.enabled:
                self.logger.debug(f"Recognizer {config.name} is disabled, skipping")
            else:
                self.logger.warning(f"Unknown recognizer name: {config.name}")

        return recognizers

    def generate_signals(self, data: pd.DataFrame, current_index: int) -> SignalList:
        """
        Generate action signals for the current market data.

        Now uses component-based architecture for better maintainability.

        Args:
            data: OHLCV DataFrame
            current_index: Current bar index to analyze

        Returns:
            List of ActionSignal objects
        """
        start_time = time.time()

        if current_index >= len(data):
            return []

        # Check cache first using CacheManager
        cache_key = self._get_cache_key(data, current_index)
        if self.config.enable_caching:
            cached_signals = self.cache_manager.get_cached_signal(cache_key)
            if cached_signals is not None:
                # Check if cache is still valid (within last few bars)
                # Note: CacheManager handles TTL, but we add additional logic for bar proximity
                processing_time = time.time() - start_time
                self.performance_tracker.record_cache_operation(
                    processing_time, hit=True
                )
                # Ensure cached_signals is always a list
                if isinstance(cached_signals, list):
                    return cached_signals
                else:
                    return [cached_signals]

        # Convert DataFrame row to observation array for SignalGenerator
        try:
            observation = self._convert_to_observation(data, current_index)
        except Exception as e:
            self.logger.error(
                f"Failed to convert data to observation at index {current_index}: {e}"
            )
            processing_time = time.time() - start_time
            self.performance_tracker.record_cache_operation(processing_time, hit=False)
            return []

        # Generate multi-timeframe features if available
        multi_timeframe_data = None
        if self.use_multi_timeframe and self.multi_timeframe_system:
            try:
                # Generate multi-timeframe features for enhanced pattern validation
                mtf_features = self.multi_timeframe_system.process_multi_timeframe_data(
                    data,
                    current_timeframe="1h",  # Assume 1h, could be parameterized
                )
                multi_timeframe_data = {
                    "higher_timeframe_trend": mtf_features.get("trend_strength", 0),
                    "multi_timeframe_support": mtf_features.get(
                        "support_resistance", {}
                    ),
                    "timeframe_alignment": mtf_features.get("timeframe_alignment", 0.5),
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate multi-timeframe features: {e}")
                multi_timeframe_data = None

        # Generate signal using SignalGenerator component
        try:
            signal = self.signal_generator.generate_signal(
                data, current_index, multi_timeframe_data
            )

            # Convert component signal to legacy ActionSignal format
            action_signals = self._convert_to_legacy_signals(
                signal, data, current_index
            )

            # Filter and prioritize signals (legacy logic)
            action_signals = self._filter_and_prioritize_signals(action_signals)

            # Store in history
            self.signal_history.extend(action_signals)

            # Cache results if enabled
            if self.config.enable_caching:
                self.cache_manager.cache_signal(cache_key, action_signals)

            processing_time = time.time() - start_time
            self.performance_tracker.record_cache_operation(processing_time, hit=False)

            return action_signals

        except Exception as e:
            self.logger.error(f"Error generating signals at index {current_index}: {e}")
            processing_time = time.time() - start_time
            self.performance_tracker.record_error("signal_generation", str(e))
            return []

    def _convert_to_observation(
        self, data: pd.DataFrame, current_index: int
    ) -> np.ndarray:
        """
        Convert DataFrame row to observation array for SignalGenerator.

        Args:
            data: OHLCV DataFrame
            current_index: Current bar index

        Returns:
            Observation array
        """
        if current_index >= len(data):
            raise ValueError(
                f"Index {current_index} out of bounds for data with length {len(data)}"
            )

        # Extract current bar data
        current_bar = data.iloc[current_index]

        # Build observation array (OHLCV + any additional features)
        observation_data = [
            current_bar.get("open", 0.0),
            current_bar.get("high", 0.0),
            current_bar.get("low", 0.0),
            current_bar.get("close", 0.0),
            current_bar.get("volume", 0.0),
        ]

        # Add any additional features if available
        if hasattr(current_bar, "index") and len(current_bar.index) > 5:
            for i in range(5, len(current_bar.index)):
                observation_data.append(current_bar.iloc[i])

        return np.array(observation_data, dtype=np.float32)

    def _convert_to_legacy_signals(
        self, signal: ActionSignal, data: pd.DataFrame, current_index: int
    ) -> SignalList:
        """
        Convert SignalGenerator output to legacy ActionSignal format.

        Args:
            signal: Signal from SignalGenerator component
            data: Original DataFrame
            current_index: Current bar index

        Returns:
            List of legacy ActionSignal objects
        """
        if signal.direction == 0:
            return []

        # Create legacy ActionSignal (same format as new ActionSignal)
        ActionSignal = _get_action_signal_class()
        timestamp = (
            pd.Timestamp.now()
            if not hasattr(data.index, "__getitem__")
            else data.index[current_index]
        )

        legacy_signal = ActionSignal(
            timestamp=timestamp,
            direction=signal.direction,
            strength=signal.strength,
            confidence=signal.confidence,
            signal_type=signal.signal_type,
            description=signal.description,
            metadata=signal.metadata,
            source_patterns=signal.source_patterns,
        )

        return [legacy_signal]

    def _get_cache_key(self, data: pd.DataFrame, current_index: int) -> str:
        """
        Generate cache key for the given data and index.

        Args:
            data: DataFrame
            current_index: Current index

        Returns:
            Cache key string
        """
        if current_index >= len(data):
            return f"invalid_{current_index}"

        # Use a combination of timestamp and key data points for uniqueness
        current_bar = data.iloc[current_index]
        timestamp = current_bar.name if hasattr(current_bar, "name") else current_index

        # Create a hash from key data points
        key_data = f"{timestamp}_{current_bar.get('close', 0):.6f}_{current_bar.get('volume', 0):.2f}"
        return f"signal_{hash(key_data) % 1000000}"

    def _adjust_strength_by_guidance(self, base_strength: float) -> float:
        """Adjust signal strength based on guidance level."""
        if self.guidance_level == GuidanceLevel.NONE:
            return base_strength * 0.1  # Very weak guidance
        elif self.guidance_level == GuidanceLevel.WEAK:
            return base_strength * 0.4
        elif self.guidance_level == GuidanceLevel.STRONG:
            return base_strength * 0.8
        elif self.guidance_level == GuidanceLevel.FULL:
            return min(1.0, base_strength * 1.2)
        else:
            # This should never happen, but provides type safety
            raise ValueError(f"Unknown guidance level: {self.guidance_level}")

    def _filter_and_prioritize_signals(self, signals: SignalList) -> SignalList:
        """Filter and prioritize signals to avoid conflicts and redundancy."""
        if not signals:
            return signals

        # Sort by strength (highest first)
        signals.sort(key=lambda x: x.strength, reverse=True)

        # Limit number of signals per bar
        filtered_signals = signals[: self.config.max_signals_per_bar]

        # Check for conflicting signals and resolve
        buy_signals = [
            s for s in filtered_signals if s.direction > 0.1
        ]  # Positive direction signals
        sell_signals = [
            s for s in filtered_signals if s.direction < -0.1
        ]  # Negative direction signals

        # If we have both buy and sell signals, keep only the stronger ones
        if buy_signals and sell_signals:
            # Compare strongest buy vs strongest sell
            strongest_buy = max(
                buy_signals, key=lambda x: abs(x.direction) * x.strength
            )
            strongest_sell = max(
                sell_signals, key=lambda x: abs(x.direction) * x.strength
            )

            if (
                abs(strongest_buy.direction) * strongest_buy.strength
                > abs(strongest_sell.direction) * strongest_sell.strength
            ):
                filtered_signals = [
                    s for s in filtered_signals if s.direction >= -0.1
                ]  # Keep positive and neutral
            elif (
                abs(strongest_sell.direction) * strongest_sell.strength
                > abs(strongest_buy.direction) * strongest_buy.strength
            ):
                filtered_signals = [
                    s for s in filtered_signals if s.direction <= 0.1
                ]  # Keep negative and neutral
            else:
                # Equal strength - keep both but reduce their strength
                for s in filtered_signals:
                    s.strength *= 0.8

        # Record final signals in PatternStatistics component
        for signal in filtered_signals:
            # Note: Pattern statistics are now handled by the SignalGenerator component
            # This method now only handles legacy signal filtering
            pass

        return filtered_signals

    # Component-based public API methods

    def get_recognizer_status(self) -> RecognizerStatus:
        """
        Get status information about all pattern recognizers.

        Returns:
            Dictionary with recognizer status information
        """
        self._ensure_recognizers_loaded()

        # Safely get recognizer counts
        def safe_len(attr_name: str) -> int:
            return len(getattr(self, attr_name, []))

        # Count total recognizers
        total_count = (
            safe_len("candlestick_recognizers")
            + safe_len("fibonacci_recognizers")
            + safe_len("gann_recognizers")
            + safe_len("wave_recognizers")
            + safe_len("harmonic_recognizers")
            + safe_len("oscillator_recognizers")
            + safe_len("volume_recognizers")
            + safe_len("bollinger_recognizers")
            + safe_len("adx_recognizers")
            + safe_len("granville_recognizers")
            + safe_len("heikin_ashi_recognizers")
            + safe_len("dow_theory_recognizers")
        )

        # Safely get recognizer lists
        def safe_list(attr_name: str) -> List[str]:
            recognizers = getattr(self, attr_name, [])
            return [r.__class__.__name__ for r in recognizers]

        status = {
            "total_recognizers": total_count,
            "guidance_level": getattr(
                self.guidance_level, "value", str(self.guidance_level)
            ),
            "recognizer_groups": {
                "candlestick": {
                    "enabled": getattr(
                        self.config, "enable_candlestick_patterns", True
                    ),
                    "count": safe_len("candlestick_recognizers"),
                    "recognizers": safe_list("candlestick_recognizers"),
                },
                "fibonacci": {
                    "enabled": getattr(self.config, "enable_fibonacci_patterns", True),
                    "count": safe_len("fibonacci_recognizers"),
                    "recognizers": safe_list("fibonacci_recognizers"),
                },
                "gann": {
                    "enabled": getattr(self.config, "enable_gann_patterns", True),
                    "count": safe_len("gann_recognizers"),
                    "recognizers": safe_list("gann_recognizers"),
                },
                "wave": {
                    "enabled": getattr(self.config, "enable_wave_patterns", True),
                    "count": safe_len("wave_recognizers"),
                    "recognizers": safe_list("wave_recognizers"),
                },
                "harmonic": {
                    "enabled": getattr(self.config, "enable_harmonic_patterns", True),
                    "count": safe_len("harmonic_recognizers"),
                    "recognizers": safe_list("harmonic_recognizers"),
                },
                "oscillator": {
                    "enabled": getattr(self.config, "enable_oscillator_patterns", True),
                    "count": safe_len("oscillator_recognizers"),
                    "recognizers": safe_list("oscillator_recognizers"),
                },
                "volume": {
                    "enabled": getattr(self.config, "enable_volume_patterns", True),
                    "count": safe_len("volume_recognizers"),
                    "recognizers": safe_list("volume_recognizers"),
                },
                "bollinger": {
                    "enabled": getattr(self.config, "enable_bollinger_patterns", True),
                    "count": safe_len("bollinger_recognizers"),
                    "recognizers": safe_list("bollinger_recognizers"),
                },
                "adx": {
                    "enabled": getattr(self.config, "enable_adx_patterns", True),
                    "count": safe_len("adx_recognizers"),
                    "recognizers": safe_list("adx_recognizers"),
                },
                "granville": {
                    "enabled": getattr(self.config, "enable_granville_patterns", True),
                    "count": safe_len("granville_recognizers"),
                    "recognizers": safe_list("granville_recognizers"),
                },
                "heikin_ashi": {
                    "enabled": getattr(
                        self.config, "enable_heikin_ashi_patterns", True
                    ),
                    "count": safe_len("heikin_ashi_recognizers"),
                    "recognizers": safe_list("heikin_ashi_recognizers"),
                },
                "dow_theory": {
                    "enabled": getattr(self.config, "enable_dow_theory_patterns", True),
                    "count": safe_len("dow_theory_recognizers"),
                    "recognizers": safe_list("dow_theory_recognizers"),
                },
            },
            "config": {
                "max_signals_per_bar": getattr(self.config, "max_signals_per_bar", 3),
                "enable_caching": getattr(self.config, "enable_caching", True),
                "enable_parallel_processing": getattr(
                    self.config, "enable_parallel_processing", False
                ),
                "lazy_loading": getattr(self.config, "lazy_loading", False),
            },
        }

        return status

    def set_guidance_level(self, level: GuidanceLevel) -> None:
        """
        Set the guidance level for signal processing.

        Args:
            level: The new guidance level (NONE, WEAK, STRONG, FULL)
        """
        if not isinstance(level, GuidanceLevel):
            raise ValueError(
                f"Invalid guidance level: {level}. Must be a GuidanceLevel enum value."
            )

        self.guidance_level = level
        self.config.guidance_level = level
        self.logger.info(f"Guidance level set to: {level.value}")

    def get_performance_summary(self) -> PerformanceStats:
        """
        Get comprehensive performance summary from PerformanceTracker.

        Returns:
            Dictionary with performance metrics
        """
        return self.performance_tracker.get_performance_summary()

    def get_pattern_statistics(
        self, pattern_type: Optional[str] = None
    ) -> PatternStats:
        """
        Get pattern statistics from PatternStatistics component.

        Args:
            pattern_type: Specific pattern type, or None for all

        Returns:
            Dictionary with pattern statistics
        """
        return self.pattern_statistics.get_pattern_statistics(pattern_type)

    def get_cache_stats(self) -> CacheStats:
        """
        Get cache statistics from CacheManager.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache_manager.get_cache_stats()

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self.cache_manager.invalidate_cache()
        self.logger.info("Cache cleared")

    def reset_statistics(self) -> None:
        """
        Reset all performance and pattern statistics.
        """
        self.performance_tracker.reset_metrics()
        self.pattern_statistics.reset_statistics()
        self.logger.info("Statistics reset")

    def analyze_pattern_effectiveness(
        self, trading_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of different patterns based on trading results.

        Args:
            trading_results: List of trading result dictionaries containing signals and performance metrics.

        Returns:
            Dictionary containing pattern effectiveness analysis with win rates, profitability, and risk metrics.
        """
        if trading_results is None or not trading_results:
            return {
                "total_trades": 0,
                "patterns": {},
                "summary": "No trading results provided",
            }

        # Initialize pattern statistics
        pattern_stats = {}
        total_trades = len(trading_results)

        # Aggregate data by pattern
        for result in trading_results:
            signals = result.get("signals", [])
            profit = result.get("profit", 0)
            win_rate = result.get("win_rate", 0)
            sharpe_ratio = result.get("sharpe_ratio", 0)
            max_drawdown = result.get("max_drawdown", 0)

            for signal in signals:
                source_patterns = signal.get("source_patterns", [])
                for pattern in source_patterns:
                    if pattern not in pattern_stats:
                        pattern_stats[pattern] = {
                            "trades": [],
                            "total_profit": 0,
                            "wins": 0,
                            "total_win_rate": 0,
                            "total_sharpe": 0,
                            "total_drawdown": 0,
                            "count": 0,
                        }

                    pattern_stats[pattern]["trades"].append(profit)
                    pattern_stats[pattern]["total_profit"] += profit
                    if profit > 0:
                        pattern_stats[pattern]["wins"] += 1
                    pattern_stats[pattern]["total_win_rate"] += win_rate
                    pattern_stats[pattern]["total_sharpe"] += sharpe_ratio
                    pattern_stats[pattern]["total_drawdown"] += max_drawdown
                    pattern_stats[pattern]["count"] += 1

        # Calculate metrics
        analysis = {
            "total_trades": total_trades,
            "patterns": {},
            "summary": f"Analyzed {total_trades} trades across {len(pattern_stats)} patterns",
        }

        for pattern, stats in pattern_stats.items():
            count = stats["count"]
            if count > 0:
                avg_profit = stats["total_profit"] / count
                win_rate = stats["wins"] / count
                avg_win_rate = stats["total_win_rate"] / count
                avg_sharpe = stats["total_sharpe"] / count
                avg_drawdown = stats["total_drawdown"] / count

                analysis["patterns"][pattern] = {
                    "trade_count": count,
                    "average_profit": avg_profit,
                    "win_rate": win_rate,
                    "average_win_rate": avg_win_rate,
                    "average_sharpe_ratio": avg_sharpe,
                    "average_max_drawdown": avg_drawdown,
                }

        return analysis

    def analyze_pattern_effectiveness(self, trading_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze the effectiveness of different pattern recognizers.

        Returns:
            Dictionary containing pattern effectiveness analysis
        """
        enabled_patterns = []
        disabled_patterns = []
        pattern_stats = {}

        # Ensure recognizers are loaded for lazy loading mode
        self._ensure_recognizers_loaded()

        # Collect enabled/disabled patterns and their statistics
        pattern_groups = [
            'candlestick', 'fibonacci', 'gann', 'wave', 'harmonic',
            'oscillator', 'volume', 'bollinger', 'adx', 'granville',
            'heikin_ashi', 'dow_theory'
        ]

        for pattern_group in pattern_groups:
            attr_name = f'{pattern_group}_recognizers'
            recognizers = getattr(self, attr_name, [])

            # Check if pattern group is enabled
            enable_flag = getattr(self.config, f'enable_{pattern_group}_patterns', True)

            # Ensure recognizers is a list
            if not isinstance(recognizers, list):
                recognizers = []

            for recognizer in recognizers:
                pattern_name = getattr(recognizer, '__class__', type(recognizer)).__name__.replace('Recognizer', '').lower()

                if enable_flag:
                    enabled_patterns.append(pattern_name)
                else:
                    disabled_patterns.append(pattern_name)

                # Initialize pattern stats
                if pattern_name not in pattern_stats:
                    pattern_stats[pattern_name] = {
                        'signals_generated': 0,
                        'enabled': enable_flag
                    }

        # Count signals generated by each pattern from signal history
        for signal in self.signal_history:
            for pattern in signal.source_patterns:
                pattern_name = pattern.lower()
                if pattern_name in pattern_stats:
                    pattern_stats[pattern_name]['signals_generated'] += 1

        # If trading results provided, analyze correlations
        correlations = {}
        if trading_results:
            for pattern_name in pattern_stats.keys():
                pattern_trades = []
                for result in trading_results:
                    for signal in result.get("signals", []):
                        if pattern_name in [p.lower() for p in signal.get("source_patterns", [])]:
                            pattern_trades.append(result["profit"])

                if pattern_trades:
                    avg_profit = sum(pattern_trades) / len(pattern_trades)
                    win_rate = sum(1 for p in pattern_trades if p > 0) / len(pattern_trades)
                    correlations[pattern_name] = {
                        "average_profit": avg_profit,
                        "win_rate": win_rate,
                        "total_trades": len(pattern_trades)
                    }

        result = {
            'total_signals': len(self.signal_history),
            'enabled_patterns': enabled_patterns,
            'disabled_patterns': disabled_patterns,
            'pattern_stats': pattern_stats
        }

        if correlations:
            result['correlations'] = correlations

        return result

    def generate_validation_report(self) -> str:
        """
        Generate a validation report for the ActionSignalGuide.

        Returns:
            String containing the validation report
        """
        analysis = self.analyze_pattern_effectiveness()

        report_lines = []
        report_lines.append("ActionSignalGuide Validation Report")
        report_lines.append("=" * 40)
        report_lines.append("")

        report_lines.append(f"Total Signals Generated: {analysis['total_signals']}")
        report_lines.append(f"Enabled Pattern Groups: {len(analysis['enabled_patterns'])}")
        report_lines.append(f"Disabled Pattern Groups: {len(analysis['disabled_patterns'])}")
        report_lines.append("")

        report_lines.append("Pattern Statistics:")
        report_lines.append("-" * 20)

        for pattern, stats in analysis['pattern_stats'].items():
            status = "ENABLED" if stats['enabled'] else "DISABLED"
            signals = stats['signals_generated']
            report_lines.append(f"  {pattern} ({status}): {signals} signals")

        report_lines.append("")
        report_lines.append("Configuration:")
        report_lines.append(f"  Max signals per bar: {getattr(self.config, 'max_signals_per_bar', 'N/A')}")
        report_lines.append(f"  Caching enabled: {getattr(self.config, 'enable_caching', 'N/A')}")
        report_lines.append(f"  Parallel processing: {getattr(self.config, 'enable_parallel_processing', 'N/A')}")

        return "\n".join(report_lines)

    def analyze_sac_learning_correlation(
        self,
        sac_learning_logs: Optional[List[Dict[str, Any]]] = None,
        correlation_window: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze correlation between SAC learning performance and signal quality.

        Args:
            sac_learning_logs: SAC learning metrics (rewards, losses, etc.)
            correlation_window: Rolling window size for correlation analysis

        Returns:
            Dictionary containing correlation analysis results
        """
        return self.signal_performance_analyzer.analyze_sac_learning_correlation(
            sac_learning_logs, correlation_window
        )

    def calculate_signal_quality_score(
        self,
        signal_strength: float,
        signal_confidence: float,
        pattern_type: str,
        historical_success_rate: Optional[float] = None,
        consistency_score: Optional[float] = None
    ) -> float:
        """
        Calculate comprehensive signal quality score.

        Args:
            signal_strength: Signal strength (0-1)
            signal_confidence: Signal confidence (0-1)
            pattern_type: Type of pattern
            historical_success_rate: Historical success rate for this pattern
            consistency_score: Signal consistency score

        Returns:
            Quality score (0-1)
        """
        # Get historical success rate from pattern statistics if not provided
        if historical_success_rate is None:
            pattern_stats = self.pattern_statistics.get_pattern_statistics(pattern_type)
            historical_success_rate = pattern_stats.get('success_rate', 0.5)

        # Calculate consistency score if not provided
        if consistency_score is None:
            consistency_score = self._calculate_signal_consistency(pattern_type)

        return self.signal_performance_analyzer.calculate_signal_quality_score(
            signal_strength, signal_confidence, pattern_type,
            historical_success_rate, consistency_score
        )

    def generate_signal_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive signal performance report.

        Returns:
            Dictionary containing performance analysis and recommendations
        """
        return self.signal_performance_analyzer.generate_performance_report()

    def _calculate_signal_consistency(self, pattern_type: str) -> float:
        """
        Calculate signal consistency score for a pattern type.

        Args:
            pattern_type: Pattern type to analyze

        Returns:
            Consistency score (0-1)
        """
        pattern_stats = self.pattern_statistics.get_pattern_statistics(pattern_type)

        # Use variance in success rates as inverse consistency measure
        success_rates = pattern_stats.get('success_rate_history', [])
        if len(success_rates) < 2:
            return 0.5  # Default moderate consistency

        # Lower variance = higher consistency
        variance = np.var(success_rates) if success_rates else 0.25
        consistency = max(0.0, 1.0 - variance * 4)  # Scale variance to 0-1 range

        return consistency
