"""
Action Signal Guide - Main Implementation

This module provides the main ActionSignalGuide class that integrates all pattern
recognition systems for classical technical analysis signals in the SAC RL system.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .pattern_recognition.base import SignalResult
from .pattern_recognition.candlestick_patterns import (
    SakataFiveMethodsRecognizer,
    MorningStarRecognizer,
    EveningStarRecognizer,
    HammerRecognizer,
    HangingManRecognizer
)
from .pattern_recognition.fibonacci_patterns import (
    FibonacciRetracementRecognizer,
    FibonacciExtensionRecognizer,
    FibonacciProjectionRecognizer
)
from .pattern_recognition.gann_analysis import (
    GannAngleRecognizer,
    GannSquareRecognizer,
    GannTimeClusterRecognizer
)
from .pattern_recognition.wave_counting import (
    ImpulseWaveRecognizer,
    CorrectiveWaveRecognizer,
    WaveExtensionRecognizer
)
from .pattern_recognition.harmonic_patterns import (
    GartleyRecognizer,
    ButterflyRecognizer,
    BatRecognizer,
    CrabRecognizer
)
from ztb.trading.constants import ACTION_HOLD, ACTION_BUY, ACTION_SELL


class GuidanceLevel(Enum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"
    FULL = "full"


@dataclass
class ActionSignal:
    """Represents a complete action signal with all relevant information."""
    timestamp: pd.Timestamp
    direction: int  # 1 for buy, -1 for sell, 0 for hold
    strength: float  # 0.0 to 1.0
    signal_type: str
    description: str
    metadata: Dict[str, Any]
    source_patterns: List[str]  # List of pattern names that contributed


class ActionSignalGuide:
    """
    Main class for generating classical technical analysis signals.
    
    This class integrates multiple pattern recognition systems to provide
    comprehensive technical analysis signals for the SAC RL training system.
    """
    
    def __init__(self, guidance_level: GuidanceLevel = GuidanceLevel.STRONG) -> None:
        self.guidance_level = guidance_level
        
        # Initialize all pattern recognizers
        self._initialize_recognizers()
        
        # Signal history for context
        self.signal_history: List[ActionSignal] = []
        
        # Configuration
        self.min_signal_strength = 0.5
        self.max_signals_per_bar = 3
    
    def _initialize_recognizers(self) -> None:
        """Initialize all pattern recognition systems."""
        # Candlestick patterns
        self.candlestick_recognizers = [
            SakataFiveMethodsRecognizer(),
            MorningStarRecognizer(),
            EveningStarRecognizer(),
            HammerRecognizer(),
            HangingManRecognizer()
        ]
        
        # Fibonacci patterns
        self.fibonacci_recognizers = [
            FibonacciRetracementRecognizer(),
            FibonacciExtensionRecognizer(),
            FibonacciProjectionRecognizer()
        ]
        
        # Gann analysis
        self.gann_recognizers = [
            GannAngleRecognizer(),
            GannSquareRecognizer(),
            GannTimeClusterRecognizer()
        ]
        
        # Wave counting
        self.wave_recognizers = [
            ImpulseWaveRecognizer(),
            CorrectiveWaveRecognizer(),
            WaveExtensionRecognizer()
        ]
        
        # Harmonic patterns
        self.harmonic_recognizers = [
            GartleyRecognizer(),
            ButterflyRecognizer(),
            BatRecognizer(),
            CrabRecognizer()
        ]
        
        # Combine all recognizers
        self.all_recognizers = (
            self.candlestick_recognizers +
            self.fibonacci_recognizers +
            self.gann_recognizers +
            self.wave_recognizers +
            self.harmonic_recognizers
        )
    
    def generate_signals(self, data: pd.DataFrame, current_index: int) -> List[ActionSignal]:
        """
        Generate action signals for the current market data.
        
        Args:
            data: OHLCV DataFrame
            current_index: Current bar index to analyze
            
        Returns:
            List of ActionSignal objects
        """
        if current_index >= len(data):
            return []
            
        current_bar = data.iloc[current_index]
        signals = []
        
        # Collect signals from all recognizers
        for recognizer in self.all_recognizers:
            try:
                signal_result = recognizer.recognize(data, current_index)
                if signal_result and signal_result.strength >= self.min_signal_strength:
                    action_signal = ActionSignal(
                        timestamp=current_bar.name if isinstance(current_bar, pd.Series) and hasattr(current_bar, 'name') and current_bar.name is not None else pd.Timestamp.now(),
                        direction=signal_result.direction,
                        strength=self._adjust_strength_by_guidance(signal_result.strength),
                        signal_type=signal_result.signal_type,
                        description=signal_result.description,
                        metadata=signal_result.metadata,
                        source_patterns=[signal_result.signal_type]
                    )
                    signals.append(action_signal)
            except Exception as e:
                # Log error but continue with other recognizers
                print(f"Error in {recognizer.__class__.__name__}: {e}")
                continue
        
        # Filter and prioritize signals
        signals = self._filter_and_prioritize_signals(signals)
        
        # Store in history
        self.signal_history.extend(signals)
        
        return signals
    
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
            return base_strength
    
    def _filter_and_prioritize_signals(self, signals: List[ActionSignal]) -> List[ActionSignal]:
        """Filter and prioritize signals to avoid conflicts and redundancy."""
        if not signals:
            return signals
            
        # Sort by strength (highest first)
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        # Limit number of signals per bar
        filtered_signals = signals[:self.max_signals_per_bar]
        
        # Check for conflicting signals and resolve
        buy_signals = [s for s in filtered_signals if s.direction == 1]
        sell_signals = [s for s in filtered_signals if s.direction == -1]
        
        # If we have both buy and sell signals, keep only the stronger ones
        if buy_signals and sell_signals:
            # Compare strongest buy vs strongest sell
            strongest_buy = max(buy_signals, key=lambda x: x.strength)
            strongest_sell = max(sell_signals, key=lambda x: x.strength)
            
            if strongest_buy.strength > strongest_sell.strength:
                filtered_signals = [s for s in filtered_signals if s.direction != -1]
            elif strongest_sell.strength > strongest_buy.strength:
                filtered_signals = [s for s in filtered_signals if s.direction != 1]
            else:
                # Equal strength - keep both but reduce their strength
                for s in filtered_signals:
                    s.strength *= 0.8
        
        return filtered_signals
    
    def get_consolidated_signal(self, signals: List[ActionSignal]) -> Optional[ActionSignal]:
        """
        Consolidate multiple signals into a single action recommendation.
        
        Args:
            signals: List of individual signals
            
        Returns:
            Consolidated ActionSignal or None if no clear direction
        """
        if not signals:
            return None
            
        # Calculate weighted direction
        total_weight = sum(s.strength for s in signals)
        if total_weight == 0:
            return None
            
        weighted_direction = sum(s.direction * s.strength for s in signals) / total_weight
        
        # Determine final direction
        if abs(weighted_direction) < 0.3:
            final_direction = ACTION_HOLD  # Hold
            final_strength = 0.5
        else:
            final_direction = ACTION_BUY if weighted_direction > 0 else ACTION_SELL
            final_strength = min(1.0, abs(weighted_direction))
        
        # Combine metadata
        all_patterns = []
        combined_metadata = {}
        descriptions = []
        
        for signal in signals:
            all_patterns.extend(signal.source_patterns)
            descriptions.append(signal.description)
            combined_metadata.update(signal.metadata)
        
        return ActionSignal(
            timestamp=signals[0].timestamp,
            direction=final_direction,
            strength=final_strength,
            signal_type="consolidated_signal",
            description=" | ".join(descriptions[:3]),  # Limit to top 3 descriptions
            metadata={
                **combined_metadata,
                "signal_count": len(signals),
                "avg_strength": total_weight / len(signals)
            },
            source_patterns=list(set(all_patterns))
        )
    
    def get_signal_history(self, limit: int = 100) -> List[ActionSignal]:
        """Get recent signal history."""
        return self.signal_history[-limit:] if limit else self.signal_history
    
    def set_guidance_level(self, level: GuidanceLevel) -> None:
        """Update the guidance level for signal generation."""
        self.guidance_level = level
    
    def get_recognizer_status(self) -> Dict[str, int]:
        """Get status information about all recognizers."""
        status = {
            'candlestick_recognizers': len(self.candlestick_recognizers),
            'fibonacci_recognizers': len(self.fibonacci_recognizers),
            'gann_recognizers': len(self.gann_recognizers),
            'wave_recognizers': len(self.wave_recognizers),
            'harmonic_recognizers': len(self.harmonic_recognizers),
            'total_recognizers': len(self.all_recognizers)
        }
        return status
    
    def reset_history(self) -> None:
        """Clear signal history."""
        self.signal_history.clear()