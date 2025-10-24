"""
Wave Counting Module

This module provides pattern recognition for wave counting analysis,
primarily based on Elliott Wave Theory including impulse waves, corrective waves,
and various wave patterns.
"""

import pandas as pd
from typing import Optional, List, Dict, NamedTuple
from enum import Enum

from ztb.trading.environment.constants import EPSILON

from .base import PatternRecognizer, SignalResult


class WaveType(Enum):
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"


class WaveDegree(Enum):
    GRAND_SUPERCYCLE = "grand_supercycle"
    SUPERCYCLE = "supercycle"
    CYCLE = "cycle"
    PRIMARY = "primary"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    MINUTE = "minute"
    MINUETTE = "minuette"
    SUBMINUETTE = "subminuette"


class WavePoint(NamedTuple):
    index: int
    price: float
    wave_label: str
    degree: WaveDegree


class WaveAnalyzer:
    """
    Utility class for wave counting and analysis.

    Provides methods to:
    - Find significant pivot points in price data for wave identification.
    - Identify Elliott Wave structures such as impulse and corrective patterns.
    """
    
    @staticmethod
    def find_pivot_points(data: pd.DataFrame, lookback: int = 20, 
                         min_distance: int = 3) -> List[WavePoint]:
        """Find significant pivot points in the data."""
        if len(data) < lookback:
            return []
            
        highs = data['high']
        lows = data['low']
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(lookback // 2, len(data) - lookback // 2):
            # Check for pivot high
            is_pivot_high = True
            for j in range(1, lookback // 2 + 1):
                if highs.iloc[i] <= highs.iloc[i - j] or highs.iloc[i] <= highs.iloc[i + j]:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                # Check minimum distance from previous pivot
                if not pivot_highs or (i - pivot_highs[-1].index) >= min_distance:
                    pivot_highs.append(WavePoint(i, highs.iloc[i], "PH", WaveDegree.MINOR))
            
            # Check for pivot low
            is_pivot_low = True
            for j in range(1, lookback // 2 + 1):
                if lows.iloc[i] >= lows.iloc[i - j] or lows.iloc[i] >= lows.iloc[i + j]:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                # Check minimum distance from previous pivot
                if not pivot_lows or (i - pivot_lows[-1].index) >= min_distance:
                    pivot_lows.append(WavePoint(i, lows.iloc[i], "PL", WaveDegree.MINOR))
        
        # Combine and sort by index
        all_pivots = pivot_highs + pivot_lows
        all_pivots.sort(key=lambda x: x.index)
        
        return all_pivots
    
    @staticmethod
    def identify_wave_structure(pivots: List[WavePoint]) -> Optional[Dict]:
        """Identify wave structure from pivot points."""
        if len(pivots) < 5:
            return None
            
        # Look for 5-wave impulse pattern
        # Wave 1: up, 2: down (correction), 3: up, 4: down (correction), 5: up
        
        # Find potential wave 1-5 sequence
        for i in range(len(pivots) - 4):
            w1, w2, w3, w4, w5 = pivots[i:i+5]
            
            # Basic impulse wave rules
            if (w1.price < w3.price > w5.price and  # Waves 1,3,5 trending up
                w2.price < w1.price and w4.price < w3.price and  # Corrections
                w3.price > w1.price and w5.price > w3.price):  # Progression
                
                # Check wave ratios (Fibonacci relationships)
                wave1_length = w1.price - min(w1.price, w2.price)
                wave3_length = w3.price - w2.price
                wave5_length = w5.price - w4.price
                
                # Wave 3 should be the longest
                if wave3_length > wave1_length and wave3_length > wave5_length:
                    # Check Fibonacci extensions
                    total_length = w5.price - w1.price
                    wave3_ratio = wave3_length / total_length
                    
                    if 0.5 < wave3_ratio < 0.8:  # Wave 3 typically 50-80% of total
                        return {
                            'type': WaveType.IMPULSE,
                            'degree': WaveDegree.MINOR,
                            'waves': [w1, w2, w3, w4, w5],
                            'direction': 1,  # Bullish impulse
                            'strength': 0.8,
                            'completion_index': w5.index
                        }
        
        # Look for corrective ABC pattern
        for i in range(len(pivots) - 2):
            a, b, c = pivots[i:i+3]
            
            # ABC correction: A down, B up (partial retracement), C down (beyond A)
            if (a.price > b.price and b.price < c.price and  # B is lower than both A and C
                c.price < a.price):  # C goes below A
                
                # Check Fibonacci retracement of B from A
                ab_range = a.price - b.price
                if ab_range == 0:
                    continue  # Skip this pattern, avoid division by zero
                bc_retracement = (b.price - c.price) / ab_range
                
                if 0.5 < bc_retracement < 0.8:  # 50-80% retracement
                    return {
                        'type': WaveType.CORRECTIVE,
                        'degree': WaveDegree.MINOR,
                        'waves': [a, b, c],
                        'direction': -1,  # Bearish correction
                        'strength': 0.7,
                        'completion_index': c.index
                    }
        
        return None


class ImpulseWaveRecognizer(PatternRecognizer):
    """Recognizes Elliott Wave impulse patterns (5-wave structures)."""
    
    def __init__(self, lookback_period: int = 50, min_pivot_distance: int = 3):
        self.lookback_period = lookback_period
        self.min_pivot_distance = min_pivot_distance
        self.wave_analyzer = WaveAnalyzer()
    
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize impulse wave patterns at the given index."""
        if index < self.lookback_period:
            return None
            
        # Get pivot points in the lookback period
        lookback_data = data.iloc[max(0, index - self.lookback_period):index+1]
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data, lookback=self.lookback_period // 2, 
            min_distance=self.min_pivot_distance
        )
        
        # Adjust pivot indices to global index
        offset = max(0, index - self.lookback_period)
        adjusted_pivots = [
            WavePoint(p.index + offset, p.price, p.wave_label, p.degree) 
            for p in pivots
        ]
        
        wave_structure = self.wave_analyzer.identify_wave_structure(adjusted_pivots)
        
        if wave_structure and wave_structure['type'] == WaveType.IMPULSE:
            # Check if we're at or near the completion of wave 5
            if abs(index - wave_structure['completion_index']) <= 2:  # Within 2 bars
                strength = wave_structure['strength']
                direction = wave_structure['direction']
                
                signal_type = "impulse_wave_completion"
                
                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    direction=direction,
                    description="Elliott Wave Impulse Pattern (5-wave structure)",
                    metadata={
                        "pattern": "impulse_wave",
                        "wave_type": wave_structure['type'].value,
                        "degree": wave_structure['degree'].value,
                        "wave_count": len(wave_structure['waves']),
                        "confidence": strength
                    }
                )
        
        return None


class CorrectiveWaveRecognizer(PatternRecognizer):
    """Recognizes Elliott Wave corrective patterns (ABC structures)."""
    
    def __init__(self, lookback_period: int = 40, min_pivot_distance: int = 3):
        self.lookback_period = lookback_period
        self.min_pivot_distance = min_pivot_distance
        self.wave_analyzer = WaveAnalyzer()
    
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize corrective wave patterns at the given index."""
        if index < self.lookback_period:
            return None
            
        # Get pivot points in the lookback period
        lookback_data = data.iloc[max(0, index - self.lookback_period):index+1]
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data, lookback=self.lookback_period // 2, 
            min_distance=self.min_pivot_distance
        )
        
        # Adjust pivot indices to global index
        offset = max(0, index - self.lookback_period)
        adjusted_pivots = [
            WavePoint(p.index + offset, p.price, p.wave_label, p.degree) 
            for p in pivots
        ]
        
        wave_structure = self.wave_analyzer.identify_wave_structure(adjusted_pivots)
        
        if wave_structure and wave_structure['type'] == WaveType.CORRECTIVE:
            # Check if we're at or near the completion of wave C
            if abs(index - wave_structure['completion_index']) <= 2:  # Within 2 bars
                strength = wave_structure['strength']
                direction = wave_structure['direction']
                
                signal_type = "corrective_wave_completion"
                
                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    direction=direction,
                    description="Elliott Wave Corrective Pattern (ABC structure)",
                    metadata={
                        "pattern": "corrective_wave",
                        "wave_type": wave_structure['type'].value,
                        "degree": wave_structure['degree'].value,
                        "wave_count": len(wave_structure['waves']),
                        "confidence": strength
                    }
                )
        
        return None


class WaveExtensionRecognizer(PatternRecognizer):
    """Recognizes wave extensions and truncations."""
    
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.wave_analyzer = WaveAnalyzer()
    
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize wave extensions or truncations at the given index."""
        if index < self.lookback_period:
            return None
            
        lookback_data = data.iloc[max(0, index - self.lookback_period):index+1]
        pivots = self.wave_analyzer.find_pivot_points(
            lookback_data, lookback=self.lookback_period // 3
        )
        
        if len(pivots) < 3:
            return None
            
        # Look for extended waves (wave 3 much longer than others)
        recent_pivots = pivots[-5:] if len(pivots) >= 5 else pivots
        
        if len(recent_pivots) >= 3:
            # Calculate wave lengths
            wave_lengths = []
            for i in range(len(recent_pivots) - 1):
                length = abs(recent_pivots[i+1].price - recent_pivots[i].price)
                wave_lengths.append(length)
            
            if len(wave_lengths) >= 3:
                # Check for wave 3 extension (much longer than waves 1 and 5)
                if (wave_lengths[1] > wave_lengths[0] * 1.5 and 
                    wave_lengths[1] > wave_lengths[2] * 1.5):
                    strength = min(0.85, 0.6 + (wave_lengths[1] / max(wave_lengths[0], wave_lengths[2], EPSILON) - 1.5) * 0.1)
                    direction = 1 if recent_pivots[-1].price > recent_pivots[0].price else -1
                    strength = min(0.85, 0.6 + (wave_lengths[1] / max(wave_lengths[0], wave_lengths[2]) - 1.5) * 0.1)
                    
                    return SignalResult(
                        signal_type="wave_extension",
                        strength=strength,
                        direction=direction,
                        description="Elliott Wave Extension (extended wave 3)",
                        metadata={
                            "pattern": "wave_extension",
                            "extension_ratio": wave_lengths[1] / max(wave_lengths[0], wave_lengths[2]),
                            "confidence": strength
                        }
                    )
        
        # Look for truncated waves (wave 5 fails to exceed wave 3)
        if len(recent_pivots) >= 5:
            w1, w2, w3, w4, w5 = recent_pivots[-5:]
            
            # Check for truncation: wave 5 doesn't exceed wave 3 high (in bullish case)
            if w3.price > w1.price and w5.price < w3.price:
                strength = 0.7
                
                return SignalResult(
                    signal_type="wave_truncation",
                    strength=strength,
                    direction=-1,  # Bearish signal (failure)
                    description="Elliott Wave Truncation (wave 5 fails to exceed wave 3)",
                    metadata={
                        "pattern": "wave_truncation",
                        "wave3_high": w3.price,
                        "wave5_high": w5.price,
                        "confidence": strength
                    }
                )
        
        return None