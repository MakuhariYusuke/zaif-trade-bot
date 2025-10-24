"""
Fibonacci Pattern Recognition Module

This module provides pattern recognition for Fibonacci-based technical analysis,
including retracements, extensions, projections, and Fibonacci-based patterns.
"""

import pandas as pd
from typing import Optional, Dict

from .base import PatternRecognizer, SignalResult


class FibonacciAnalyzer:
    """Utility class for Fibonacci calculations and analysis."""
    
    # Standard Fibonacci ratios
    RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSION_LEVELS = [0.618, 1.0, 1.236, 1.382, 1.618, 2.0, 2.618]
    PROJECTION_LEVELS = [0.618, 1.0, 1.236, 1.382, 1.618, 2.0, 2.618]
    
    @staticmethod
    def calculate_retracement_levels(high: float, low: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels between high and low."""
        diff = high - low
        levels = {}
        for ratio in FibonacciAnalyzer.RETRACEMENT_LEVELS:
            levels[ratio] = low + diff * ratio
        return levels
    
    @staticmethod
    def calculate_extension_levels(high: float, low: float, direction: int = 1) -> Dict[float, float]:
        """Calculate Fibonacci extension levels."""
        diff = high - low
        levels = {}
        for ratio in FibonacciAnalyzer.EXTENSION_LEVELS:
            if direction == 1:  # Bullish extension from low
                levels[ratio] = low + diff * ratio
            else:  # Bearish extension from high
                levels[ratio] = high - diff * ratio
        return levels
    
    @staticmethod
    def find_fibonacci_retracement(data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Dict]:
        """Find if price has retraced to a Fibonacci level within a swing."""
        if start_idx >= end_idx or end_idx >= len(data):
            return None
            
        swing_high = data.iloc[start_idx:end_idx+1]['high'].max()
        swing_low = data.iloc[start_idx:end_idx+1]['low'].min()
        
        # Find the retracement point (current close)
        current_close = data.iloc[end_idx]['close']
        
        # Calculate retracement ratio
        total_range = swing_high - swing_low
        if total_range == 0:
            return None
            
        retracement_ratio = (swing_high - current_close) / total_range
        
        # Find closest Fibonacci level
        closest_level = min(FibonacciAnalyzer.RETRACEMENT_LEVELS, 
                          key=lambda x: abs(x - retracement_ratio))
        
        tolerance = 0.02  # 2% tolerance
        if abs(retracement_ratio - closest_level) <= tolerance:
            return {
                'level': closest_level,
                'actual_ratio': retracement_ratio,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'current_price': current_close,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        
        return None


class FibonacciRetracementRecognizer(PatternRecognizer):
    """Recognizes Fibonacci retracement levels in price action."""
    
    def __init__(self, min_swing_length: int = 5, max_swing_length: int = 50):
        self.min_swing_length = min_swing_length
        self.max_swing_length = max_swing_length
        self.fib_analyzer = FibonacciAnalyzer()
    
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Fibonacci retracement at the given index."""
        if index < self.max_swing_length:
            return None
            
        # Look for swing points within the range
        for swing_length in range(self.min_swing_length, min(self.max_swing_length, index + 1)):
            start_idx = index - swing_length
            
            fib_retracement = self.fib_analyzer.find_fibonacci_retracement(
                data, start_idx, index
            )
            
            if fib_retracement:
                level = fib_retracement['level']
                direction = 1 if data.iloc[index]['close'] > data.iloc[start_idx]['close'] else -1
                
                # Strength based on how close to the exact level and level significance
                level_strength = {0.236: 0.6, 0.382: 0.7, 0.5: 0.8, 0.618: 0.9, 0.786: 0.75}
                base_strength = level_strength.get(level, 0.6)
                
                # Adjust for accuracy
                accuracy_penalty = abs(fib_retracement['actual_ratio'] - level) * 5
                strength = max(0.4, base_strength - accuracy_penalty)
                
                signal_type = "fib_retracement_support" if direction == 1 else "fib_retracement_resistance"
                
                return SignalResult(
                    signal_type=signal_type,
                    strength=strength,
                    direction=direction,
                    description=f"Fibonacci Retracement at {level:.3f} level",
                    metadata={
                        "pattern": "fibonacci_retracement",
                        "level": level,
                        "swing_length": swing_length,
                        "confidence": strength
                    }
                )
        
        return None


class FibonacciExtensionRecognizer(PatternRecognizer):
    """Recognizes Fibonacci extension targets."""
    
    def __init__(self, min_swing_length: int = 5, max_swing_length: int = 50):
        self.min_swing_length = min_swing_length
        self.max_swing_length = max_swing_length
        self.fib_analyzer = FibonacciAnalyzer()
    
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Fibonacci extension targets at the given index."""
        if index < self.max_swing_length:
            return None
            
        current_price = data.iloc[index]['close']
        
        # Look for completed swings to project extensions
        for swing_length in range(self.min_swing_length, min(self.max_swing_length, index + 1)):
            start_idx = index - swing_length
            
            swing_high = data.iloc[start_idx:index+1]['high'].max()
            swing_low = data.iloc[start_idx:index+1]['low'].min()
            
            # Determine trend direction
            trend_direction = 1 if current_price > (swing_high + swing_low) / 2 else -1
            
            # Calculate extension levels
            extension_levels = self.fib_analyzer.calculate_extension_levels(
                swing_high, swing_low, trend_direction
            )
            
            # Check if current price is near an extension level
            for ratio, level in extension_levels.items():
                tolerance = abs(swing_high - swing_low) * 0.02  # 2% tolerance
                
                if abs(current_price - level) <= tolerance:
                    strength = 0.7 + (ratio - 1.0) * 0.1  # Higher ratios get higher strength
                    strength = min(0.9, max(0.5, strength))
                    
                    signal_type = "fib_extension_target" if trend_direction == 1 else "fib_extension_target_bearish"
                    
                    return SignalResult(
                        signal_type=signal_type,
                        strength=strength,
                        direction=trend_direction,
                        description=f"Fibonacci Extension target at {ratio:.3f} level",
                        metadata={
                            "pattern": "fibonacci_extension",
                            "level": ratio,
                            "target_price": level,
                            "swing_length": swing_length,
                            "confidence": strength
                        }
                    )
        
        return None


class FibonacciProjectionRecognizer(PatternRecognizer):
    """Recognizes Fibonacci price projections from multiple swings."""
    
    def __init__(self, min_swing_length: int = 3, max_lookback: int = 20):
        self.min_swing_length = min_swing_length
        self.max_lookback = max_lookback
        self.fib_analyzer = FibonacciAnalyzer()
    
    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """Recognize Fibonacci projections at the given index."""
        if index < self.max_lookback:
            return None
            
        current_price = data.iloc[index]['close']
        
        # Look for two swings to create a projection
        for first_swing_end in range(index - self.min_swing_length, index - 2*self.min_swing_length, -1):
            if first_swing_end < 0:
                break
                
            first_swing_start = first_swing_end - self.min_swing_length
            if first_swing_start < 0:
                continue
                
            # First swing
            first_high = data.iloc[first_swing_start:first_swing_end+1]['high'].max()
            first_low = data.iloc[first_swing_start:first_swing_end+1]['low'].min()
            first_range = first_high - first_low
            
            # Second swing (from first_swing_end to current)
            second_high = data.iloc[first_swing_end:index+1]['high'].max()
            second_low = data.iloc[first_swing_end:index+1]['low'].min()
            
            # Determine projection direction
            if second_high > first_high:  # Bullish projection
                projection_base = first_low
                projection_range = second_high - first_low
            elif second_low < first_low:  # Bearish projection
                projection_base = first_high
                projection_range = first_high - second_low
            else:
                continue
            
            # Calculate projection levels
            for ratio in FibonacciAnalyzer.PROJECTION_LEVELS:
                projected_price = projection_base + projection_range * ratio
                
                tolerance = first_range * 0.03  # 3% tolerance
                
                if abs(current_price - projected_price) <= tolerance:
                    direction = 1 if projected_price > projection_base else -1
                    strength = 0.6 + (ratio - 1.0) * 0.15
                    strength = min(0.85, max(0.5, strength))
                    
                    signal_type = "fib_projection_target"
                    
                    return SignalResult(
                        signal_type=signal_type,
                        strength=strength,
                        direction=direction,
                        description=f"Fibonacci Projection at {ratio:.3f} level",
                        metadata={
                            "pattern": "fibonacci_projection",
                            "level": ratio,
                            "projected_price": projected_price,
                            "confidence": strength
                        }
                    )
        
        return None