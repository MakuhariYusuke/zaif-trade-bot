"""
SignalFeatureGenerator Component.

Integrates Action Signal Guide signals into SAC observation space.
Provides real-time signal features for HeavyTradingEnv.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from ztb.utils.logging_utils import get_logger

from ..action_signal_guide import ActionSignalGuide
from ..types import SignalList


class SignalFeatureGenerator:
    """
    Generates signal-based features for SAC observation space.

    Integrates Action Signal Guide signals into the trading environment's
    observation space for enhanced decision making.
    """

    def __init__(self, signal_guide: ActionSignalGuide):
        """
        Initialize SignalFeatureGenerator.

        Args:
            signal_guide: Configured ActionSignalGuide instance
        """
        self.signal_guide = signal_guide
        self.logger = get_logger("ztb.trading.strategies.signal_feature_generator")

        # Feature dimensions
        self.signal_feature_dim = 10  # Number of signal-based features

        # Cache for performance
        self._last_signals: Optional[SignalList] = None
        self._last_features: Optional[np.ndarray] = None

    def generate_signal_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Generate signal-based features from market data.

        Args:
            market_data: Current market data (OHLCV, indicators)

        Returns:
            np.ndarray: Signal features array
        """
        try:
            # Get signals from Action Signal Guide
            signals = self.signal_guide.generate_signals(market_data, current_index=len(market_data)-1)

            # Cache signals for potential reuse
            self._last_signals = signals

            # Extract features from signals
            features = self._extract_features_from_signals(signals, market_data)

            # Cache features
            self._last_features = features

            return features

        except Exception as e:
            self.logger.error(f"Error generating signal features: {e}")
            # Return zero features on error
            return np.zeros(self.signal_feature_dim, dtype=np.float32)

    def _extract_features_from_signals(self, signals: SignalList, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features from signal list.

        Args:
            signals: List of ActionSignal objects
            market_data: Market data for context

        Returns:
            np.ndarray: Feature array
        """
        features = np.zeros(self.signal_feature_dim, dtype=np.float32)

        if not signals:
            return features

        # Feature 1: Average signal strength
        strengths = [s.strength for s in signals]
        features[0] = np.mean(strengths) if strengths else 0.0

        # Feature 2: Average signal confidence
        confidences = [s.confidence for s in signals]
        features[1] = np.mean(confidences) if confidences else 0.0

        # Feature 3: Net direction bias (-1 to 1)
        directions = [s.direction for s in signals]
        if directions:
            buy_signals = sum(1 for d in directions if d > 0)
            sell_signals = sum(1 for d in directions if d < 0)
            total_signals = len(directions)
            features[2] = (buy_signals - sell_signals) / total_signals if total_signals > 0 else 0.0

        # Feature 4: Signal diversity (number of different pattern types)
        pattern_types = set(s.pattern_type for s in signals)
        features[3] = len(pattern_types) / 10.0  # Normalize by max expected types

        # Feature 5: High confidence signal ratio
        high_conf_signals = sum(1 for s in signals if s.confidence > 0.8)
        features[4] = high_conf_signals / len(signals) if signals else 0.0

        # Feature 6: Trend alignment score
        features[5] = self._calculate_trend_alignment(signals, market_data)

        # Feature 7: Momentum alignment score
        features[6] = self._calculate_momentum_alignment(signals, market_data)

        # Feature 8: Volume confirmation score
        features[7] = self._calculate_volume_confirmation(signals, market_data)

        # Feature 9: Multi-timeframe consistency
        features[8] = self._calculate_multitimeframe_consistency(signals)

        # Feature 10: Signal freshness (time since last signal)
        features[9] = self._calculate_signal_freshness(signals)

        return features

    def _calculate_trend_alignment(self, signals: SignalList, market_data: pd.DataFrame) -> float:
        """Calculate how well signals align with current trend."""
        # Implementation for trend alignment
        return 0.0  # Placeholder

    def _calculate_momentum_alignment(self, signals: SignalList, market_data: pd.DataFrame) -> float:
        """Calculate momentum alignment score."""
        return 0.0  # Placeholder

    def _calculate_volume_confirmation(self, signals: SignalList, market_data: pd.DataFrame) -> float:
        """Calculate volume confirmation score."""
        return 0.0  # Placeholder

    def _calculate_multitimeframe_consistency(self, signals: SignalList) -> float:
        """Calculate multi-timeframe consistency."""
        return 0.0  # Placeholder

    def _calculate_signal_freshness(self, signals: SignalList) -> float:
        """Calculate signal freshness score."""
        return 0.0  # Placeholder

    def get_feature_names(self) -> List[str]:
        """Get names of generated features."""
        return [
            "avg_signal_strength",
            "avg_signal_confidence",
            "net_direction_bias",
            "signal_diversity",
            "high_conf_ratio",
            "trend_alignment",
            "momentum_alignment",
            "volume_confirmation",
            "multitimeframe_consistency",
            "signal_freshness"
        ]
