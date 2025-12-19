"""
Unit tests for BasicRegimeDetector.
"""

import numpy as np

from ztb.analysis.regime.basic_regime_detector import MarketRegimeDetector


class TestMarketRegimeDetector:
    """Test MarketRegimeDetector functionality."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MarketRegimeDetector()

        assert detector.regime_detection_window == 20
        assert detector.adaptation_frequency == 10
        assert detector.current_regime == "sideways"  # Default regime

    def test_price_update_and_regime_detection(self):
        """Test price updates and regime detection."""
        detector = MarketRegimeDetector(regime_detection_window=10)

        # Detect regime with price data
        regime = detector.detect_regime(current_price=110.0, step=10)
        assert regime in ['bull', 'bear', 'sideways', 'volatile']

        # Detect again with different price
        regime2 = detector.detect_regime(current_price=105.0, step=20)
        assert regime2 in ['bull', 'bear', 'sideways', 'volatile']

    def test_regime_statistics(self):
        """Test regime statistics generation."""
        detector = MarketRegimeDetector()

        # Detect regimes multiple times
        for i in range(5):
            price = 100 + np.sin(i * 0.1) * 5
            detector.detect_regime(current_price=price, step=i*10)

        # Check that regime counts are tracked
        assert len(detector.regime_counts) > 0
        assert detector.total_steps_tracked > 0
        assert sum(detector.regime_counts.values()) == detector.total_steps_tracked

    def test_multiple_regime_detections(self):
        """Test multiple regime detections maintain state."""
        detector = MarketRegimeDetector()

        # Detect regimes multiple times
        for i in range(3):
            price = 100.0 + i * 2
            regime = detector.detect_regime(current_price=price, step=i*10)
            assert regime in ['bull', 'bear', 'sideways', 'volatile']
            assert len(detector.price_history) > 0