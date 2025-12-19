"""
Signal Quality Scorer

テクニカル指標と市場条件に基づく決定論的信号品質スコアリング
確率的アプローチから決定論的アプローチへの移行

Features:
- Deterministic scoring (0-100) based on technical indicators
- Market condition awareness
- Position context consideration
- Signal frequency optimization for scalping (20-50 signals/day)
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ztb.trading.signal.technical_indicators import TechnicalIndicators
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SignalQualityScorer:
    """
    Deterministic signal quality scorer using technical indicators

    Replaces probabilistic signal guidance with score-based approach
    for higher frequency and accuracy
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize signal quality scorer

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.technical_indicators = TechnicalIndicators()

        # Scoring weights
        self.weights = self.config.get('weights', {
            'rsi': 0.4,    # Increased RSI weight for stronger SELL signals
            'macd': 0.2,   # Reduced MACD weight
            'bollinger': 0.2,  # Reduced Bollinger weight
            'atr': 0.1,   # Reduced ATR weight
            'trend': 0.1  # Reduced Trend weight
        })

        # Thresholds for signal generation (adjusted for 20-50 signals/day target)
        self.buy_threshold = self.config.get('buy_threshold', 85)  # Higher threshold for BUY
        self.sell_threshold = self.config.get('sell_threshold', 5)  # Lower threshold for SELL to allow RSI to drive signals
        self.hold_threshold = self.config.get('hold_threshold', 45)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'weights': {
                'rsi': 0.4,    # Increased RSI weight for stronger SELL signals
                'macd': 0.2,   # Reduced MACD weight
                'bollinger': 0.2,  # Reduced Bollinger weight
                'atr': 0.1,   # Reduced ATR weight
                'trend': 0.1  # Reduced Trend weight
            },
            'buy_threshold': 85,  # Higher threshold for BUY signals
            'sell_threshold': 5,  # Lower threshold for SELL signals to allow RSI to drive signals
            'hold_threshold': 45,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'atr_period': 14,
            'trend_window': 10
        }

    def calculate_signal_quality(self, df: pd.DataFrame,
                                continuous_action: float,
                                portfolio: Dict[str, Any]) -> Tuple[int, float]:
        """
        Calculate signal quality score and determine action

        Args:
            df: Market data DataFrame with OHLCV
            continuous_action: Raw continuous action from model (-1 to 1)
            portfolio: Current portfolio state

        Returns:
            Tuple of (discrete_action, quality_score)
        """
        try:
            # Get technical indicators
            tech_signals = self.technical_indicators.get_technical_signals(df)

            # Check for strong oversold RSI condition (force SELL signal)
            rsi = tech_signals.get('rsi', 50.0)
            if rsi < 40:
                logger.debug(f"Strong oversold RSI detected: {rsi:.1f} - Forcing SELL signal")
                return -1, 0.0  # Force SELL signal for strongly oversold conditions

            # Calculate component scores
            rsi_score = self._calculate_rsi_score(tech_signals)
            macd_score = self._calculate_macd_score(tech_signals)
            bollinger_score = self._calculate_bollinger_score(tech_signals, df)
            atr_score = self._calculate_atr_score(tech_signals)
            trend_score = self._calculate_trend_score(df)

            # Combine scores with weights
            total_score = (
                rsi_score * self.weights['rsi'] +
                macd_score * self.weights['macd'] +
                bollinger_score * self.weights['bollinger'] +
                atr_score * self.weights['atr'] +
                trend_score * self.weights['trend']
            )

            # Apply position context adjustments
            total_score = self._apply_position_adjustments(total_score, portfolio)

            # Apply continuous action influence
            final_score = self._blend_continuous_action(total_score, continuous_action)

            # Determine discrete action
            discrete_action = self._score_to_action(final_score)

            logger.debug(f"Signal scores - RSI: {rsi:.1f}, MACD: {macd_score:.1f}, "
                        f"BB: {bollinger_score:.1f}, ATR: {atr_score:.1f}, "
                        f"Trend: {trend_score:.1f}, Total: {final_score:.1f}, Action: {discrete_action}")

            return discrete_action, final_score

        except Exception as e:
            logger.error(f"Error calculating signal quality: {e}")
            # Fallback to continuous action conversion
            return self._fallback_action(continuous_action), 50.0

    def _calculate_rsi_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate RSI-based score (0-100)"""
        rsi = tech_signals.get('rsi', 50.0)

        if rsi <= 30:
            # Oversold - bearish signal (SELL) - stronger signal when more oversold
            score = 30 - (rsi / 30) * 30  # 0-30, lower RSI = lower score
            return max(0, min(30, score))  # Cap at 0-30 for SELL signals
        elif rsi >= 70:
            # Overbought - bullish signal (BUY) - stronger signal when more overbought
            score = 70 + ((rsi - 70) / 30) * 30  # 70-100, higher RSI = higher score
            return max(70, min(100, score))  # Cap at 70-100 for BUY signals
        else:
            # Neutral zone - scale linearly
            return 30 + ((rsi - 30) / 40) * 40  # 30-70

    def _calculate_macd_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate MACD-based score (0-100)"""
        macd_line = tech_signals.get('macd_line', 0.0)
        signal_line = tech_signals.get('macd_signal', 0.0)
        histogram = tech_signals.get('macd_histogram', 0.0)

        # MACD crossover signals
        if macd_line > signal_line and histogram > 0:
            # Bullish crossover
            score = 75 + min(histogram * 100, 25)  # 75-100
        elif macd_line < signal_line and histogram < 0:
            # Bearish crossover
            score = 25 - min(abs(histogram) * 100, 25)  # 0-25
        else:
            # No clear signal
            score = 50

        return max(0, min(100, score))

    def _calculate_bollinger_score(self, tech_signals: Dict[str, float], df: pd.DataFrame) -> float:
        """Calculate Bollinger Bands-based score (0-100)"""
        bb_position = tech_signals.get('bb_position', 0.5)

        if len(df) == 0:
            return 50.0

        current_price = df['close'].iloc[-1]

        # Bollinger Band position scoring
        if bb_position <= 0.1:
            # Near lower band - potential bounce (bullish)
            return 80
        elif bb_position >= 0.9:
            # Near upper band - potential reversal (bearish)
            return 20
        elif bb_position <= 0.3:
            # Lower half - moderately bullish
            return 65
        elif bb_position >= 0.7:
            # Upper half - moderately bearish
            return 35
        else:
            # Middle - neutral
            return 50

    def _calculate_atr_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate ATR-based score (0-100)"""
        atr = tech_signals.get('atr', 0.0)

        if atr <= 0:
            return 50.0

        # Higher ATR indicates higher volatility
        # Scale ATR to 0-100 score (higher volatility = higher score for trading opportunities)
        # Assuming typical ATR range, normalize to 0-100
        normalized_atr = min(atr * 1000, 100)  # Scale factor may need adjustment

        return normalized_atr

    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend-based score (0-100)"""
        if len(df) < self.config['trend_window']:
            return 50.0

        recent_prices = df['close'].tail(self.config['trend_window']).values
        if len(recent_prices) < 2:
            return 50.0

        # Simple linear trend
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]

        # Normalize slope to score
        # Positive slope = bullish, negative slope = bearish
        if slope > 0:
            score = 50 + min(slope * 1000, 50)  # 50-100
        else:
            score = 50 + max(slope * 1000, -50)  # 0-50

        return max(0, min(100, score))

    def _apply_position_adjustments(self, score: float, portfolio: Dict[str, Any]) -> float:
        """Apply position context adjustments to score"""
        # Get position information
        btc_balance = portfolio.get('btc_balance', 0.0)
        jpy_balance = portfolio.get('jpy_balance', 0.0)
        current_price = portfolio.get('current_price', 0.0)

        if current_price <= 0:
            return score

        # Calculate position ratio
        btc_value = btc_balance * current_price
        total_value = btc_value + jpy_balance
        position_ratio = btc_value / total_value if total_value > 0 else 0.0

        # Adjust score based on position
        if position_ratio > 0.8:
            # Overexposed - reduce buy signals, increase sell signals
            if score > 50:
                score = score * 0.8  # Reduce bullish signals
            else:
                score = score * 1.2  # Increase bearish signals
        elif position_ratio < 0.2:
            # Underexposed - increase buy signals, reduce sell signals
            if score > 50:
                score = score * 1.2  # Increase bullish signals
            else:
                score = score * 0.8  # Reduce bearish signals

        return max(0, min(100, score))

    def _blend_continuous_action(self, quality_score: float, continuous_action: float) -> float:
        """Blend quality score with continuous action"""
        # Convert continuous action (-1, 1) to score (0, 100)
        action_score = (continuous_action + 1) * 50

        # Blend with quality score (80% quality, 20% action for more deterministic behavior)
        blended_score = quality_score * 0.8 + action_score * 0.2

        return max(0, min(100, blended_score))

    def _score_to_action(self, score: float) -> int:
        """Convert score to discrete action"""
        if score >= self.buy_threshold:
            return 1  # BUY
        elif score <= self.sell_threshold:
            return -1  # SELL
        else:
            return 0  # HOLD

