#!/usr/bin/env python3
"""
Market Regime Classification System for SAC v428.

This module provides sophisticated market condition analysis including:
- Trend strength and direction classification
- Volatility level assessment
- Market regime detection (bull/bear/sideways)
- Performance analysis by market conditions
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.analysis.market_regime_types import MarketRegime
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VolatilityLevel(Enum):
    """Volatility level classifications."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MarketCondition:
    """Market condition classification result."""

    regime: MarketRegime
    volatility: VolatilityLevel
    trend_strength: float
    trend_direction: float  # -1 (bear) to +1 (bull)
    confidence: float
    timestamp: datetime
    features: Dict[str, float]


class MarketRegimeClassifier:
    """
    Advanced market regime classification system.

    Classifies market conditions based on multiple technical indicators:
    - Trend analysis (SMA, EMA, MACD)
    - Volatility measures (ATR, Bollinger Bands)
    - Momentum indicators (RSI, Stochastic)
    - Volume analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market regime classifier.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "trend_window": 20,
            "volatility_window": 20,
            "momentum_window": 14,
            "regime_threshold": 0.1,
            "volatility_percentiles": [0.2, 0.4, 0.6, 0.8],
            "min_periods": 50,
        }

    def classify_market_conditions(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        price_column: str = "close",
        volume_column: str = "volume",
    ) -> List[MarketCondition]:
        """
        Classify market conditions for the entire dataset.

        Args:
            data: Market data DataFrame
            timestamp_column: Name of timestamp column
            price_column: Name of price column
            volume_column: Name of volume column

        Returns:
            List of MarketCondition objects
        """
        if len(data) < self.config["min_periods"]:
            self.logger.warning(
                f"Insufficient data: {len(data)} < {self.config['min_periods']}"
            )
            return []

        # Ensure data is sorted by timestamp
        data = data.sort_values(timestamp_column).copy()

        # Calculate technical indicators
        features_df = self._calculate_technical_features(
            data, price_column, volume_column
        )

        # Classify each period
        conditions = []
        for idx, row in features_df.iterrows():
            try:
                # Skip rows with NaN values in critical indicators
                if pd.isna(row.get("trend_direction", 0)) or pd.isna(
                    row.get("volatility", 0)
                ):
                    continue

                condition = self._classify_single_period(row)
                condition.timestamp = pd.to_datetime(row[timestamp_column])
                conditions.append(condition)
            except Exception as e:
                self.logger.warning(f"Classification failed for row {idx}: {e}")
                continue

        self.logger.info(f"Classified {len(conditions)} market conditions")
        return conditions

    def _calculate_technical_features(
        self, data: pd.DataFrame, price_column: str, volume_column: str
    ) -> pd.DataFrame:
        """
        Calculate technical indicators for regime classification.

        Args:
            data: Market data DataFrame

        Returns:
            DataFrame with technical features
        """
        df = data.copy()

        # Trend indicators
        df["sma_short"] = df[price_column].rolling(window=10).mean()
        df["sma_long"] = (
            df[price_column].rolling(window=self.config["trend_window"]).mean()
        )
        df["ema_short"] = df[price_column].ewm(span=12).mean()
        df["ema_long"] = df[price_column].ewm(span=26).mean()

        # MACD
        df["macd"] = df["ema_short"] - df["ema_long"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Volatility indicators
        df["returns"] = df[price_column].pct_change()
        df["volatility"] = (
            df["returns"].rolling(window=self.config["volatility_window"]).std()
        )
        df["atr"] = self._calculate_atr(
            df, high="high", low="low", close=price_column, window=14
        )

        # Bollinger Bands
        df["bb_middle"] = df[price_column].rolling(window=20).mean()
        df["bb_std"] = df[price_column].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Momentum indicators
        df["rsi"] = self._calculate_rsi(
            df[price_column], window=self.config["momentum_window"]
        )
        df["stoch_k"] = self._calculate_stochastic(df, window=14)
        df["williams_r"] = self._calculate_williams_r(df, window=14)

        # Volume indicators
        if volume_column in df.columns:
            df["volume_sma"] = df[volume_column].rolling(window=20).mean()
            df["volume_ratio"] = df[volume_column] / df["volume_sma"]
        else:
            df["volume_ratio"] = 1.0

        # Trend strength and direction
        df["trend_direction"] = (df["sma_short"] - df["sma_long"]) / df["sma_long"]
        df["trend_strength"] = abs(df["trend_direction"])

        return df

    def _calculate_atr(
        self, df: pd.DataFrame, high: str, low: str, close: str, window: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df[high] - df[low]
        high_close = (df[high] - df[close].shift(1)).abs()
        low_close = (df[low] - df[close].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Stochastic Oscillator %K."""
        lowest_low = df["low"].rolling(window=window).min()
        highest_high = df["high"].rolling(window=window).max()
        return 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)

    def _calculate_williams_r(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df["high"].rolling(window=window).max()
        lowest_low = df["low"].rolling(window=window).min()
        return -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    def _classify_single_period(self, row: pd.Series) -> MarketCondition:
        """
        Classify market condition for a single time period.

        Args:
            row: DataFrame row with technical indicators

        Returns:
            MarketCondition object
        """
        # Classify trend regime
        regime, trend_confidence = self._classify_trend_regime(row)

        # Classify volatility level
        volatility, vol_confidence = self._classify_volatility(row)

        # Calculate overall confidence
        confidence = min(trend_confidence, vol_confidence)

        # Extract features for analysis
        features = {
            "trend_direction": row.get("trend_direction", 0),
            "trend_strength": row.get("trend_strength", 0),
            "volatility": row.get("volatility", 0),
            "rsi": row.get("rsi", 50),
            "macd_histogram": row.get("macd_histogram", 0),
            "bb_width": row.get("bb_width", 0),
            "volume_ratio": row.get("volume_ratio", 1.0),
        }

        return MarketCondition(
            regime=regime,
            volatility=volatility,
            trend_strength=row.get("trend_strength", 0),
            trend_direction=row.get("trend_direction", 0),
            confidence=confidence,
            timestamp=pd.Timestamp.now(),  # Will be set by caller
            features=features,
        )

    def _classify_trend_regime(self, row: pd.Series) -> Tuple[MarketRegime, float]:
        """
        Classify trend regime based on technical indicators.

        Returns:
            Tuple of (regime, confidence)
        """
        trend_direction = row.get("trend_direction", 0)
        trend_strength = row.get("trend_strength", 0)
        macd_hist = row.get("macd_histogram", 0)
        rsi = row.get("rsi", 50)

        threshold = self.config["regime_threshold"]

        # Strong bull signals
        if (
            trend_direction > threshold
            and trend_strength > threshold
            and macd_hist > 0
            and rsi > 60
        ):
            return MarketRegime.BULL, 0.8

        # Strong bear signals
        elif (
            trend_direction < -threshold
            and trend_strength > threshold
            and macd_hist < 0
            and rsi < 40
        ):
            return MarketRegime.BEAR, 0.8

        # Sideways market
        else:
            return MarketRegime.SIDEWAYS, 0.6

    def _classify_volatility(self, row: pd.Series) -> Tuple[VolatilityLevel, float]:
        """
        Classify volatility level.

        Returns:
            Tuple of (volatility_level, confidence)
        """
        volatility = row.get("volatility", 0)
        bb_width = row.get("bb_width", 0)

        # Use historical percentiles for classification
        # For now, use fixed thresholds (can be made adaptive)
        if volatility > 0.05 or bb_width > 0.10:  # Very high volatility
            return VolatilityLevel.VERY_HIGH, 0.9
        elif volatility > 0.03 or bb_width > 0.07:  # High volatility
            return VolatilityLevel.HIGH, 0.8
        elif volatility > 0.02 or bb_width > 0.05:  # Medium volatility
            return VolatilityLevel.MEDIUM, 0.7
        elif volatility > 0.01 or bb_width > 0.03:  # Low volatility
            return VolatilityLevel.LOW, 0.6
        else:  # Very low volatility
            return VolatilityLevel.VERY_LOW, 0.5

    def get_regime_statistics(
        self, conditions: List[MarketCondition]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about market regime distribution.

        Args:
            conditions: List of MarketCondition objects

        Returns:
            Dictionary with regime statistics
        """
        if not conditions:
            return {}

        regimes = [c.regime.value for c in conditions]
        volatilities = [c.volatility.value for c in conditions]

        total_periods = len(conditions)

        return {
            "total_periods": total_periods,
            "regime_distribution": {
                regime: regimes.count(regime) / total_periods for regime in set(regimes)
            },
            "volatility_distribution": {
                vol: volatilities.count(vol) / total_periods
                for vol in set(volatilities)
            },
            "avg_trend_strength": np.mean([c.trend_strength for c in conditions]),
            "avg_confidence": np.mean([c.confidence for c in conditions]),
            "regime_transitions": self._calculate_regime_transitions(conditions),
        }

    def _calculate_regime_transitions(
        self, conditions: List[MarketCondition]
    ) -> Dict[str, int]:
        """Calculate regime transition frequencies."""
        transitions = {}
        for i in range(1, len(conditions)):
            prev_regime = conditions[i - 1].regime.value
            curr_regime = conditions[i].regime.value
            transition = f"{prev_regime}_to_{curr_regime}"
            transitions[transition] = transitions.get(transition, 0) + 1
        return transitions
