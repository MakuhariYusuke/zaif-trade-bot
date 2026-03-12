"""
SAC v437 Feature Engineering

Enhanced feature engineering for market-adaptive ensemble system.
Includes quality-filtered features, improved bull/bear balance, and regime awareness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ztb.sac_v427_market_adaptive_system import SACv427MarketAdaptiveSystem
except ImportError:
    SACv427MarketAdaptiveSystem = None

from ztb.features.feature_set_config import get_feature_config
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class SACv437FeatureEngineer:
    """
    Enhanced feature engineering for SAC v437.

    Generates quality-filtered, market-aware features including:
    - Quality-filtered regime-specific indicators (removed harmful features)
    - Balanced bull/bear market features
    - Correlation features with quality control
    - Ensemble prediction signals
    - Risk-adjusted technical indicators
    """

    def __init__(
        self,
        market_system: SACv427MarketAdaptiveSystem | None = None,
        config_path: str | None = None,
    ):
        if SACv427MarketAdaptiveSystem is None:
            self.market_system = None
        else:
            self.market_system = market_system or SACv427MarketAdaptiveSystem()
        # Use default config path if not specified
        if config_path is None:
            from pathlib import Path

            config_path = (
                Path(__file__).parent.parent.parent
                / "config"
                / "features"
                / "v437_feature_config.yaml"
            )
        self.config_path = config_path
        self.config = get_feature_config(config_path)
        self.feature_flags = self.config.get("feature_flags", {}) if hasattr(self.config, 'get') else {}

        # Quality control settings
        self.quality_thresholds = {
            "max_nan_rate": 0.10,  # 10% max NaN rate
            "min_variance": 1e-8,  # Minimum variance threshold
            "max_zero_rate": 0.80,  # 80% max zero rate
            "max_outlier_rate": 0.30,  # 30% max outlier rate
            "max_correlation": 0.95,  # 95% max correlation
        }

        logger.info(f"Initialized SACv437FeatureEngineer with config: {config_path}")

    def generate_v437_features(
        self,
        df: pd.DataFrame,
        feature_set: str = "high_quality",
        include_market_regime: bool = True,
    ) -> pd.DataFrame:
        """
        Generate enhanced v437 features with quality filtering.

        Args:
            df: Input dataframe with OHLCV data
            feature_set: Feature set to use ('minimal', 'balanced', 'high_quality', 'full')
            include_market_regime: Whether to include market regime features

        Returns:
            DataFrame with generated features
        """
        logger.info(f"Generating v437 features with set: {feature_set}")

        # Start with base features
        features_df = self._generate_base_features(df)

        # Add quality-filtered technical indicators
        if self.feature_flags.get("technical_indicators", True):
            features_df = self._add_technical_indicators(features_df, df)

        # Add market regime features
        if include_market_regime and self.market_system:
            features_df = self._add_market_regime_features(features_df, df)

        # Add correlation features with quality control
        if self.feature_flags.get("correlation_features", True):
            features_df = self._add_correlation_features(features_df, df)

        # Add ensemble signals
        if self.feature_flags.get("ensemble_signals", True):
            features_df = self._add_ensemble_signals(features_df, df)

        # Quality filter the features
        features_df = self._quality_filter_features(features_df)

        # Check and handle correlations
        if self.feature_flags.get("correlation_control", True):
            features_df = self._check_feature_correlations(features_df)

        # Always ensure essential price data is included for trading environment
        essential_data = self._get_essential_price_data(df)
        features_df = pd.concat([essential_data, features_df], axis=1)

        # Remove duplicates if any
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]

        logger.info(f"Generated {len(features_df.columns)} v437 features after quality filtering")
        return features_df

    def _generate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate base features (price and volume derived)."""
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Keep essential price data for trading environment
        features["close"] = close
        features["high"] = high
        features["low"] = low
        features["open"] = df.get("open", close)
        features["volume"] = volume

        # Basic price features (avoiding redundant OHLCV correlations in derived features)
        features["returns"] = close.pct_change()
        features["log_returns"] = np.log(close / close.shift(1)).fillna(0)

        # Volume features
        features["volume_ma_5"] = volume.rolling(5).mean()
        features["volume_ma_20"] = volume.rolling(20).mean()
        features["volume_ratio"] = volume / volume.rolling(20).mean()

        # Volatility features
        features["realized_volatility"] = features["returns"].rolling(20).std()
        features["close_volatility"] = close.rolling(20).std()

        return features

    def _add_technical_indicators(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality-filtered technical indicators."""
        close = df["close"]
        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Trend indicators
        features["sma_5"] = close.rolling(5).mean()
        features["sma_20"] = close.rolling(20).mean()
        features["ema_12"] = close.ewm(span=12).mean()
        features["ema_26"] = close.ewm(span=26).mean()

        # Momentum indicators
        features["rsi_14"] = self._calculate_rsi(close, 14)
        features["stoch_k"] = self._calculate_stoch_k(high, low, close, 14)
        features["williams_r"] = self._calculate_williams_r(high, low, close, 14)

        # MACD
        macd_features = self._calculate_macd(close)
        features = pd.concat([features, macd_features], axis=1)

        # Volume indicators
        features["obv"] = self._calculate_obv(close, volume)
        features["vpt"] = self._calculate_vpt(close, volume)
        features["vwap"] = self._calculate_vwap(close, volume)

        # Volatility indicators
        features["atr_14"] = self._calculate_atr(high, low, close, 14)
        features["bb_upper"], features["bb_lower"] = self._calculate_bollinger_bands(close, 20)

        return features

    def _add_market_regime_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime specific features."""
        if not self.market_system:
            return features

        # Bull market features
        bull_features = self._generate_bull_market_features(df)
        features = pd.concat([features, bull_features], axis=1)

        # Bear market features
        bear_features = self._generate_bear_market_features(df)
        features = pd.concat([features, bear_features], axis=1)

        return features

    def _add_correlation_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add correlation features with quality control."""
        close = df["close"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Price-volume correlation
        features["price_volume_corr_20"] = close.rolling(20).corr(volume)

        # Rolling correlations between indicators
        if "rsi_14" in features.columns and "stoch_k" in features.columns:
            features["rsi_stoch_corr_20"] = features["rsi_14"].rolling(20).corr(features["stoch_k"])

        return features

    def _add_ensemble_signals(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add ensemble prediction signals."""
        # Ensemble signals based on multiple indicators
        features["trend_ensemble"] = (
            ((features.get("sma_5", 0) > features.get("sma_20", 0)) * 1) +
            ((features.get("ema_12", 0) > features.get("ema_26", 0)) * 1)
        ) / 2.0

        features["momentum_ensemble"] = (
            ((features.get("rsi_14", 50) > 50) * 1) +
            ((features.get("stoch_k", 50) > 50) * 1)
        ) / 2.0

        return features

    def _quality_filter_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive quality filters to remove harmful features."""
        filtered_features = features.copy()
        removed_features = []

        # Essential columns that should never be removed (needed for trading environment)
        essential_columns = {"close", "high", "low", "open", "volume"}

        for col in features.columns:
            # Skip essential columns
            if col in essential_columns:
                continue

            series = features[col]

            # Check for excessive NaN
            nan_rate = series.isna().mean()
            if nan_rate > self.quality_thresholds["max_nan_rate"]:
                logger.warning(f"Removing feature {col}: NaN rate {nan_rate:.3f} > {self.quality_thresholds['max_nan_rate']}")
                filtered_features.drop(col, axis=1, inplace=True)
                removed_features.append(col)
                continue

            # Check for zero variance
            try:
                var_value = series.var()
                if pd.isna(var_value) or var_value <= self.quality_thresholds["min_variance"]:
                    logger.warning(f"Removing feature {col}: variance {var_value} <= {self.quality_thresholds['min_variance']}")
                    filtered_features.drop(col, axis=1, inplace=True)
                    removed_features.append(col)
                    continue
            except Exception:
                logger.warning(f"Removing feature {col}: variance calculation failed")
                filtered_features.drop(col, axis=1, inplace=True)
                removed_features.append(col)
                continue

            # Check for excessive zeros
            zero_rate = (series == 0).mean()
            if zero_rate > self.quality_thresholds["max_zero_rate"]:
                logger.warning(f"Removing feature {col}: zero rate {zero_rate:.3f} > {self.quality_thresholds['max_zero_rate']}")
                filtered_features.drop(col, axis=1, inplace=True)
                removed_features.append(col)
                continue

            # Check for excessive outliers (using IQR method)
            if len(series.dropna()) > 10:  # Need minimum data points
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outlier_rate = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).mean()
                if outlier_rate > self.quality_thresholds["max_outlier_rate"]:
                    logger.warning(f"Removing feature {col}: outlier rate {outlier_rate:.3f} > {self.quality_thresholds['max_outlier_rate']}")
                    filtered_features.drop(col, axis=1, inplace=True)
                    removed_features.append(col)
                    continue

        # Log quality filtering results
        if removed_features:
            logger.info(f"Quality filtering removed {len(removed_features)} harmful features: {removed_features}")
        logger.info(f"Quality filtering kept {len(filtered_features.columns)} features out of {len(features.columns)}")

        return filtered_features

    def _check_feature_correlations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Check and handle highly correlated features."""
        if len(features.columns) < 2:
            return features

        # Calculate correlation matrix
        corr_matrix = features.corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                try:
                    corr_float = np.abs(corr_val)
                    if corr_float > self.quality_thresholds["max_correlation"]:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
                except Exception:
                    continue

        # Remove one feature from each highly correlated pair
        # Prioritize keeping features with higher variance
        to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 in to_remove or feat2 in to_remove:
                continue

            var1 = features[feat1].var()
            var2 = features[feat2].var()

            # Keep the feature with higher variance
            if var1 >= var2:
                to_remove.add(feat2)
                logger.info(f"Removing {feat2} due to high correlation ({corr:.3f}) with {feat1} (keeping higher variance feature)")
            else:
                to_remove.add(feat1)
                logger.info(f"Removing {feat1} due to high correlation ({corr:.3f}) with {feat2} (keeping higher variance feature)")

        if to_remove:
            features = features.drop(columns=list(to_remove))
            logger.info(f"Correlation filtering removed {len(to_remove)} features")

        return features

    def _generate_bull_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features optimized for bull markets."""
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Momentum features
        features["bull_momentum_5"] = close.pct_change(5)
        features["bull_momentum_20"] = close.pct_change(20)

        # Volume confirmation
        features["bull_volume_ratio"] = volume / volume.rolling(20).mean()
        features["bull_volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # RSI divergence (bullish signals)
        rsi = self._calculate_rsi(close, 14)
        features["bull_rsi_divergence"] = rsi - rsi.rolling(20).mean()

        # MACD strength
        macd_features = self._calculate_macd(close)
        features["bull_macd_strength"] = macd_features["macd"] - macd_features["macd_signal"]

        # Trend strength
        features["bull_trend_strength"] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-8)

        return features

    def _generate_bear_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features optimized for bear markets."""
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df.get("high", close)
        low = df.get("low", close)

        # Downtrend momentum
        features["bear_momentum_5"] = -close.pct_change(5)  # Negative for bearish
        features["bear_momentum_20"] = -close.pct_change(20)

        # Support breakdown signals
        features["bear_support_break"] = (close < close.rolling(20).min()).astype(int)

        # Increased volatility in downtrends
        returns = close.pct_change()
        features["bear_volatility"] = returns[returns < 0].rolling(20).std().fillna(0)

        # Bearish RSI signals
        rsi = self._calculate_rsi(close, 14)
        features["bear_rsi_oversold"] = (rsi < 30).astype(int)

        return features

    # Technical indicator calculation methods (same as v427 but quality-controlled)

    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower

    def _calculate_stoch_k(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window).max()
        lowest_low = low.rolling(window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """Calculate Average True Range."""
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(0.0, index=close.index, dtype=float)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend."""
        close = df["close"]
        essential["close"] = close
        essential["high"] = df.get("high", close)
        essential["low"] = df.get("low", close)
        essential["open"] = df.get("open", close)
        essential["volume"] = df.get("volume", pd.Series(1, index=df.index))

        return essential