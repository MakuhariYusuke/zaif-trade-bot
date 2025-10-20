#!/usr/bin/env python3
"""
Data Augmentation and Bias Mitigation Tools for BTC Trading Data
過去BTCデータの拡張とバイアス軽減ツール

This module provides tools to extend historical BTC data and mitigate biases
by adding diverse market conditions and synthetic data generation.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor

logger = get_logger(__name__)


class BTCDataAugmentor:
    """BTCデータ拡張クラス - 過去データの拡張とバイアス軽減"""

    def __init__(self, base_data_path: str):
        """Initialize with base dataset"""
        self.base_data_path = Path(base_data_path)
        self.performance_monitor = PerformanceMonitor("btc_data_augmentor")
        self.base_data = None
        self._load_base_data()

    def _load_base_data(self):
        """Load base BTC dataset"""
        try:
            if self.base_data_path.suffix == ".csv":
                self.base_data = pd.read_csv(self.base_data_path)
            elif self.base_data_path.suffix == ".pkl":
                self.base_data = pd.read_pickle(self.base_data_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {self.base_data_path.suffix}"
                )

            # Convert timestamp to datetime
            if "timestamp" in self.base_data.columns:
                self.base_data["timestamp"] = pd.to_datetime(
                    self.base_data["timestamp"]
                )
                self.base_data = self.base_data.sort_values("timestamp").reset_index(
                    drop=True
                )

            logger.info(
                f"Loaded base data: {len(self.base_data)} records from {self.base_data['timestamp'].min()} to {self.base_data['timestamp'].max()}"
            )

        except Exception as e:
            logger.error(f"Failed to load base data: {e}")
            raise

    def analyze_data_bias(self) -> Dict[str, Any]:
        """Analyze potential biases in the current dataset"""
        with self.performance_monitor:
            if self.base_data is None:
                raise ValueError("Base data not loaded")

            bias_analysis = {}

            # Time period analysis
            time_range = (
                self.base_data["timestamp"].max() - self.base_data["timestamp"].min()
            )
            bias_analysis["time_range_days"] = time_range.days
            bias_analysis["data_start"] = self.base_data["timestamp"].min()
            bias_analysis["data_end"] = self.base_data["timestamp"].max()

            # Price trend analysis
            if "close" in self.base_data.columns:
                start_price = self.base_data["close"].iloc[0]
                end_price = self.base_data["close"].iloc[-1]
                total_return = (end_price - start_price) / start_price

                # Calculate trend strength
                returns = self.base_data["close"].pct_change().dropna()
                positive_days = (returns > 0).sum()
                total_days = len(returns)
                bias_analysis["trend_bias"] = (
                    positive_days / total_days if total_days > 0 else 0
                )
                bias_analysis["total_return"] = total_return

                # Volatility analysis
                bias_analysis["volatility"] = returns.std()
                bias_analysis["max_drawdown"] = self._calculate_max_drawdown(
                    self.base_data["close"]
                )

            # Market regime analysis
            if "returns" in self.base_data.columns:
                bias_analysis["regime_distribution"] = self._analyze_market_regimes()

            # Volume analysis
            if "volume" in self.base_data.columns:
                bias_analysis["avg_volume"] = self.base_data["volume"].mean()
                bias_analysis["volume_volatility"] = (
                    self.base_data["volume"].std() / self.base_data["volume"].mean()
                )

            logger.info(f"Bias analysis completed: {bias_analysis}")
            return bias_analysis

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def _analyze_market_regimes(self) -> Dict[str, float]:
        """Analyze distribution of market regimes"""
        returns = self.base_data["returns"].dropna()

        # Simple regime classification based on returns
        bullish = (returns > 0.01).sum()
        bearish = (returns < -0.01).sum()
        sideways = len(returns) - bullish - bearish

        total = len(returns)
        return {
            "bullish_pct": bullish / total if total > 0 else 0,
            "bearish_pct": bearish / total if total > 0 else 0,
            "sideways_pct": sideways / total if total > 0 else 0,
        }

    def extend_historical_data(self, years_back: int = 2) -> pd.DataFrame:
        """Extend data backwards by generating synthetic historical data"""
        with self.performance_monitor:
            if self.base_data is None:
                raise ValueError("Base data not loaded")

            logger.info(f"Extending data {years_back} years back")

            # Get base statistics
            base_stats = self._calculate_base_statistics()

            # Generate extended data
            extended_data = []
            current_date = self.base_data["timestamp"].min()

            for i in range(years_back * 365):  # Approximate days
                # Generate synthetic data for each day
                day_data = self._generate_synthetic_day(current_date, base_stats)
                extended_data.extend(day_data)
                current_date -= timedelta(days=1)

            # Convert to DataFrame and combine
            extended_df = pd.DataFrame(extended_data)
            extended_df["timestamp"] = pd.to_datetime(extended_df["timestamp"])

            # Combine with original data
            combined_data = pd.concat([extended_df, self.base_data], ignore_index=True)
            combined_data = combined_data.sort_values("timestamp").reset_index(
                drop=True
            )

            logger.info(
                f"Extended data from {len(self.base_data)} to {len(combined_data)} records"
            )
            return combined_data

    def _calculate_base_statistics(self) -> Dict[str, Any]:
        """Calculate statistical properties of base data for synthetic generation"""
        stats = {}

        if "close" in self.base_data.columns:
            prices = self.base_data["close"]
            stats["price_mean"] = prices.mean()
            stats["price_std"] = prices.std()
            stats["price_min"] = prices.min()
            stats["price_max"] = prices.max()

        if "volume" in self.base_data.columns:
            volumes = self.base_data["volume"]
            stats["volume_mean"] = volumes.mean()
            stats["volume_std"] = volumes.std()

        if "returns" in self.base_data.columns:
            returns = self.base_data["returns"].dropna()
            stats["return_mean"] = returns.mean()
            stats["return_std"] = returns.std()

        return stats

    def _generate_synthetic_day(
        self, date: datetime, base_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate synthetic data for a single day"""
        day_data = []

        # Generate multiple price points per day (assuming minute data)
        minutes_per_day = 1440  # 24 hours * 60 minutes
        base_price = base_stats.get("price_mean", 5000000)

        # Add some trend and volatility variation
        day_trend = np.random.normal(0, 0.02)  # Random daily trend
        day_volatility = np.random.uniform(0.5, 1.5) * base_stats.get(
            "return_std", 0.01
        )

        current_price = base_price * (
            1 + np.random.normal(0, 0.05)
        )  # Start near base price

        for minute in range(minutes_per_day):
            timestamp = date + timedelta(minutes=minute)

            # Generate price movement
            return_change = np.random.normal(
                day_trend / minutes_per_day, day_volatility
            )
            current_price *= 1 + return_change

            # Ensure price stays within reasonable bounds
            current_price = np.clip(
                current_price,
                base_stats.get("price_min", 1000000),
                base_stats.get("price_max", 10000000),
            )

            # Generate OHLC
            volatility = abs(np.random.normal(0, 0.005))
            high = current_price * (1 + volatility)
            low = current_price * (1 - volatility)
            open_price = current_price * (1 + np.random.normal(0, 0.002))
            close = current_price

            # Generate volume
            volume = max(
                1,
                np.random.normal(
                    base_stats.get("volume_mean", 100), base_stats.get("volume_std", 50)
                ),
            )

            # Create record
            record = {
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "returns": return_change,
            }

            # Add technical indicators if they exist in base data
            if "sma_5" in self.base_data.columns:
                record.update(self._generate_technical_indicators(record, day_data))

            day_data.append(record)

        return day_data

    def _generate_technical_indicators(
        self, current_record: Dict[str, Any], day_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate basic technical indicators for synthetic data"""
        indicators = {}

        # Simple moving averages (simplified)
        if len(day_data) >= 5:
            closes = [r["close"] for r in day_data[-4:]] + [current_record["close"]]
            indicators["sma_5"] = np.mean(closes)

        if len(day_data) >= 10:
            closes = [r["close"] for r in day_data[-9:]] + [current_record["close"]]
            indicators["sma_10"] = np.mean(closes)

        if len(day_data) >= 20:
            closes = [r["close"] for r in day_data[-19:]] + [current_record["close"]]
            indicators["sma_20"] = np.mean(closes)

        # RSI (simplified)
        if len(day_data) >= 14:
            gains = []
            losses = []
            for r in day_data[-13:] + [current_record]:
                ret = r.get("returns", 0)
                if ret > 0:
                    gains.append(ret)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(ret))

            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0

            if avg_loss == 0:
                indicators["rsi_14"] = 100
            else:
                rs = avg_gain / avg_loss
                indicators["rsi_14"] = 100 - (100 / (1 + rs))

        return indicators

    def add_diverse_market_conditions(
        self, target_samples: int = 10000
    ) -> pd.DataFrame:
        """Add diverse market conditions to mitigate bias"""
        with self.performance_monitor:
            if self.base_data is None:
                raise ValueError("Base data not loaded")

            logger.info(f"Adding {target_samples} diverse market condition samples")

            diverse_data = []
            base_stats = self._calculate_base_statistics()

            # Generate data for different market regimes
            regimes = [
                "strong_bull",
                "moderate_bull",
                "sideways",
                "moderate_bear",
                "strong_bear",
                "high_volatility",
                "low_volatility",
            ]

            samples_per_regime = target_samples // len(regimes)

            for regime in regimes:
                regime_data = self._generate_regime_data(
                    regime, samples_per_regime, base_stats
                )
                diverse_data.extend(regime_data)

            # Convert to DataFrame
            diverse_df = pd.DataFrame(diverse_data)
            diverse_df["timestamp"] = pd.to_datetime(diverse_df["timestamp"])

            # Combine with original data
            combined_data = pd.concat([self.base_data, diverse_df], ignore_index=True)
            combined_data = combined_data.sort_values("timestamp").reset_index(
                drop=True
            )

            logger.info(f"Added diverse conditions: {len(diverse_df)} new samples")
            return combined_data

    def _generate_regime_data(
        self, regime: str, num_samples: int, base_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate data for specific market regime"""
        data = []

        # Regime-specific parameters
        regime_params = {
            "strong_bull": {
                "trend": 0.02,
                "volatility": 0.015,
                "volume_multiplier": 1.5,
            },
            "moderate_bull": {
                "trend": 0.005,
                "volatility": 0.01,
                "volume_multiplier": 1.2,
            },
            "sideways": {"trend": 0.0, "volatility": 0.008, "volume_multiplier": 0.8},
            "moderate_bear": {
                "trend": -0.005,
                "volatility": 0.012,
                "volume_multiplier": 1.1,
            },
            "strong_bear": {
                "trend": -0.02,
                "volatility": 0.02,
                "volume_multiplier": 1.8,
            },
            "high_volatility": {
                "trend": 0.0,
                "volatility": 0.03,
                "volume_multiplier": 2.0,
            },
            "low_volatility": {
                "trend": 0.0,
                "volatility": 0.003,
                "volume_multiplier": 0.5,
            },
        }

        params = regime_params[regime]
        base_price = base_stats.get("price_mean", 5000000)

        # Generate start date before existing data
        start_date = self.base_data["timestamp"].min() - timedelta(days=365)

        current_price = base_price * (1 + np.random.normal(0, 0.1))

        for i in range(num_samples):
            timestamp = start_date + timedelta(minutes=i)

            # Generate price movement based on regime
            return_change = np.random.normal(
                params["trend"] / 1440, params["volatility"]
            )
            current_price *= 1 + return_change

            # Generate OHLC
            volatility = abs(np.random.normal(0, params["volatility"]))
            high = current_price * (1 + volatility)
            low = current_price * (1 - volatility)
            open_price = current_price * (
                1 + np.random.normal(0, params["volatility"] * 0.5)
            )
            close = current_price

            # Generate volume
            base_volume = base_stats.get("volume_mean", 100)
            volume = max(
                1,
                np.random.normal(
                    base_volume * params["volume_multiplier"],
                    base_stats.get("volume_std", 50),
                ),
            )

            record = {
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "returns": return_change,
                "market_regime": regime,
            }

            data.append(record)

        return data

    def save_augmented_data(self, augmented_data: pd.DataFrame, output_path: str):
        """Save augmented dataset"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            augmented_data.to_csv(output_path, index=False)
        elif output_path.suffix == ".pkl":
            augmented_data.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        logger.info(
            f"Saved augmented data to {output_path} ({len(augmented_data)} records)"
        )


class BTCBiasDetector:
    """BTCバイアス検出クラス"""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor("btc_bias_detector")

    def detect_data_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect various types of bias in BTC trading data"""
        with self.performance_monitor:
            bias_report = {}

            # Time period bias
            bias_report["time_period_bias"] = self._check_time_period_bias(data)

            # Trend bias
            bias_report["trend_bias"] = self._check_trend_bias(data)

            # Volatility bias
            bias_report["volatility_bias"] = self._check_volatility_bias(data)

            # Market regime bias
            bias_report["regime_bias"] = self._check_regime_bias(data)

            # Volume bias
            bias_report["volume_bias"] = self._check_volume_bias(data)

            # BTC-specific biases
            bias_report["btc_specific_bias"] = self._check_btc_specific_bias(data)

            return bias_report

    def _check_time_period_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if data covers diverse time periods"""
        if "timestamp" not in data.columns:
            return {"bias_detected": True, "reason": "No timestamp column"}

        timestamps = pd.to_datetime(data["timestamp"])
        time_range = timestamps.max() - timestamps.min()

        # Check for seasonal coverage
        months_covered = set(timestamps.dt.month)
        years_covered = set(timestamps.dt.year)

        bias_score = 1 - (len(months_covered) / 12)  # Lower score = more diverse

        return {
            "bias_score": bias_score,
            "time_range_days": time_range.days,
            "months_covered": len(months_covered),
            "years_covered": len(years_covered),
            "bias_detected": bias_score > 0.7,
        }

    def _check_trend_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for trend bias (overwhelming bull/bear markets)"""
        if "returns" not in data.columns and "close" not in data.columns:
            return {"bias_detected": True, "reason": "No returns or price data"}

        if "returns" in data.columns:
            returns = data["returns"].dropna()
        else:
            returns = data["close"].pct_change().dropna()

        positive_returns = (returns > 0).sum()
        total_returns = len(returns)

        trend_ratio = positive_returns / total_returns if total_returns > 0 else 0.5

        # Ideal is around 0.5 (balanced up/down days)
        bias_score = abs(trend_ratio - 0.5) * 2

        return {
            "bias_score": bias_score,
            "positive_ratio": trend_ratio,
            "bias_detected": bias_score > 0.6,
        }

    def _check_volatility_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for volatility clustering or bias"""
        if "returns" not in data.columns and "close" not in data.columns:
            return {"bias_detected": True, "reason": "No returns or price data"}

        if "returns" in data.columns:
            returns = data["returns"].dropna()
        else:
            returns = data["close"].pct_change().dropna()

        volatility = returns.std()

        # Check for volatility clustering (periods of high/low vol)
        rolling_vol = returns.rolling(50).std()
        vol_of_vol = (
            rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() != 0 else 0
        )

        # High vol_of_vol indicates volatility clustering
        bias_score = min(vol_of_vol * 10, 1.0)

        return {
            "bias_score": bias_score,
            "volatility": volatility,
            "vol_of_vol": vol_of_vol,
            "bias_detected": bias_score > 0.7,
        }

    def _check_regime_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for market regime bias"""
        if "returns" not in data.columns and "close" not in data.columns:
            return {"bias_detected": True, "reason": "No returns or price data"}

        if "returns" in data.columns:
            returns = data["returns"].dropna()
        else:
            returns = data["close"].pct_change().dropna()

        # Classify regimes
        high_vol_threshold = returns.std() * 2
        trend_threshold = returns.mean() * 2

        high_vol_periods = (returns.abs() > high_vol_threshold).sum()
        bull_periods = (returns > trend_threshold).sum()
        bear_periods = (returns < -trend_threshold).sum()
        sideways_periods = len(returns) - high_vol_periods - bull_periods - bear_periods

        total = len(returns)
        regime_distribution = {
            "high_vol_pct": high_vol_periods / total,
            "bull_pct": bull_periods / total,
            "bear_pct": bear_periods / total,
            "sideways_pct": sideways_periods / total,
        }

        # Check if any regime dominates
        max_regime_pct = max(regime_distribution.values())
        bias_score = max_regime_pct - 0.25  # Ideal is ~25% each

        return {
            "bias_score": bias_score,
            "regime_distribution": regime_distribution,
            "bias_detected": bias_score > 0.5,
        }

    def _check_volume_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for volume-related biases"""
        if "volume" not in data.columns:
            return {"bias_detected": False, "reason": "No volume data available"}

        volumes = data["volume"].dropna()

        # Check for volume concentration
        volume_skewness = volumes.skew()
        volume_kurtosis = volumes.kurtosis()

        # Extreme skewness/kurtosis indicates volume bias
        bias_score = (abs(volume_skewness) + abs(volume_kurtosis)) / 10

        return {
            "bias_score": bias_score,
            "volume_skewness": volume_skewness,
            "volume_kurtosis": volume_kurtosis,
            "bias_detected": bias_score > 0.5,
        }

    def _check_btc_specific_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for BTC-specific biases (halving cycles, news events, etc.)"""
        bias_issues = []

        if "timestamp" in data.columns:
            timestamps = pd.to_datetime(data["timestamp"])

            # Check for halving cycle bias (BTC halves every 4 years)
            years = set(timestamps.dt.year)
            if len(years) < 2:
                bias_issues.append(
                    "Limited to single year - missing halving cycle effects"
                )

            # Check for weekend/weekday bias (crypto trades 24/7)
            weekday_dist = timestamps.dt.weekday.value_counts(normalize=True)
            if weekday_dist.std() > 0.1:  # Significant variation
                bias_issues.append("Uneven weekday distribution")

        # Check for price level bias (only high/low price ranges)
        if "close" in data.columns:
            prices = data["close"]
            price_range = prices.max() / prices.min()
            if price_range < 2:  # Less than 2x range
                bias_issues.append("Limited price range - missing full market cycles")

        return {
            "bias_detected": len(bias_issues) > 0,
            "issues": bias_issues,
            "bias_score": len(bias_issues) / 5,  # Normalize to 0-1
        }
