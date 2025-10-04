#!/usr/bin/env python3
"""
Enhanced training data generator with comprehensive features.

Generates training data with all 25+ technical indicators from FeatureRegistry
plus the original 7 basic features for comprehensive model training.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.features import FeatureRegistry
from ztb.utils.data.data_generation import generate_synthetic_market_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_enhanced_training_data(
    n_samples: int = 10000, output_path: str = "ml-dataset-enhanced.csv", seed: int = 42
) -> pd.DataFrame:
    """
    Generate enhanced training data with comprehensive features.

    Args:
        n_samples: Number of samples to generate
        output_path: Output CSV file path
        seed: Random seed for reproducibility

    Returns:
        DataFrame with enhanced features
    """
    logger.info(f"Generating enhanced training data with {n_samples} samples...")

    # Initialize FeatureRegistry
    FeatureRegistry.initialize(seed=seed)

    # Generate base market data (OHLCV)
    logger.info("Generating synthetic market data...")
    market_df = generate_synthetic_market_data(n_samples=n_samples, seed=seed)

    # Ensure we have required OHLCV columns
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in market_df.columns for col in required_cols):
        raise ValueError(f"Market data missing required columns: {required_cols}")

    # Create timestamp column
    market_df["ts"] = pd.date_range("2023-01-01", periods=n_samples, freq="1min")

    # Add pair column
    market_df["pair"] = "BTC/JPY"

    # Generate trading signals (simplified)
    np.random.seed(seed)
    market_df["side"] = np.random.choice(["buy", "sell"], size=n_samples)

    # Compute basic features (matching original training data)
    logger.info("Computing basic features...")

    # RSI (14-period)
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    market_df["rsi"] = calculate_rsi(market_df["close"], period=14)

    # SMA short (5-period) and long (20-period)
    market_df["sma_short"] = market_df["close"].rolling(window=5).mean()
    market_df["sma_long"] = market_df["close"].rolling(window=20).mean()

    # Price (normalized)
    market_df["price"] = market_df["close"]  # Keep original scale for now

    # Quantity (mock trading volume)
    market_df["qty"] = np.random.uniform(0.001, 0.01, size=n_samples)

    # PnL and win flag (simplified simulation)
    market_df["pnl"] = np.random.normal(0, 1000, size=n_samples)
    market_df["win"] = (market_df["pnl"] > 0).astype(int)

    # Add ATR column required by some features (Donchian, Supertrend)
    logger.info("Computing ATR for dependent features...")
    market_df["atr_10"] = (
        market_df["high"].rolling(window=10).max()
        - market_df["low"].rolling(window=10).min()
    )
    market_df["ATR_simplified"] = market_df["atr_10"]  # Some features expect this name

    # Add EMA columns required by EMACross
    market_df["ema_5"] = market_df["close"].ewm(span=5, adjust=False).mean()
    market_df["rolling_mean_20"] = market_df["close"].rolling(window=20).mean()

    # Compute advanced features using FeatureRegistry
    logger.info("Computing advanced features...")
    advanced_features = FeatureRegistry.list()
    logger.info(
        f"Computing {len(advanced_features)} advanced features: {advanced_features}"
    )

    # Sort features to handle dependencies (ATR-dependent features last)
    atr_dependent_features = [
        "Donchian_Pos_20",
        "Donchian_Width_Rel_20",
        "Donchian_Slope_20",
        "Supertrend",
        "Supertrend_Direction",
    ]
    other_features = [f for f in advanced_features if f not in atr_dependent_features]
    sorted_features = other_features + atr_dependent_features

    for feature_name in sorted_features:
        try:
            logger.debug(f"Computing {feature_name}...")
            feature_func = FeatureRegistry.get(feature_name)
            market_df[feature_name] = feature_func(market_df)
        except Exception as e:
            logger.warning(f"Failed to compute {feature_name}: {e}")
            market_df[feature_name] = np.nan

    # Handle NaN values with better strategy
    logger.info("Handling NaN values...")
    # First forward fill, then backward fill, then fill remaining with 0
    market_df = market_df.ffill().bfill().fillna(0)

    # Reorder columns to put basic features first
    basic_cols = [
        "ts",
        "pair",
        "side",
        "rsi",
        "sma_short",
        "sma_long",
        "price",
        "qty",
        "pnl",
        "win",
        "source",
    ]
    advanced_cols = sorted([col for col in market_df.columns if col not in basic_cols])

    # Ensure all basic columns exist
    for col in basic_cols:
        if col not in market_df.columns:
            if col == "source":
                market_df[col] = "enhanced"
            else:
                market_df[col] = 0.0

    final_cols = basic_cols + advanced_cols
    market_df = market_df[final_cols]

    # Save to CSV
    logger.info(f"Saving enhanced training data to {output_path}...")
    market_df.to_csv(output_path, index=False)

    logger.info(f"Enhanced training data saved. Shape: {market_df.shape}")
    logger.info(
        f"Total features: {len(market_df.columns) - len(['ts', 'pair', 'side', 'source'])}"
    )
    logger.info(f"Basic features: 7 (rsi, sma_short, sma_long, price, qty, pnl, win)")
    logger.info(f"Advanced features: {len(advanced_cols)}")

    return market_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate enhanced training data with comprehensive features"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default="ml-dataset-enhanced.csv",
        help="Output CSV file path (default: ml-dataset-enhanced.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    try:
        df = generate_enhanced_training_data(
            n_samples=args.n_samples, output_path=args.output, seed=args.seed
        )
        logger.info("✅ Enhanced training data generation completed successfully!")
        logger.info(f"Generated {len(df)} samples with {len(df.columns)} columns")

    except Exception as e:
        logger.error(f"❌ Failed to generate enhanced training data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
