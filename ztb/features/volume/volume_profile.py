"""
volume_profile.py
Volume Profile analysis for price levels and volume distribution
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Volume_Profile_Point_of_Control")
def compute_volume_profile_poc(df: pd.DataFrame) -> pd.Series:
    """Volume Profile Point of Control (price level with highest volume)"""
    extended_features = calculate_volume_profile_extended(df)
    return extended_features["volume_profile_poc"]


@FeatureRegistry.register("Volume_Profile_Value_Area_High")
def compute_volume_profile_vah(df: pd.DataFrame) -> pd.Series:
    """Volume Profile Value Area High (70% of volume)"""
    extended_features = calculate_volume_profile_extended(df)
    return extended_features["volume_profile_vah"]


@FeatureRegistry.register("Volume_Profile_Value_Area_Low")
def compute_volume_profile_val(df: pd.DataFrame) -> pd.Series:
    """Volume Profile Value Area Low (70% of volume)"""
    extended_features = calculate_volume_profile_extended(df)
    return extended_features["volume_profile_val"]


@FeatureRegistry.register("Volume_Profile_Distribution")
def compute_volume_profile_distribution(df: pd.DataFrame) -> pd.Series:
    """Volume Profile Distribution (normalized volume at each price level)"""
    extended_features = calculate_volume_profile_extended(df)
    return extended_features["volume_profile_distribution"]


def calculate_volume_profile_extended(
    df: pd.DataFrame,
    bins: int = 50,
    value_area_percent: float = 0.7,
) -> pd.DataFrame:
    """
    Calculate Volume Profile features using precise price-volume distribution.

    This implementation uses a more accurate method where volume is distributed
    across all price levels that a bar spans, proportional to the price range.

    Args:
        df: DataFrame with OHLCV data
        bins: Number of price bins for volume distribution
        value_area_percent: Percentage of volume for value area calculation

    Returns:
        DataFrame with Volume Profile features
    """
    result = pd.DataFrame(index=df.index)

    # Calculate price range for binning
    price_min = df["low"].min()
    price_max = df["high"].max()
    price_range = price_max - price_min

    if price_range == 0:
        # Handle edge case of no price movement
        result["volume_profile_poc"] = df["close"]
        result["volume_profile_vah"] = df["close"]
        result["volume_profile_val"] = df["close"]
        result["volume_profile_distribution"] = 0.0
        return result

    # Create price bins with equal range
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_width = (price_max - price_min) / bins

    n = len(df)
    poc_values = np.zeros(n)
    vah_values = np.zeros(n)
    val_values = np.zeros(n)
    distribution_values = []

    # Rolling window for volume profile (use last 100 periods or all available)
    window_size = min(100, n)

    for i in range(n):
        start_idx = max(0, i - window_size + 1)

        # Get data for current window
        window_data = df.iloc[start_idx : i + 1]

        # Calculate volume profile using precise distribution
        poc, vah, val, current_dist, full_dist = _compute_volume_profile_single(
            window_data["high"].to_numpy(dtype=np.float64),
            window_data["low"].to_numpy(dtype=np.float64),
            window_data["volume"].to_numpy(dtype=np.float64),
            bin_edges,
            bin_width,
            value_area_percent,
            df["close"].iloc[i],
        )

        poc_values[i] = poc
        vah_values[i] = vah
        val_values[i] = val
        distribution_values.append(current_dist)

    result["volume_profile_poc"] = poc_values
    result["volume_profile_vah"] = vah_values
    result["volume_profile_val"] = val_values
    # Store the volume distribution at current price level (scalar value)
    result["volume_profile_distribution"] = distribution_values

    return result


def _compute_volume_profile_single(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    bin_edges: npt.NDArray[np.float64],
    bin_width: float,
    value_area_percent: float,
    current_price: float,
) -> tuple[float, float, float, float, NDArray[np.float64]]:
    """
    Calculate Volume Profile for a single window using vectorized numpy operations.

    Distributes volume across price levels more accurately by considering
    the full range that each price bar spans.
    """
    n_bars = len(high)
    n_bins = len(bin_edges) - 1

    if n_bars == 0:
        return current_price, current_price, current_price, 0.0, np.zeros(n_bins)

    # Vectorized calculation of bin indices for all bars
    start_bins = np.floor((low - bin_edges[0]) / bin_width).astype(int)
    end_bins = np.floor((high - bin_edges[0]) / bin_width).astype(int)

    # Ensure bin indices are within bounds
    start_bins = np.clip(start_bins, 0, n_bins - 1)
    end_bins = np.clip(end_bins, 0, n_bins - 1)

    # Initialize volume histogram
    volume_histogram = np.zeros(n_bins)

    # Vectorized volume distribution
    # For bars that span single bins
    single_bin_mask = start_bins == end_bins
    if np.any(single_bin_mask):
        single_bin_indices = start_bins[single_bin_mask]
        single_bin_volumes = volume[single_bin_mask]
        np.add.at(volume_histogram, single_bin_indices, single_bin_volumes)

    # For bars that span multiple bins
    multi_bin_mask = start_bins < end_bins
    if np.any(multi_bin_mask):
        multi_start = start_bins[multi_bin_mask]
        multi_end = end_bins[multi_bin_mask]
        multi_volumes = volume[multi_bin_mask]

        # Calculate number of bins spanned for each bar
        bins_spanned = multi_end - multi_start + 1
        volume_per_bin = multi_volumes / bins_spanned

        # Create indices for all bin assignments
        # This is a vectorized way to distribute volume across multiple bins
        max_bins_spanned = np.max(bins_spanned)
        bin_indices = np.arange(max_bins_spanned)[:, None] + multi_start[None, :]

        # Create mask for valid bin indices
        valid_mask = bin_indices <= multi_end[None, :]
        bin_indices = bin_indices[valid_mask]

        # Create corresponding volumes
        bar_indices = np.arange(len(multi_start))
        bar_repeated = np.repeat(bar_indices, bins_spanned)
        volumes_repeated = volume_per_bin[bar_repeated]

        # Add volumes to histogram
        np.add.at(volume_histogram, bin_indices, volumes_repeated)

    # Find Point of Control (POC) - price level with highest volume
    if np.sum(volume_histogram) == 0:
        poc = current_price
        vah = current_price
        val = current_price
        current_dist = 0.0
        full_dist = np.zeros_like(volume_histogram)
    else:
        poc_idx = np.argmax(volume_histogram)
        poc = bin_edges[poc_idx] + bin_width / 2  # Center of the bin

        # Calculate Value Area (VAH/VAL) - price levels containing X% of volume
        total_volume = np.sum(volume_histogram)
        target_volume = total_volume * value_area_percent

        # Sort bins by volume (descending)
        sorted_indices = np.argsort(volume_histogram)[::-1]
        cumulative_volume = 0.0
        value_area_bins = []

        for idx in sorted_indices:
            cumulative_volume += volume_histogram[idx]
            value_area_bins.append(idx)
            if cumulative_volume >= target_volume:
                break

        # Value Area High and Low
        vah = float(bin_edges[max(value_area_bins)] + bin_width / 2)
        val = float(bin_edges[min(value_area_bins)] + bin_width / 2)

        # Volume distribution at current price
        current_bin = int(np.floor((current_price - bin_edges[0]) / bin_width))
        current_bin = max(0, min(current_bin, n_bins - 1))
        current_dist = float(volume_histogram[current_bin] / total_volume)
        # Normalized full distribution
        full_dist = volume_histogram / total_volume

    return poc, vah, val, current_dist, full_dist


def volume_profile_feature_summary() -> dict[str, str]:
    """
    Returns a dictionary summarizing each Volume Profile feature.
    """
    return {
        "volume_profile_poc": "Point of Control - price level with highest volume",
        "volume_profile_vah": "Value Area High - upper bound of 70% volume concentration",
        "volume_profile_val": "Value Area Low - lower bound of 70% volume concentration",
        "volume_profile_distribution": "Normalized volume at current price level",
    }


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 500

    # Generate test OHLCV data
    test_data = pd.DataFrame(
        {
            "high": np.random.uniform(100, 200, n),
            "low": np.random.uniform(50, 150, n),
            "close": np.random.uniform(75, 175, n),
            "volume": np.random.uniform(1000, 10000, n),
        }
    )

    # Ensure high >= close >= low
    test_data["high"] = np.maximum(test_data["high"], test_data["close"])
    test_data["low"] = np.minimum(test_data["low"], test_data["close"])
    # Ensure low <= close after high is adjusted
    # Calculate features
    features = calculate_volume_profile_extended(test_data)

    print("Volume Profile Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    # NaN values may occur if there is insufficient data in the rolling window or due to edge cases in the calculation.
    print(f"NaN count per column:\n{features.isnull().sum()}")

    # Summary
    summary = volume_profile_feature_summary()
    print(f"\nFeature summary:\n{summary}")
    summary = volume_profile_feature_summary()
    print(f"\nFeature summary:\n{summary}")
