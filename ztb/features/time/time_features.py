"""
time_features.py
Time-based features for intraday and interday analysis
"""

import numpy as np
import pandas as pd
from typing import Union

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Time_Session")
def compute_time_session(df: pd.DataFrame) -> pd.Series:
    """Market Session (0=pre-market, 1=regular, 2=after-hours)"""
    extended_features = calculate_time_features_extended(df)
    return extended_features["time_session"]


@FeatureRegistry.register("Time_Day_of_Week")
def compute_time_day_of_week(df: pd.DataFrame) -> pd.Series:
    """Day of Week (0=Monday, 6=Sunday)"""
    extended_features = calculate_time_features_extended(df)
    return extended_features["time_day_of_week"]


@FeatureRegistry.register("Time_Hour_of_Day")
def compute_time_hour_of_day(df: pd.DataFrame) -> pd.Series:
    """Hour of Day (0-23)"""
    extended_features = calculate_time_features_extended(df)
    return extended_features["time_hour_of_day"]


@FeatureRegistry.register("Time_Volatility_Adjustment")
def compute_time_volatility_adjustment(df: pd.DataFrame) -> pd.Series:
    """Time-based Volatility Adjustment Factor"""
    extended_features = calculate_time_features_extended(df)
    return extended_features["time_volatility_adjustment"]


def calculate_time_features_extended(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate extended time-based features including:
    - Market session identification
    - Day of week effects
    - Hour of day patterns
    - Time-based volatility adjustments

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with time-based features
    """
    result = pd.DataFrame(index=df.index)

    # Extract time components
    datetime_index: Union[pd.DatetimeIndex, pd.Series]
    if isinstance(df.index, pd.DatetimeIndex):
        datetime_index = df.index
    elif "timestamp" in df.columns:
        datetime_index = pd.to_datetime(df["timestamp"])
    else:
        # Create synthetic time patterns
        n = len(df)
        # Create synthetic datetime index (daily data)
        base_date = pd.Timestamp("2023-01-01")
        datetime_index = pd.date_range(
            base_date, periods=min(n, 365 * 10), freq="D"
        )  # Limit to 10 years

    result["time_day_of_week"] = (
        datetime_index.dayofweek
        if hasattr(datetime_index, "dayofweek")
        else datetime_index.dayofweek
    )  # 0=Monday, 6=Sunday
    result["time_hour_of_day"] = datetime_index.hour  # 0-23

    # Market session identification (assuming 9:30-16:00 is regular hours)
    # For crypto/stocks, adjust these hours as needed
    if isinstance(datetime_index, pd.DatetimeIndex):
        minutes_since_midnight = result["time_hour_of_day"] * 60 + datetime_index.minute
    else:
        minutes_since_midnight = result["time_hour_of_day"] * 60 + datetime_index.dt.minute

    # Define session boundaries (adjust for your market)
    pre_market_start = 0  # 00:00
    regular_start = 9 * 60 + 30  # 09:30
    regular_end = 16 * 60  # 16:00

    # Vectorized session identification using np.select for better performance
    result["time_session"] = np.select(
        [
            (minutes_since_midnight >= pre_market_start)
            & (minutes_since_midnight < regular_start),
            (minutes_since_midnight >= regular_start)
            & (minutes_since_midnight <= regular_end),
        ],
        [0, 1],  # Pre-market, Regular hours
        default=2,  # After hours
    )

    # Time-based volatility adjustment
    # Higher volatility often occurs at market open/close and certain days
    day_volatility = np.where(
        result["time_day_of_week"].isin([0, 4]),  # Monday, Friday
        1.2,  # Higher volatility
        1.0,  # Normal volatility
    )

    hour_volatility = np.where(
        result["time_hour_of_day"].isin([9, 10, 15, 16]),  # Market open/close hours
        1.3,  # Higher volatility
        1.0,  # Normal volatility
    )

    result["time_volatility_adjustment"] = day_volatility * hour_volatility

    # Additional time features
    if isinstance(datetime_index, pd.DatetimeIndex):
        result["time_month"] = datetime_index.month  # 1-12
        result["time_quarter"] = datetime_index.quarter  # 1-4
    else:
        # For Series, use dt accessor
        result["time_month"] = datetime_index.dt.month  # 1-12
        result["time_quarter"] = datetime_index.dt.quarter  # 1-4
    result["time_is_weekend"] = result["time_day_of_week"].isin([5, 6]).astype(int)

    # Business day vs weekend
    result["time_is_business_day"] = (~result["time_day_of_week"].isin([5, 6])).astype(
        int
    )

    # Time since market open (normalized 0-1 within session)
    session_minutes = minutes_since_midnight - regular_start
    session_length = regular_end - regular_start
    result["time_session_progress"] = np.where(
        result["time_session"] == 1,  # Regular session
        np.clip(session_minutes / session_length, 0, 1),
        0,  # Outside regular session
    )

    return result


def time_feature_summary() -> dict[str, str]:
    """
    Returns a dictionary summarizing each time-based feature.
    """
    return {
        "time_session": "Market session (0=pre-market, 1=regular, 2=after-hours)",
        "time_day_of_week": "Day of week (0=Monday, 6=Sunday)",
        "time_hour_of_day": "Hour of day (0-23)",
        "time_month": "Month of year (1-12)",
        "time_quarter": "Quarter of year (1-4)",
        "time_is_weekend": "Is weekend (1=yes, 0=no)",
        "time_is_business_day": "Is business day (1=yes, 0=no)",
        "time_volatility_adjustment": "Time-based volatility multiplier",
        "time_session_progress": "Progress through trading session (0-1)",
    }


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 500

    # Create datetime index
    base_date = pd.Timestamp("2023-01-01")
    datetime_index = pd.date_range(base_date, periods=n, freq="H")  # Hourly data

    # Generate test OHLC data
    test_data = pd.DataFrame(
        {
            "high": np.random.uniform(100, 200, n),
            "low": np.random.uniform(50, 150, n),
            "close": np.random.uniform(75, 175, n),
            "volume": np.random.uniform(1000, 5000, n),
        },
        index=datetime_index,
    )

    # Ensure high >= close >= low
    test_data["high"] = np.maximum(test_data["high"], test_data["close"])
    test_data["low"] = np.minimum(test_data["low"], test_data["close"])

    # Calculate features
    features = calculate_time_features_extended(test_data)

    print("Time-based Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"NaN count per column:\n{features.isnull().sum()}")

    # Summary
    summary = time_feature_summary()
    print(f"\nFeature summary:\n{summary}")
