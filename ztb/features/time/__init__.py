"""
Time-based features.
時間ベースの特徴量
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("DOW")
def compute_dow(df: pd.DataFrame) -> pd.Series:
    """Day of Week"""
    if 'ts' in df.columns:
        dt = pd.to_datetime(df['ts'], unit='s')
        dow = dt.dt.dayofweek  # 0=Monday, 6=Sunday
    else:
        dow = pd.Series([0] * len(df))  # デフォルト月曜日
    return pd.Series(dow, name='DOW', index=df.index)


@FeatureRegistry.register("HourOfDay")
def compute_hour_of_day(df: pd.DataFrame) -> pd.Series:
    """Hour of Day"""
    if 'ts' in df.columns:
        dt = pd.to_datetime(df['ts'], unit='s')
        hour = dt.dt.hour
    else:
        hour = pd.Series([12] * len(df))  # デフォルト正午
    return pd.Series(hour, name='HourOfDay', index=df.index)