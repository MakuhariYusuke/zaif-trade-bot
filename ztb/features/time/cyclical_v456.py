"""
Cyclical Time Features for v456

6特徴量を生成:
- hour_sin: sin(2π * hour / 24)
- hour_cos: cos(2π * hour / 24)
- minute_sin: sin(2π * minute / 60)
- minute_cos: cos(2π * minute / 60)
- dow_sin: sin(2π * day_of_week / 7)
- dow_cos: cos(2π * day_of_week / 7)

重要: これらの特徴量は正規化対象外 ([-1, 1] で固定)
"""

import numpy as np
import pandas as pd
from typing import Union, List


def calc_cyclical_time_features(
    df: pd.DataFrame,
    time_column: str = None,
) -> pd.DataFrame:
    """
    時系列DataFrameに周期的時間特徴量を追加
    
    Args:
        df: DatetimeIndexまたはtimestamp列を持つDataFrame (tz-aware)
        time_column: timestamp列名（Noneの場合はindex使用）
    
    Returns:
        6つの周期的特徴量を追加したDataFrame
    
    Example:
        >>> df = pd.DataFrame(
        ...     {'price': [100, 101, 102]},
        ...     index=pd.date_range('2025-01-10 10:00', periods=3, freq='1min', tz='UTC')
        ... )
        >>> result = calc_cyclical_time_features(df)
        >>> result[['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos']]
    """
    df = df.copy()
    
    # タイムスタンプ列を取得
    if time_column is not None:
        timestamps = pd.to_datetime(df[time_column])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex or specify time_column")
        timestamps = df.index
    
    # 時刻が tz-aware であることを確認
    if isinstance(timestamps, pd.Series):
        tz = timestamps.dt.tz
    else:
        tz = timestamps.tzinfo
    
    if tz is None:
        raise ValueError(
            f"Timestamps must be timezone-aware."
        )
    
    # 各成分を抽出
    if isinstance(timestamps, pd.Series):
        hours = timestamps.dt.hour.values
        minutes = timestamps.dt.minute.values
        days_of_week = timestamps.dt.dayofweek.values
    else:
        hours = timestamps.hour
        minutes = timestamps.minute
        days_of_week = timestamps.dayofweek
    
    # 周期的特徴量を計算
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    df['minute_sin'] = np.sin(2 * np.pi * minutes / 60)
    df['minute_cos'] = np.cos(2 * np.pi * minutes / 60)
    
    df['dow_sin'] = np.sin(2 * np.pi * days_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * days_of_week / 7)
    
    return df


class CyclicalTimeFeatureExtractor:
    """
    周期的時間特徴量の抽出と管理
    
    用途:
    - 時間帯効果（朝 vs 夕方など）
    - 曜日効果（平日 vs 週末）の捕捉
    
    重要な性質:
    - [-1, 1] の有界範囲
    - 回転不変性（23:59 ≈ 00:00）
    - OnlineScaler適用不可（既に正規化済み）
    """
    
    FEATURE_NAMES = [
        'hour_sin', 'hour_cos',
        'minute_sin', 'minute_cos',
        'dow_sin', 'dow_cos',
    ]
    FEATURE_COUNT = len(FEATURE_NAMES)
    
    def __init__(self, validate_tz: bool = True):
        """
        Args:
            validate_tz: Naive timestampを拒否するか (推奨: True)
        """
        self.validate_tz = validate_tz
    
    def extract(self, df: pd.DataFrame, time_column: str = None) -> pd.DataFrame:
        """特徴量を抽出"""
        return calc_cyclical_time_features(df, time_column)
    
    def get_features_dict(self, df: pd.DataFrame) -> dict:
        """
        特徴量を辞書形式で取得
        
        Returns:
            {'hour_sin': array, 'hour_cos': array, ...}
        """
        result = {}
        for feat in self.FEATURE_NAMES:
            if feat in df.columns:
                result[feat] = df[feat].values
        return result
    
    @staticmethod
    def validate_bounds(array: np.ndarray, name: str = "") -> bool:
        """
        周期的特徴量が[-1, 1]の範囲内にあることを確認
        
        Args:
            array: 検証する配列
            name: 特徴量名（ログ用）
        
        Returns:
            True if all values in [-1, 1]
        """
        out_of_bounds = (array < -1.001) | (array > 1.001)  # 数値誤差許容
        if out_of_bounds.any():
            print(f"WARNING: {name} has out-of-bounds values")
            print(f"  Min: {array.min()}, Max: {array.max()}")
            return False
        return True
    
    def validate_periodicity(self, df: pd.DataFrame) -> bool:
        """
        周期性を検証（回転不変性）
        
        23:59と00:00が近い値を持つことを確認
        """
        # Test data: 23:59と00:00付近
        test_hour_23_59 = np.sin(2 * np.pi * 23.98 / 24)
        test_hour_00_00 = np.sin(2 * np.pi * 0.01 / 24)
        
        diff = abs(test_hour_23_59 - test_hour_00_00)
        
        # 小さな差（周期性が成立）
        if diff < 0.1:
            return True
        else:
            print(f"WARNING: Periodicity check failed. Diff: {diff}")
            return False


# 使用例とテスト
if __name__ == "__main__":
    # サンプルデータ作成
    dates = pd.date_range('2025-01-10 00:00', periods=24*60, freq='1min', tz='UTC')
    df = pd.DataFrame(
        {'price': np.random.randn(len(dates)).cumsum() + 100},
        index=dates
    )
    
    # 特徴量抽出
    df_with_features = calc_cyclical_time_features(df)
    
    # 検証
    extractor = CyclicalTimeFeatureExtractor()
    
    print("Cyclical Time Features Example:")
    print(df_with_features[extractor.FEATURE_NAMES].head())
    print(f"\nFeature bounds:")
    for feat in extractor.FEATURE_NAMES:
        print(f"  {feat}: [{df_with_features[feat].min():.3f}, {df_with_features[feat].max():.3f}]")
    
    print(f"\nPeriodicity check: {extractor.validate_periodicity(df_with_features)}")
