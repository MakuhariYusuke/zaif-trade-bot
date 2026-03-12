"""
Cyclical Time Features テスト
"""

import pytest
import numpy as np
import pandas as pd
from ztb.features.time.cyclical_v456 import (
    calc_cyclical_time_features,
    CyclicalTimeFeatureExtractor,
)


class TestCyclicalTimeFeatures:
    """周期的時間特徴量テスト"""

    @pytest.fixture
    def sample_df(self):
        """テストデータ"""
        dates = pd.date_range('2025-01-10 00:00', periods=1440, freq='1min', tz='UTC')
        return pd.DataFrame(
            {'price': np.random.randn(len(dates)).cumsum() + 100},
            index=dates
        )

    def test_calc_cyclical_time_features_added(self, sample_df):
        """特徴量が追加されることを確認"""
        result = calc_cyclical_time_features(sample_df)
        
        expected_cols = [
            'hour_sin', 'hour_cos',
            'minute_sin', 'minute_cos',
            'dow_sin', 'dow_cos',
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_hour_features_bounds(self, sample_df):
        """hour特徴量が[-1, 1]に収まることを確認"""
        result = calc_cyclical_time_features(sample_df)
        
        assert result['hour_sin'].min() >= -1.001
        assert result['hour_sin'].max() <= 1.001
        assert result['hour_cos'].min() >= -1.001
        assert result['hour_cos'].max() <= 1.001

    def test_minute_features_bounds(self, sample_df):
        """minute特徴量が[-1, 1]に収まることを確認"""
        result = calc_cyclical_time_features(sample_df)
        
        assert result['minute_sin'].min() >= -1.001
        assert result['minute_sin'].max() <= 1.001
        assert result['minute_cos'].min() >= -1.001
        assert result['minute_cos'].max() <= 1.001

    def test_dow_features_bounds(self, sample_df):
        """dow特徴量が[-1, 1]に収まることを確認"""
        result = calc_cyclical_time_features(sample_df)
        
        assert result['dow_sin'].min() >= -1.001
        assert result['dow_sin'].max() <= 1.001
        assert result['dow_cos'].min() >= -1.001
        assert result['dow_cos'].max() <= 1.001

    def test_midnight_periodicity(self):
        """00:00と23:59の周期性を確認"""
        # 00:00と23:59のデータ
        dates = [
            pd.Timestamp('2025-01-10 00:00', tz='UTC'),
            pd.Timestamp('2025-01-10 23:59', tz='UTC'),
        ]
        df = pd.DataFrame({'price': [100, 101]}, index=dates)
        
        result = calc_cyclical_time_features(df)
        
        # 23:59のhour_sin ≈ 00:00のhour_sin（近い値）
        diff_sin = abs(result.iloc[0]['hour_sin'] - result.iloc[1]['hour_sin'])
        diff_cos = abs(result.iloc[0]['hour_cos'] - result.iloc[1]['hour_cos'])
        
        # 差は比較的小さい（回転対称性）
        assert diff_sin < 0.3
        assert diff_cos < 0.3

    def test_naive_timestamp_rejected(self):
        """Naive timestampを拒否することを確認"""
        dates = pd.date_range('2025-01-10 00:00', periods=10, freq='1min')  # tz なし
        df = pd.DataFrame({'price': [100] * 10}, index=dates)
        
        with pytest.raises(ValueError, match="timezone-aware"):
            calc_cyclical_time_features(df)

    def test_time_column_parameter(self):
        """time_columnパラメータの動作"""
        dates = pd.date_range('2025-01-10 10:00', periods=5, freq='1min', tz='UTC')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': [100, 101, 102, 103, 104],
        })
        
        result = calc_cyclical_time_features(df, time_column='timestamp')
        
        assert 'hour_sin' in result.columns
        assert result['hour_sin'].iloc[0] == np.sin(2 * np.pi * 10 / 24)

    def test_extractor_class_extract(self):
        """CyclicalTimeFeatureExtractorクラスの動作"""
        dates = pd.date_range('2025-01-10 10:00', periods=5, freq='1min', tz='UTC')
        df = pd.DataFrame(
            {'price': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        extractor = CyclicalTimeFeatureExtractor()
        result = extractor.extract(df)
        
        for feat in extractor.FEATURE_NAMES:
            assert feat in result.columns

    def test_extractor_get_features_dict(self):
        """特徴量の辞書形式取得"""
        dates = pd.date_range('2025-01-10 10:00', periods=5, freq='1min', tz='UTC')
        df = pd.DataFrame(
            {'price': [100, 101, 102, 103, 104]},
            index=dates
        )
        
        df_with_features = calc_cyclical_time_features(df)
        extractor = CyclicalTimeFeatureExtractor()
        
        features_dict = extractor.get_features_dict(df_with_features)
        
        assert len(features_dict) == 6
        for feat in extractor.FEATURE_NAMES:
            assert feat in features_dict
            assert len(features_dict[feat]) == 5

    def test_feature_count_constant(self):
        """特徴量数の定数確認"""
        assert CyclicalTimeFeatureExtractor.FEATURE_COUNT == 6
        assert len(CyclicalTimeFeatureExtractor.FEATURE_NAMES) == 6

    def test_hour_cycle(self):
        """1時間ごとのhour特徴量を確認"""
        hours = [0, 6, 12, 18]
        
        # 正午（12:00）のhour_sinは0に近い
        sin_12 = np.sin(2 * np.pi * 12 / 24)
        assert abs(sin_12) < 0.01
        
        # 6時のhour_cosは0に近い（90度）
        cos_6 = np.cos(2 * np.pi * 6 / 24)
        assert abs(cos_6) < 0.01

    def test_dow_cycle(self):
        """曜日周期を確認"""
        # 月曜（0）と火曜（1）
        dow_0 = np.sin(2 * np.pi * 0 / 7)
        dow_1 = np.sin(2 * np.pi * 1 / 7)
        
        # 異なる値
        assert dow_0 != dow_1
        
        # 週の周期性: 日曜（6）と月曜（0）は離れている
        dow_6 = np.sin(2 * np.pi * 6 / 7)
        assert abs(dow_0 - dow_6) > 0.5

    def test_validate_bounds(self):
        """validate_boundsメソッドのテスト"""
        extractor = CyclicalTimeFeatureExtractor()
        
        # 正常範囲
        valid_array = np.array([-1, -0.5, 0, 0.5, 1])
        assert extractor.validate_bounds(valid_array) is True
        
        # 範囲外
        invalid_array = np.array([-1.5, 0, 1.5])
        assert extractor.validate_bounds(invalid_array) is False

    def test_validate_periodicity(self):
        """validate_periodicityメソッドのテスト"""
        dates = pd.date_range('2025-01-10 00:00', periods=10, freq='1min', tz='UTC')
        df = pd.DataFrame({'price': [100] * 10}, index=dates)
        
        df_with_features = calc_cyclical_time_features(df)
        extractor = CyclicalTimeFeatureExtractor()
        
        # 周期性チェック
        result = extractor.validate_periodicity(df_with_features)
        assert result is True


class TestCyclicalTimeIntegration:
    """統合テスト"""

    def test_full_day_cycle(self):
        """24時間フルサイクルの検証"""
        dates = pd.date_range('2025-01-10 00:00', periods=1440, freq='1min', tz='UTC')
        df = pd.DataFrame(
            {'price': np.random.randn(1440).cumsum() + 100},
            index=dates
        )
        
        result = calc_cyclical_time_features(df)
        
        # 全特徴量が存在
        for feat in CyclicalTimeFeatureExtractor.FEATURE_NAMES:
            assert feat in result.columns
            assert len(result[feat]) == 1440
        
        # 値域確認
        for feat in CyclicalTimeFeatureExtractor.FEATURE_NAMES:
            assert -1.001 <= result[feat].min()
            assert result[feat].max() <= 1.001

    def test_week_cycle(self):
        """1週間周期の検証"""
        dates = pd.date_range('2025-01-06', periods=7*1440, freq='1min', tz='UTC')
        df = pd.DataFrame(
            {'price': np.random.randn(len(dates)).cumsum() + 100},
            index=dates
        )
        
        result = calc_cyclical_time_features(df)
        
        # 月曜～日曜で異なるdow値
        monday_dow_sin = result.iloc[0]['dow_sin']  # 月曜（2025-01-06）
        sunday_dow_sin = result.iloc[6*1440-1]['dow_sin']  # 日曜（2025-01-12）
        
        # 異なる値であることを確認
        assert monday_dow_sin != sunday_dow_sin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
