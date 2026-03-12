"""
Tests for Global Market Features v456

テスト対象:
- 9特徴量生成 (6連続 + 3フラグ)
- スプレッド計算とフラグ
- リターン相関計算
- ボラティリティ計算と比率
- データ鮮度判定
- スタール処理
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

from ztb.features.global_market_v456 import GlobalMarketFeatureEngineerV456


@pytest.fixture
def utc_tz():
    return pytz.UTC


@pytest.fixture
def jst_tz():
    return pytz.timezone('Asia/Tokyo')


@pytest.fixture
def sample_local_df(utc_tz):
    """Zaif (JPY) サンプルデータ"""
    dates = pd.date_range('2025-01-10 10:00', periods=30, freq='1min', tz=utc_tz)
    
    # BTCレートの模擬値 (JPY) 9000～9100 で変動
    np.random.seed(42)
    prices = 9000 + np.cumsum(np.random.randn(30) * 2)
    
    return pd.DataFrame({
        'open': prices + np.random.randn(30) * 0.5,
        'high': prices + np.abs(np.random.randn(30)) + 2,
        'low': prices - np.abs(np.random.randn(30)) - 2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 30),
    }, index=dates)


@pytest.fixture
def sample_global_df(utc_tz):
    """Binance (USD) サンプルデータ"""
    dates = pd.date_range('2025-01-10 10:00', periods=30, freq='1min', tz=utc_tz)
    
    # BTC USD相場 (58000～58100)
    np.random.seed(43)
    prices = 58000 + np.cumsum(np.random.randn(30) * 3)
    
    return pd.DataFrame({
        'open': prices + np.random.randn(30) * 1,
        'high': prices + np.abs(np.random.randn(30)) + 5,
        'low': prices - np.abs(np.random.randn(30)) - 5,
        'close': prices,
        'volume': np.random.randint(10000, 100000, 30),
    }, index=dates)


class TestGlobalMarketFeatureGeneration:
    """グローバル市場特徴量の生成テスト"""
    
    def test_feature_count_is_nine(self, sample_local_df, sample_global_df):
        """特徴量数が9であることを確認"""
        engineer = GlobalMarketFeatureEngineerV456(binance_df=sample_global_df)
        
        current = sample_local_df.index[-1]
        features = engineer.generate_features(sample_local_df, current)
        
        assert features.shape == (9,)
        assert features.dtype == np.float32
    
    def test_feature_names_count(self):
        """特徴量名の数が正しい"""
        assert len(GlobalMarketFeatureEngineerV456.FEATURE_NAMES) == 9
        assert len(GlobalMarketFeatureEngineerV456.FEATURE_NAMES_CONTINUOUS) == 6
        assert len(GlobalMarketFeatureEngineerV456.FEATURE_NAMES_FLAGS) == 3
    
    def test_continuous_features_in_valid_range(self, sample_local_df, sample_global_df):
        """連続値特徴量が適切な範囲内"""
        engineer = GlobalMarketFeatureEngineerV456(binance_df=sample_global_df)
        
        current = sample_local_df.index[-1]
        features = engineer.generate_features(sample_local_df, current)
        
        # spread (idx 0): [-1000, 1000] bps
        assert -1000 <= features[0] <= 1000
        
        # return_1m (idx 1): [-100, 100] %
        assert -100 <= features[1] <= 100
        
        # return_5m (idx 2): [-100, 100] %
        assert -100 <= features[2] <= 100
        
        # vol_1m (idx 3): [0, 10] %
        assert 0 <= features[3] <= 10
        
        # vol_ratio (idx 4): [0.1, 10]
        assert 0.1 <= features[4] <= 10.0
        
        # usdt_premium (idx 5): [-10, 10] %
        assert -10 <= features[5] <= 10
    
    def test_flag_features_are_binary(self, sample_local_df, sample_global_df):
        """フラグ特徴量が 0 or 1"""
        engineer = GlobalMarketFeatureEngineerV456(binance_df=sample_global_df)
        
        current = sample_local_df.index[-1]
        features = engineer.generate_features(sample_local_df, current)
        
        # flag_spread (idx 6)
        assert features[6] in [0.0, 1.0]
        
        # flag_return (idx 7)
        assert features[7] in [0.0, 1.0]
        
        # stale_flag (idx 8)
        assert features[8] in [0.0, 1.0]
    
    def test_empty_dataframe_returns_zeros(self, sample_global_df):
        """空のDataFrameに対して0を返す"""
        engineer = GlobalMarketFeatureEngineerV456(binance_df=sample_global_df)
        
        empty_df = pd.DataFrame({
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
        }, index=pd.DatetimeIndex([]))
        
        current = pd.Timestamp('2025-01-10 10:30', tz='UTC')
        features = engineer.generate_features(empty_df, current)
        
        # spread と stale_flag 以外は0
        assert features[0] == 0.0  # spread
        assert features[8] == 1.0  # stale_flag (empty = stale)


class TestSpreadCalculation:
    """スプレッド計算のテスト"""
    
    def test_spread_calculation_usd_adjustment(self, utc_tz):
        """USD/JPY調整でのスプレッド計算"""
        # ローカル: JPY, グローバル: USD
        local_df = pd.DataFrame({
            'open': [9000], 'high': [9100], 'low': [8900], 'close': [9000], 'volume': [1000]
        }, index=pd.DatetimeIndex(['2025-01-10 10:00'], tz=utc_tz))
        
        # Binance: 58000 USD
        global_df = pd.DataFrame({
            'open': [58000], 'high': [58100], 'low': [57900], 'close': [58000], 'volume': [10000]
        }, index=pd.DatetimeIndex(['2025-01-10 10:00'], tz=utc_tz))
        
        engineer = GlobalMarketFeatureEngineerV456(binance_df=global_df, usdjpy_rate=155.0)
        
        spread_bps = engineer._compute_spread(local_df, local_df.index[-1])
        
        # local_price = 9000 JPY = 9000/155 USD ≈ 58.06 USD
        # global_price = 58000 USD
        # spread = (58.06 - 58000) / 58000 * 10000 ≈ -1000 bps (かなり逆ざや)
        
        assert isinstance(spread_bps, float)
        assert -1000 <= spread_bps <= 1000
    
    def test_spread_flag_threshold(self):
        """スプレッドフラグが50bpsで立つ"""
        engineer = GlobalMarketFeatureEngineerV456()
        
        # spread < 50 bps
        assert engineer._compute_spread_flag(30.0) is False
        
        # spread > 50 bps
        assert engineer._compute_spread_flag(60.0) is True
        
        # negative spread
        assert engineer._compute_spread_flag(-60.0) is True


class TestReturnCalculation:
    """リターン計算のテスト"""
    
    def test_return_calculation_1m(self, utc_tz):
        """1分リターン計算"""
        dates = pd.date_range('2025-01-10 10:00', periods=2, freq='1min', tz=utc_tz)
        
        # 価格が 100 → 102 (2%増加)
        df = pd.DataFrame({
            'close': [100.0, 102.0],
            'open': [100.0, 102.0],
            'high': [100.5, 102.5],
            'low': [99.5, 101.5],
            'volume': [1000, 1000],
        }, index=dates)
        
        engineer = GlobalMarketFeatureEngineerV456()
        ret = engineer._compute_return(df, df.index[-1], window=1)
        
        assert isinstance(ret, float)
        assert pytest.approx(ret, rel=0.05) == 2.0
    
    def test_return_flag_condition(self):
        """リターンフラグが条件で立つ"""
        engineer = GlobalMarketFeatureEngineerV456()
        
        # return_5m > 1% and return_1m < 0
        assert engineer._compute_return_flag(-0.5, 1.5) is True
        
        # return_5m <= 1%
        assert engineer._compute_return_flag(-0.5, 0.8) is False
        
        # return_1m >= 0
        assert engineer._compute_return_flag(0.2, 1.5) is False


class TestVolatilityCalculation:
    """ボラティリティ計算のテスト"""
    
    def test_volatility_from_atr(self, utc_tz):
        """ATRベースのボラティリティ計算"""
        dates = pd.date_range('2025-01-10 10:00', periods=10, freq='1min', tz=utc_tz)
        
        # 価格が100の周りで ±5の幅で変動
        df = pd.DataFrame({
            'open': np.full(10, 100.0),
            'high': np.full(10, 105.0),
            'low': np.full(10, 95.0),
            'close': np.full(10, 100.0),
            'volume': np.full(10, 1000),
        }, index=dates)
        
        engineer = GlobalMarketFeatureEngineerV456()
        vol = engineer._compute_volatility(df, df.index[-1], window=5)
        
        # (105 - 95) / 100 * 100 = 10%
        assert pytest.approx(vol, rel=0.1) == 10.0
    
    def test_vol_ratio_calculation(self, utc_tz):
        """ボラティリティ比の計算"""
        dates = pd.date_range('2025-01-10 10:00', periods=10, freq='1min', tz=utc_tz)
        
        # ローカル: 高ボラティリティ (±10)
        local_df = pd.DataFrame({
            'open': np.full(10, 100.0),
            'high': np.full(10, 110.0),
            'low': np.full(10, 90.0),
            'close': np.full(10, 100.0),
            'volume': np.full(10, 1000),
        }, index=dates)
        
        # グローバル: 低ボラティリティ (±2)
        global_df = pd.DataFrame({
            'open': np.full(10, 58000.0),
            'high': np.full(10, 58100.0),
            'low': np.full(10, 57900.0),
            'close': np.full(10, 58000.0),
            'volume': np.full(10, 10000),
        }, index=dates)
        
        engineer = GlobalMarketFeatureEngineerV456(binance_df=global_df)
        vol_ratio = engineer._compute_vol_ratio(local_df, local_df.index[-1])
        
        # local_vol / global_vol > 1
        assert vol_ratio >= 1.0


class TestDataStalenessDetection:
    """データ鮮度の検出テスト"""
    
    def test_stale_flag_old_data(self, utc_tz):
        """古いデータでスタールフラグが立つ"""
        # 10分前のデータ
        dates = pd.date_range('2025-01-10 10:00', periods=5, freq='1min', tz=utc_tz)
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5,
        }, index=dates)
        
        engineer = GlobalMarketFeatureEngineerV456(max_data_age_minutes=5)
        
        # 現在時刻が10分後
        current = dates[-1] + timedelta(minutes=11)
        
        stale = engineer._compute_stale_flag(df, current)
        assert stale is True
    
    def test_stale_flag_fresh_data(self, utc_tz):
        """新しいデータではスタールフラグが立たない"""
        dates = pd.date_range('2025-01-10 10:00', periods=5, freq='1min', tz=utc_tz)
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5,
        }, index=dates)
        
        engineer = GlobalMarketFeatureEngineerV456(max_data_age_minutes=5)
        
        # 現在時刻が3分後
        current = dates[-1] + timedelta(minutes=3)
        
        stale = engineer._compute_stale_flag(df, current)
        assert stale is False
    
    def test_stale_flag_naive_timestamp_raises(self, utc_tz):
        """Naiveタイムスタンプでスタールフラグが立つ"""
        # UTC aware index
        dates = pd.date_range('2025-01-10 10:00', periods=5, freq='1min', tz=utc_tz)
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5,
        }, index=dates)
        
        engineer = GlobalMarketFeatureEngineerV456()
        
        # Naive current_timestamp
        current = pd.Timestamp('2025-01-10 10:05')
        
        stale = engineer._compute_stale_flag(df, current)
        # tz-aware チェックで True（stale）を返す
        assert stale is True


class TestStaleFeatureHandling:
    """陳腐な特徴量の処理テスト"""
    
    def test_handle_stale_zeros_out_continuous(self):
        """スタールフラグが立つと連続値が0になる"""
        engineer = GlobalMarketFeatureEngineerV456()
        
        # 全て非零のサンプル特徴量
        features = np.array([50.0, 0.5, 1.0, 2.0, 1.5, 0.1, 0.0, 0.0, 1.0], dtype=np.float32)
        
        # stale_flag (idx 8) = 1
        features_after = engineer.handle_stale_global_features(features)
        
        # 連続値がゼロ化
        assert np.all(features_after[0:6] == 0.0)
        
        # フラグはそのまま
        assert features_after[6] == 0.0
        assert features_after[7] == 0.0
        assert features_after[8] == 1.0
    
    def test_handle_stale_preserves_fresh(self):
        """スタールフラグが立たないと値が変わらない"""
        engineer = GlobalMarketFeatureEngineerV456()
        
        features = np.array([50.0, 0.5, 1.0, 2.0, 1.5, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        features_orig = features.copy()
        
        # stale_flag (idx 8) = 0
        features_after = engineer.handle_stale_global_features(features)
        
        np.testing.assert_array_equal(features_after, features_orig)


class TestValidation:
    """検証メソッドのテスト"""
    
    def test_validate_feature_count_passes(self):
        """特徴量数検証が成功"""
        assert GlobalMarketFeatureEngineerV456.validate_feature_count() is True
    
    def test_feature_names_completeness(self):
        """特徴量名が完全"""
        assert len(GlobalMarketFeatureEngineerV456.FEATURE_NAMES_CONTINUOUS) == 6
        assert len(GlobalMarketFeatureEngineerV456.FEATURE_NAMES_FLAGS) == 3
        assert len(GlobalMarketFeatureEngineerV456.FEATURE_NAMES) == 9
        
        # ユニーク確認
        assert len(set(GlobalMarketFeatureEngineerV456.FEATURE_NAMES)) == 9


class TestIntegration:
    """統合テスト"""
    
    def test_end_to_end_generation(self, sample_local_df, sample_global_df):
        """エンドツーエンドの特徴量生成"""
        engineer = GlobalMarketFeatureEngineerV456(binance_df=sample_global_df)
        
        current = sample_local_df.index[-1]
        features = engineer.generate_features(sample_local_df, current)
        
        # 正しい形状
        assert features.shape == (9,)
        
        # すべて有限値
        assert np.all(np.isfinite(features))
        
        # 連続値とフラグの構造が正しい
        assert isinstance(features[0], (float, np.floating))  # spread
        assert features[6] in [0.0, 1.0]  # flag
        assert features[8] in [0.0, 1.0]  # stale_flag
    
    def test_end_to_end_with_stale_handling(self, sample_local_df, sample_global_df, utc_tz):
        """スタール処理を含むエンドツーエンド"""
        engineer = GlobalMarketFeatureEngineerV456(
            binance_df=sample_global_df,
            max_data_age_minutes=1  # 短い閾値
        )
        
        current = sample_local_df.index[-1] + timedelta(minutes=10)  # 10分後
        features = engineer.generate_features(sample_local_df, current)
        
        # stale_flag が立っているはず
        assert features[8] == 1.0
        
        # スタール処理後
        features_handled = engineer.handle_stale_global_features(features)
        
        # 連続値がゼロ
        assert np.all(features_handled[0:6] == 0.0)


# マーク
@pytest.mark.parametrize("feature_idx,expected_min,expected_max", [
    (0, -1000, 1000),      # spread
    (1, -100, 100),        # return_1m
    (2, -100, 100),        # return_5m
    (3, 0, 10),            # vol_1m
    (4, 0.1, 10),          # vol_ratio
    (5, -10, 10),          # usdt_premium
])
def test_continuous_feature_bounds(feature_idx, expected_min, expected_max, sample_local_df, sample_global_df):
    """各連続値特徴量が指定範囲内"""
    engineer = GlobalMarketFeatureEngineerV456(binance_df=sample_global_df)
    features = engineer.generate_features(sample_local_df, sample_local_df.index[-1])
    
    assert expected_min <= features[feature_idx] <= expected_max
