#!/usr/bin/env python3
"""
Test Bearish Signal Features

弱気シグナル特徴量の単体テスト
SELL bias対策として追加された特徴量の正確性を検証
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from ztb.features.unified_feature import UnifiedFeatureEngineer


class TestBearishSignalFeatures:
    """弱気シグナル特徴量のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを作成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')

        # 基本価格データ
        base_price = 100000
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        # ボラティリティのある価格変動
        high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates)),
        })

        # テクニカル指標の追加（モック）
        df['RSI'] = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # RSI-like oscillation
        df['MACD'] = np.sin(np.linspace(0, 2*np.pi, len(dates))) * 1000
        df['Stochastic'] = 50 + 40 * np.cos(np.linspace(0, 3*np.pi, len(dates)))
        df['Williams_R'] = -50 + 40 * np.sin(np.linspace(0, 2.5*np.pi, len(dates)))

        return df

    @pytest.fixture
    def feature_engineer(self):
        """UnifiedFeatureEngineerのインスタンス"""
        return UnifiedFeatureEngineer()

    def test_bearish_divergence_features(self, feature_engineer, sample_data):
        """ベアリッシュダイバージェンス特徴量のテスト"""
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        # RSIダイバージェンス特徴量が存在することを確認
        assert 'Bearish_RSI_Divergence' in features.columns
        assert 'Bearish_MACD_Divergence' in features.columns

        # 値が0または1であることを確認
        assert features['Bearish_RSI_Divergence'].isin([0, 1]).all()
        assert features['Bearish_MACD_Divergence'].isin([0, 1]).all()

    def test_bearish_candlestick_patterns(self, feature_engineer, sample_data):
        """弱気ローソク足パターンのテスト"""
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        # パターン特徴量が存在することを確認
        assert 'Bearish_Engulfing' in features.columns
        assert 'Shooting_Star' in features.columns
        assert 'Hammer_Bearish' in features.columns

        # 値が0または1であることを確認
        for col in ['Bearish_Engulfing', 'Shooting_Star', 'Hammer_Bearish']:
            assert features[col].isin([0, 1]).all()

    def test_bearish_momentum_features(self, feature_engineer, sample_data):
        """弱気モメンタム指標のテスト"""
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        # モメンタム特徴量が存在することを確認
        assert 'Bearish_Momentum_Acceleration' in features.columns
        assert 'Bearish_Trend_Continuation' in features.columns
        assert 'Bearish_Volatility_Surge' in features.columns

        # 値が適切な範囲であることを確認
        assert features['Bearish_Momentum_Acceleration'].isin([0, 1]).all()
        assert features['Bearish_Trend_Continuation'].isin([0, 1]).all()
        assert features['Bearish_Volatility_Surge'].dtype in [np.float64, np.float32]

    def test_sell_signal_oscillators(self, feature_engineer, sample_data):
        """売りシグナルオシレーターのテスト"""
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        # オシレーター特徴量が存在することを確認
        expected_cols = [
            'Stochastic_Overbought_Sell',
            'Stochastic_Bearish_Divergence',
            'RSI_Overbought_Sell',
            'RSI_Bearish_Divergence',
            'WilliamsR_Oversold_Sell'
        ]

        for col in expected_cols:
            assert col in features.columns
            assert features[col].isin([0, 1]).all()

    def test_bearish_volume_features(self, feature_engineer, sample_data):
        """弱気ボリューム指標のテスト"""
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        # ボリューム特徴量が存在することを確認
        assert 'Bearish_Volume_Surge' in features.columns
        assert 'Bearish_Volume_Trend' in features.columns

        # 値が0または1であることを確認
        assert features['Bearish_Volume_Surge'].isin([0, 1]).all()
        assert features['Bearish_Volume_Trend'].isin([0, 1]).all()

    def test_missing_columns_handling(self, feature_engineer):
        """必要な列が不足している場合の処理テスト"""
        # 必要な列が不足したデータ
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
            'some_other_column': range(10)
        })

        features = feature_engineer._generate_bearish_signal_features(incomplete_data)

        # 空のDataFrameが返されることを確認
        assert len(features.columns) == 0
        assert len(features) == 10

    def test_error_handling(self, feature_engineer, sample_data):
        """エラーハンドリングのテスト"""
        # モックを使って例外を発生させる
        with patch.object(feature_engineer, '_add_bearish_divergence_features', side_effect=Exception("Test error")):
            features = feature_engineer._generate_bearish_signal_features(sample_data)

            # エラーが発生しても空のDataFrameが返されることを確認
            assert isinstance(features, pd.DataFrame)
            assert len(features) == len(sample_data)

    def test_feature_count(self, feature_engineer, sample_data):
        """生成される特徴量数のテスト"""
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        # 少なくとも10個以上の特徴量が生成されることを確認
        assert len(features.columns) >= 10

        # すべての特徴量が数値型であることを確認
        for col in features.columns:
            assert features[col].dtype in [np.int64, np.float64, np.float32, np.int32]

    def test_integration_with_generic_features(self, feature_engineer, sample_data):
        """汎用特徴量生成との統合テスト"""
        # モックを使ってcompute_features_batchを回避し、concatされたDataFrameを返す
        with patch('ztb.features.unified_feature.compute_features_batch') as mock_compute:
            # concatされたDataFrameをモックで返す
            bearish_features = feature_engineer._generate_bearish_signal_features(sample_data)
            expected_result = pd.concat([sample_data, bearish_features], axis=1)
            mock_compute.return_value = expected_result

            result = feature_engineer._generate_generic_features(sample_data, "curated")

            # 弱気シグナル特徴量が追加されていることを確認
            bearish_cols = [col for col in result.columns if col.startswith('Bearish_') or 'Sell' in col or 'Divergence' in col]
            assert len(bearish_cols) > 0

            # 元のデータも保持されていることを確認
            assert 'close' in result.columns
            assert 'open' in result.columns
            assert 'high' in result.columns
            assert 'low' in result.columns
            assert 'volume' in result.columns

    def test_memory_efficiency(self, feature_engineer, sample_data):
        """メモリ効率のテスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 特徴量生成を実行
        features = feature_engineer._generate_bearish_signal_features(sample_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # メモリ増加が合理的な範囲内であることを確認（100MB以内）
        assert memory_increase < 100 * 1024 * 1024  # 100MB

        # 生成された特徴量が有効であることを確認
        assert len(features.columns) > 0
        assert len(features) == len(sample_data)