"""
Unit tests for V4FeatureExtractor

V4FeatureExtractorの単体テスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from ztb.features.unified_feature import V4FeatureExtractor


class TestV4FeatureExtractor:
    """V4FeatureExtractor unit tests"""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "feature_set": "curated",
            "model_type": "sac"
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')

        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

    def test_initialization(self, sample_config):
        """Test V4FeatureExtractor initialization"""
        extractor = V4FeatureExtractor(config=sample_config)

        assert extractor.config == sample_config
        assert hasattr(extractor, 'unified_engineer')
        assert extractor.unified_engineer is not None

    def test_initialization_without_config(self):
        """Test V4FeatureExtractor initialization without config"""
        extractor = V4FeatureExtractor()

        assert extractor.config == {}
        assert hasattr(extractor, 'unified_engineer')

    @patch('ztb.features.unified_feature.UnifiedFeatureEngineer.generate_features')
    def test_extract_features_success(self, mock_generate, sample_dataframe, sample_config):
        """Test successful feature extraction"""
        # Mock the unified engineer
        mock_df = sample_dataframe.copy()
        mock_df['feature_1'] = np.random.randn(100)
        mock_df['feature_2'] = np.random.randn(100)
        mock_generate.return_value = mock_df

        extractor = V4FeatureExtractor(config=sample_config)
        result = extractor.extract_features(sample_dataframe)

        # Verify the method was called correctly
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        assert args[0] is sample_dataframe
        assert kwargs['feature_set'] == 'curated'
        assert kwargs['model_type'] == 'sac'

        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert 'feature_1' in result.columns
        assert 'feature_2' in result.columns

    @patch('ztb.features.unified_feature.UnifiedFeatureEngineer.generate_features')
    def test_extract_features_with_custom_params(self, mock_generate, sample_dataframe):
        """Test feature extraction with custom parameters"""
        mock_df = sample_dataframe.copy()
        mock_df['custom_feature'] = np.random.randn(100)
        mock_generate.return_value = mock_df

        extractor = V4FeatureExtractor()
        custom_params = {'param1': 'value1', 'param2': 42}
        result = extractor.extract_features(sample_dataframe, **custom_params)

        # Verify custom parameters were passed
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        assert kwargs['param1'] == 'value1'
        assert kwargs['param2'] == 42

    @patch('ztb.features.unified_feature.UnifiedFeatureEngineer.generate_features')
    def test_extract_features_error_handling(self, mock_generate, sample_dataframe):
        """Test error handling in feature extraction"""
        # Mock an exception
        mock_generate.side_effect = Exception("Feature generation failed")

        extractor = V4FeatureExtractor()

        with pytest.raises(Exception, match="Feature generation failed"):
            extractor.extract_features(sample_dataframe)

    @patch('ztb.features.unified_feature.UnifiedFeatureEngineer.get_available_features')
    def test_get_feature_names(self, mock_get_features, sample_config):
        """Test getting feature names"""
        base_features = ['feature_1', 'feature_2', 'feature_3']
        mock_get_features.return_value = base_features

        extractor = V4FeatureExtractor(config=sample_config)
        result = extractor.get_feature_names()

        # Verify the method was called correctly
        mock_get_features.assert_called_once_with('sac')

        # Verify result includes base features plus short-term features
        expected_features = base_features + [
            'realized_volatility',
            'tick_volume_ratio',
            'order_flow_imbalance',
            'news_sentiment_score',
            'news_sentiment_intensity'
        ]
        assert result == expected_features
        assert isinstance(result, list)

    def test_extract_features_empty_dataframe(self):
        """Test feature extraction with empty dataframe"""
        empty_df = pd.DataFrame()

        extractor = V4FeatureExtractor()

        # Should handle empty dataframe gracefully
        with patch.object(extractor.unified_engineer, 'generate_features') as mock_generate:
            mock_generate.return_value = empty_df
            result = extractor.extract_features(empty_df)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_extract_features_with_kwargs(self, sample_dataframe):
        """Test feature extraction with various keyword arguments"""
        extractor = V4FeatureExtractor()

        with patch.object(extractor.unified_engineer, 'generate_features') as mock_generate:
            mock_df = sample_dataframe.copy()
            mock_generate.return_value = mock_df

            # Test with various parameters
            result = extractor.extract_features(
                sample_dataframe,
                normalize=True,
                scale_factor=1.5,
                include_time_features=False
            )

            mock_generate.assert_called_once()
            args, kwargs = mock_generate.call_args
            assert kwargs['normalize'] is True
            assert kwargs['scale_factor'] == 1.5
            assert kwargs['include_time_features'] is False

    def test_short_term_features_integration(self, sample_dataframe):
        """Test integration of short-term enhanced features"""
        extractor = V4FeatureExtractor()

        result = extractor.extract_features(sample_dataframe)

        # Check that short-term features are added
        expected_features = [
            'realized_volatility',
            'tick_volume_ratio',
            'order_flow_imbalance'
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Short-term feature {feature} not found"
            assert len(result[feature].dropna()) > 0, f"Short-term feature {feature} has no valid values"

    def test_realized_volatility_calculation(self, sample_dataframe):
        """Test realized volatility feature calculation"""
        extractor = V4FeatureExtractor()

        result = extractor.extract_features(sample_dataframe)

        # Check realized volatility properties
        rv_values = result['realized_volatility'].dropna()
        assert len(rv_values) > 0, "Realized volatility has no values"
        assert all(rv >= 0 for rv in rv_values), "Realized volatility should be non-negative"
        assert rv_values.max() > 0, "Realized volatility should have positive values"

    def test_tick_volume_ratio_calculation(self, sample_dataframe):
        """Test tick volume ratio feature calculation"""
        extractor = V4FeatureExtractor()

        result = extractor.extract_features(sample_dataframe)

        # Check tick volume ratio properties
        tv_values = result['tick_volume_ratio'].dropna()
        assert len(tv_values) > 0, "Tick volume ratio has no values"
        assert all(tv >= 0 for tv in tv_values), "Tick volume ratio should be non-negative"

    def test_order_flow_imbalance_calculation(self, sample_dataframe):
        """Test order flow imbalance feature calculation"""
        extractor = V4FeatureExtractor()

        result = extractor.extract_features(sample_dataframe)

        # Check order flow imbalance properties
        of_values = result['order_flow_imbalance'].dropna()
        assert len(of_values) > 0, "Order flow imbalance has no values"

        # Order flow imbalance can have large values when body size is very small
        # Just check that values are finite numbers
        assert all(np.isfinite(of) for of in of_values), "Order flow imbalance should be finite"

    def test_news_sentiment_integration(self, sample_dataframe):
        """Test news sentiment feature integration"""
        extractor = V4FeatureExtractor()

        # Test without news data
        result_no_news = extractor.extract_features(sample_dataframe)
        assert 'news_sentiment_score' not in result_no_news.columns
        assert 'news_sentiment_intensity' not in result_no_news.columns

        # Test with news data
        sample_news = [
            "Bitcoin shows strong upward momentum",
            "Market volatility increases due to economic factors"
        ]

        result_with_news = extractor.extract_features(sample_dataframe, news_data=sample_news)

        # News sentiment features should be added
        assert 'news_sentiment_score' in result_with_news.columns
        assert 'news_sentiment_intensity' in result_with_news.columns

        # Check that values are reasonable
        sentiment_scores = result_with_news['news_sentiment_score'].dropna()
        sentiment_intensities = result_with_news['news_sentiment_intensity'].dropna()

        assert len(sentiment_scores) > 0, "News sentiment scores should have values"
        assert len(sentiment_intensities) > 0, "News sentiment intensities should have values"

    def test_feature_names_inclusion(self):
        """Test that get_feature_names includes short-term features"""
        extractor = V4FeatureExtractor()

        feature_names = extractor.get_feature_names()

        # Check that short-term features are included
        short_term_features = [
            'realized_volatility',
            'tick_volume_ratio',
            'order_flow_imbalance',
            'news_sentiment_score',
            'news_sentiment_intensity'
        ]

        for feature in short_term_features:
            assert feature in feature_names, f"Short-term feature {feature} not in feature names list"

    def test_short_term_features_with_custom_parameters(self, sample_dataframe):
        """Test short-term features with custom window parameters"""
        extractor = V4FeatureExtractor()

        # Test with custom parameters
        result = extractor.extract_features(
            sample_dataframe,
            rv_window=20,  # Custom realized volatility window
            tv_window=10   # Custom tick volume window
        )

        # Features should still be present
        assert 'realized_volatility' in result.columns
        assert 'tick_volume_ratio' in result.columns
        assert 'order_flow_imbalance' in result.columns

        # Values should be valid
        assert len(result['realized_volatility'].dropna()) > 0
        assert len(result['tick_volume_ratio'].dropna()) > 0
        assert len(result['order_flow_imbalance'].dropna()) > 0