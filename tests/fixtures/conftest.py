"""
Test fixtures for SAC v446 testing

SAC v446テスト用のフィクスチャ
"""

import numpy as np
import pandas as pd
import pytest

from tests.helpers.market_data import make_exchange_random_walk_ohlcv_data

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    return make_exchange_random_walk_ohlcv_data(
        rows=1000,
        seed=42,
        start='2023-01-01',
        freq='1min',
        base_price=5000000.0,
        include_timestamp=True,
    )


@pytest.fixture
def large_market_data():
    """Create large dataset for performance testing"""
    return make_exchange_random_walk_ohlcv_data(
        rows=5000,
        seed=42,
        start='2023-01-01',
        freq='1min',
        base_price=100.0,
        include_timestamp=True,
    )


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame"""
    np.random.seed(42)
    n_samples = 1000

    features = pd.DataFrame({
        'sma_5': np.random.randn(n_samples),
        'sma_20': np.random.randn(n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.randn(n_samples),
        'bb_upper': np.random.randn(n_samples),
        'bb_lower': np.random.randn(n_samples),
        'volume_sma': np.random.randn(n_samples),
        'returns': np.random.normal(0, 0.02, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
        'momentum': np.random.randn(n_samples)
    })

    return features


@pytest.fixture
def mock_unified_feature_engineer():
    """Mock UnifiedFeatureEngineer for testing"""
    from unittest.mock import Mock

    mock_engineer = Mock()
    mock_engineer.generate_features.return_value = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'feature4': np.random.randn(1000),
        'feature5': np.random.randn(1000)
    })
    return mock_engineer


@pytest.fixture
def mock_isolation_forest():
    """Mock IsolationForest for testing"""
    from unittest.mock import Mock

    mock_if = Mock()
    mock_if.fit_predict.return_value = np.ones(1000)
    return mock_if


@pytest.fixture
def mock_lof():
    """Mock LocalOutlierFactor for testing"""
    from unittest.mock import Mock

    mock_lof = Mock()
    mock_lof.fit_predict.return_value = np.ones(1000)
    return mock_lof


@pytest.fixture
def config_dict():
    """Sample configuration dictionary"""
    return {
        'data': {
            'noise_threshold': 0.05,
            'anomaly_contamination': 0.1,
            'synthetic_samples': 500
        },
        'features': {
            'v4_enabled': True,
            'technical_indicators': ['sma', 'rsi', 'macd', 'bb'],
            'timeframes': ['5m', '15m', '1h']
        },
        'training': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100
        }
    }
