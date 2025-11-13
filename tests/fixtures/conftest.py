"""
Test fixtures for SAC v446 testing

SAC v446テスト用のフィクスチャ
"""

import sys
import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    np.random.seed(42)
    n_samples = 1000

    # Generate realistic market data (JPY-based for Zaif exchange)
    base_price = 5000000.0  # Realistic BTC/JPY price
    prices = []
    current_price = base_price

    for i in range(n_samples):
        # Random walk with some trend
        change = np.random.normal(0, 0.01)  # 1% volatility
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = []
    for i in range(n_samples):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.lognormal(10, 1)  # Log-normal volume

        data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
            'open': price * (1 + np.random.normal(0, 0.002)),
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(data)


@pytest.fixture
def small_market_data():
    """Create small market data for quick testing"""
    np.random.seed(42)
    n_samples = 100

    data = []
    for i in range(n_samples):
        price = 100.0 + i * 0.1  # Simple trend
        data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': 1000.0
        })

    return pd.DataFrame(data)


@pytest.fixture
def noisy_market_data():
    """Create market data with added noise"""
    np.random.seed(42)
    n_samples = 500

    # Clean data
    base_price = 100.0
    prices = []
    current_price = base_price

    for i in range(n_samples):
        change = np.random.normal(0, 0.005)
        current_price *= (1 + change)
        prices.append(current_price)

    # Add noise
    data = []
    for i in range(n_samples):
        price = prices[i]
        noise_level = 0.02  # 2% noise

        data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
            'open': price * (1 + np.random.normal(0, noise_level)),
            'high': price * (1 + abs(np.random.normal(0, noise_level))),
            'low': price * (1 - abs(np.random.normal(0, noise_level))),
            'close': price * (1 + np.random.normal(0, noise_level)),
            'volume': np.random.lognormal(10, 0.5)
        })

    return pd.DataFrame(data)


@pytest.fixture
def anomalous_market_data():
    """Create market data with anomalies"""
    np.random.seed(42)
    n_samples = 500

    # Normal data
    data = []
    for i in range(n_samples):
        price = 100.0 + np.sin(i * 0.1) * 5  # Sine wave pattern
        data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
            'open': price,
            'high': price * 1.005,
            'low': price * 0.995,
            'close': price,
            'volume': 1000.0
        })

    # Add anomalies at specific points
    anomaly_indices = [100, 200, 300, 400]
    for idx in anomaly_indices:
        if idx < len(data):
            # Extreme price movement
            data[idx]['close'] *= 2.0  # Double the price
            data[idx]['high'] *= 2.5
            data[idx]['volume'] *= 10  # 10x volume

    return pd.DataFrame(data)


@pytest.fixture
def large_market_data():
    """Create large dataset for performance testing"""
    np.random.seed(42)
    n_samples = 5000

    # Generate market data
    base_price = 100.0
    prices = []
    current_price = base_price

    for i in range(n_samples):
        change = np.random.normal(0, 0.01)
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = []
    for i in range(n_samples):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.lognormal(10, 1)

        data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
            'open': price * (1 + np.random.normal(0, 0.002)),
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(data)


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
def sample_training_data():
    """Create sample training data with features and targets"""
    np.random.seed(42)
    n_samples = 1000

    features = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples)
    })

    # Create target (simplified: positive when feature1 > 0)
    targets = (features['feature1'] > 0).astype(int)

    return features, targets


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


@pytest.fixture
def temp_directory(tmp_path):
    """Temporary directory for file operations"""
    return tmp_path


@pytest.fixture
def sample_csv_file(temp_directory, sample_market_data):
    """Create a temporary CSV file with sample data"""
    csv_path = temp_directory / "sample_data.csv"
    sample_market_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_config(temp_directory, config_dict):
    """Create a temporary JSON config file"""
    import json

    json_path = temp_directory / "config.json"
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    return json_path