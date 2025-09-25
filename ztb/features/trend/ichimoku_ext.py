#!/usr/bin/env python3
"""
ichimoku_ext.py
Extended Ichimoku features with cloud thickness, price-cloud distance, and lagging span analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_ichimoku_extended(df: pd.DataFrame, 
                              tenkan_period: int = 9,
                              kijun_period: int = 26, 
                              senkou_span_b_period: int = 52) -> pd.DataFrame:
    """
    Calculate extended Ichimoku features including:
    - Traditional lines (Tenkan, Kijun, Senkou Span A, B)
    - Cloud thickness
    - Price-cloud center distance
    - Lagging span analysis
    
    Args:
        df: DataFrame with OHLC data
        tenkan_period: Tenkan-sen period (default: 9)
        kijun_period: Kijun-sen period (default: 26)
        senkou_span_b_period: Senkou Span B period (default: 52)
    
    Returns:
        DataFrame with extended Ichimoku features
    """
    
    result = pd.DataFrame(index=df.index)
    
    # Basic Ichimoku lines
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    result['ichimoku_tenkan'] = (df['high'].rolling(tenkan_period).max() + 
                                df['low'].rolling(tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    result['ichimoku_kijun'] = (df['high'].rolling(kijun_period).max() + 
                               df['low'].rolling(kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_span_a = (result['ichimoku_tenkan'] + result['ichimoku_kijun']) / 2
    result['ichimoku_senkou_a'] = senkou_span_a.shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    senkou_span_b = (df['high'].rolling(senkou_span_b_period).max() + 
                     df['low'].rolling(senkou_span_b_period).min()) / 2
    result['ichimoku_senkou_b'] = senkou_span_b.shift(kijun_period)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    result['ichimoku_chikou'] = df['close'].shift(-kijun_period)
    
    # === Extended Features ===
    
    # 1. Cloud thickness (absolute difference between Span A and B)
    result['ichimoku_cloud_thickness'] = abs(result['ichimoku_senkou_a'] - result['ichimoku_senkou_b'])
    
    # 2. Price-cloud center distance
    cloud_center = (result['ichimoku_senkou_a'] + result['ichimoku_senkou_b']) / 2
    result['ichimoku_price_cloud_distance'] = df['close'] - cloud_center
    
    # 3. Price-cloud distance normalized by cloud thickness (avoid division by zero)
    result['ichimoku_price_cloud_normalized'] = np.where(
        result['ichimoku_cloud_thickness'] > 0,
        result['ichimoku_price_cloud_distance'] / result['ichimoku_cloud_thickness'],
        0
    )
    
    # 4. Price position relative to cloud (above=1, inside=0, below=-1)
    result['ichimoku_price_position'] = np.where(
        df['close'] > np.maximum(result['ichimoku_senkou_a'], result['ichimoku_senkou_b']), 1,
        np.where(df['close'] < np.minimum(result['ichimoku_senkou_a'], result['ichimoku_senkou_b']), -1, 0)
    )
    
    # 5. Tenkan-Kijun cross signal
    result['ichimoku_tk_cross'] = np.where(result['ichimoku_tenkan'] > result['ichimoku_kijun'], 1, -1)
    
    # 6. Chikou span vs price comparison (lagging span confirmation)
    # Compare current chikou with price 26 periods ago
    price_26_ago = df['close'].shift(kijun_period)
    result['ichimoku_chikou_confirmation'] = np.where(
        result['ichimoku_chikou'] > price_26_ago, 1,
        np.where(result['ichimoku_chikou'] < price_26_ago, -1, 0)
    )
    
    # 7. Cloud color (green=1 when Span A > Span B, red=-1 otherwise)
    result['ichimoku_cloud_color'] = np.where(result['ichimoku_senkou_a'] > result['ichimoku_senkou_b'], 1, -1)
    
    # 8. Distance ratios (normalized by close price to make scale-invariant)
    close_price = df['close']
    result['ichimoku_tenkan_ratio'] = (result['ichimoku_tenkan'] - close_price) / close_price
    result['ichimoku_kijun_ratio'] = (result['ichimoku_kijun'] - close_price) / close_price
    
    # 9. Multi-timeframe confirmation score (simple version)
    # This combines multiple signals: TK cross + price position + chikou confirmation
    result['ichimoku_composite_signal'] = (
        result['ichimoku_tk_cross'] + 
        result['ichimoku_price_position'] + 
        result['ichimoku_chikou_confirmation']
    ) / 3
    
    # Handle NaN values - forward fill for the first few rows where calculations aren't possible
    result = result.fillna(method='bfill', limit=max(tenkan_period, kijun_period, senkou_span_b_period))
    
    # Fill remaining NaN with 0 (for edge cases)
    result = result.fillna(0)
    
    return result


def ichimoku_feature_summary() -> Dict[str, str]:
    """
    Return a summary of all Ichimoku extended features for documentation
    """
    return {
        'ichimoku_tenkan': 'Tenkan-sen (Conversion Line) - short-term trend',
        'ichimoku_kijun': 'Kijun-sen (Base Line) - medium-term trend',
        'ichimoku_senkou_a': 'Senkou Span A (Leading Span A) - fast cloud edge',
        'ichimoku_senkou_b': 'Senkou Span B (Leading Span B) - slow cloud edge',
        'ichimoku_chikou': 'Chikou Span (Lagging Span) - momentum confirmation',
        'ichimoku_cloud_thickness': 'Absolute thickness of the cloud (volatility measure)',
        'ichimoku_price_cloud_distance': 'Distance from price to cloud center',
        'ichimoku_price_cloud_normalized': 'Price-cloud distance normalized by thickness',
        'ichimoku_price_position': 'Price position relative to cloud (-1/0/1)',
        'ichimoku_tk_cross': 'Tenkan-Kijun cross signal (-1/1)',
        'ichimoku_chikou_confirmation': 'Lagging span momentum confirmation (-1/0/1)',
        'ichimoku_cloud_color': 'Cloud color: green=1, red=-1',
        'ichimoku_tenkan_ratio': 'Tenkan distance ratio to price',
        'ichimoku_kijun_ratio': 'Kijun distance ratio to price',
        'ichimoku_composite_signal': 'Composite signal combining multiple Ichimoku elements'
    }


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 1000
    
    # Generate test OHLC data
    test_data = pd.DataFrame({
        'high': np.random.uniform(100, 200, n),
        'low': np.random.uniform(50, 150, n),
        'close': np.random.uniform(75, 175, n),
        'volume': np.random.uniform(1000, 5000, n)
    })
    
    # Ensure high >= close >= low
    for i in range(n):
        test_data.loc[i, 'high'] = max(test_data.loc[i, 'high'], test_data.loc[i, 'close'])
        test_data.loc[i, 'low'] = min(test_data.loc[i, 'low'], test_data.loc[i, 'close'])
    
    # Calculate features
    features = calculate_ichimoku_extended(test_data)
    
    print("Ichimoku Extended Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"NaN count per column:\n{features.isnull().sum()}")
    
    # Summary
    summary = ichimoku_feature_summary()
    print(f"\nFeature summary:\n{summary}")