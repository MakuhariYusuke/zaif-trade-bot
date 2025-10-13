#!/usr/bin/env python3
"""
Generate sample BTC/JPY dataset for optimization testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_btc_data(num_rows=5000, start_price=5000000):
    """Generate realistic-looking BTC/JPY price data"""
    
    # Start date
    start_date = datetime(2024, 1, 1)
    
    # Generate timestamps (1-minute intervals)
    timestamps = [start_date + timedelta(minutes=i) for i in range(num_rows)]
    
    # Generate price movement using random walk
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, num_rows)  # 0.1% volatility per minute
    
    # Calculate prices
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Add some variation for open/high/low
        variation = abs(np.random.normal(0, 0.0005))  # 0.05% variation
        open_price = close * (1 + np.random.uniform(-variation, variation))
        high_price = max(open_price, close) * (1 + variation)
        low_price = min(open_price, close) * (1 - variation)
        volume = np.random.uniform(50, 200)  # Random volume
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating sample BTC/JPY dataset...")
    df = generate_sample_btc_data(num_rows=5000)
    
    output_path = "btc_jpy_real_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} rows")
    print(f"   Saved to: {output_path}")
    print(f"   Price range: {df['close'].min():.0f} - {df['close'].max():.0f}")
    print(f"   Start: {df['timestamp'].iloc[0]}")
    print(f"   End: {df['timestamp'].iloc[-1]}")
