from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_range_data(
    n_samples=10000,
    start_price=1000000.0,
    volatility=0.001,
    mean_reversion_strength=0.1,
    period=100,
    noise_level=0.2,
):
    """
    Generate synthetic range-bound market data.
    """
    prices = [start_price]
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_samples)]

    # Sine wave component for range
    t = np.arange(n_samples)
    sine_wave = np.sin(2 * np.pi * t / period)

    current_price = start_price

    data = []

    for i in range(n_samples):
        # Mean reversion target (center of range)
        target = start_price + (sine_wave[i] * start_price * volatility * 10)

        # Random walk component
        noise = np.random.normal(0, start_price * volatility)

        # Mean reversion pull
        pull = (target - current_price) * mean_reversion_strength

        # Update price
        change = pull + noise
        current_price += change

        # Generate OHLC
        open_p = current_price
        high_p = current_price + abs(
            np.random.normal(0, start_price * volatility * 0.5)
        )
        low_p = current_price - abs(np.random.normal(0, start_price * volatility * 0.5))
        close_p = current_price + np.random.normal(0, start_price * volatility * 0.1)

        # Ensure High is highest and Low is lowest
        high_p = max(open_p, close_p, high_p)
        low_p = min(open_p, close_p, low_p)

        volume = np.random.uniform(1.0, 10.0)

        data.append(
            {
                "timestamp": timestamps[i],
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # 1. Tight Range (Low Volatility)
    df_tight = generate_range_data(
        volatility=0.0005, mean_reversion_strength=0.2, period=50
    )
    df_tight.to_csv("data/range_tight.csv", index=False)
    print("Generated data/range_tight.csv")

    # 2. Wide Range (High Volatility)
    df_wide = generate_range_data(
        volatility=0.005, mean_reversion_strength=0.05, period=200
    )
    df_wide.to_csv("data/range_wide.csv", index=False)
    print("Generated data/range_wide.csv")

    # 3. Choppy Range (Fast Oscillation)
    df_choppy = generate_range_data(
        volatility=0.002, mean_reversion_strength=0.5, period=20
    )
    df_choppy.to_csv("data/range_choppy.csv", index=False)
    print("Generated data/range_choppy.csv")
