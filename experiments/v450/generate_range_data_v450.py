from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_range_data_v450(
    n_samples=10000,
    start_price=1000000.0,
    volatility=0.001,
    mean_reversion_strength=0.1,
    period=100,
    noise_level=0.2,
):
    """
    v450: Generate range-bound market data for testing dynamic thresholding.
    """
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_samples)]
    t = np.arange(n_samples)
    sine_wave = np.sin(2 * np.pi * t / period)
    current_price = start_price
    data = []
    for i in range(n_samples):
        target = start_price + (sine_wave[i] * start_price * volatility * 10)
        noise = np.random.normal(0, start_price * volatility)
        pull = (target - current_price) * mean_reversion_strength
        change = pull + noise
        current_price += change
        open_p = current_price
        high_p = current_price + abs(
            np.random.normal(0, start_price * volatility * 0.5)
        )
        low_p = current_price - abs(np.random.normal(0, start_price * volatility * 0.5))
        close_p = current_price + np.random.normal(0, start_price * volatility * 0.1)
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
    # Generate multiple profiles to stress test z_score vs volatility-based thresholds
    profiles = [
        (0.0005, 0.2, 50, "range_tight.csv"),
        (0.002, 0.1, 150, "range_medium.csv"),
        (0.005, 0.05, 250, "range_wide.csv"),
    ]
    for vol, mr, period, filename in profiles:
        df = generate_range_data_v450(
            volatility=vol, mean_reversion_strength=mr, period=period
        )
        path = f"data/{filename}"
        df.to_csv(path, index=False)
        print(f"Generated {path}")
