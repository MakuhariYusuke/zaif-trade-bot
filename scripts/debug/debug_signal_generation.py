import numpy as np
import pandas as pd


def create_sample_data():
    """Create realistic sample market data for testing with clear patterns"""
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1h")
    n_points = len(dates)

    # Create clear bullish engulfing pattern data
    base_price = 100.0
    prices = []
    current_price = base_price

    # Create a downtrend first, then bullish engulfing
    for i in range(n_points):
        if i < 20:  # Downtrend
            trend = -0.002 * (20 - i)  # Getting stronger as we approach reversal
            noise = np.random.normal(0, 0.001)
            current_price *= 1 + trend + noise
        elif i == 20:  # Bullish engulfing candle (large green candle)
            # Previous candle: small red
            # Current candle: large green that engulfs previous
            current_price *= 1.03  # Large upward move
        elif i == 21:  # Confirmation candle
            current_price *= 1.015
        else:  # Continue uptrend
            trend = 0.001 * (i - 20)
            noise = np.random.normal(0, 0.001)
            current_price *= 1 + trend + noise

        prices.append(current_price)

    prices = np.array(prices)

    # Create OHLC with more realistic spreads
    highs = prices * (1 + np.random.uniform(0.001, 0.008, n_points))
    lows = prices * (1 - np.random.uniform(0.001, 0.008, n_points))

    # Create opens that create engulfing pattern
    opens = np.roll(prices, 1)  # Previous close becomes open
    opens[0] = base_price  # First open

    # Make sure engulfing pattern: previous candle small red, current large green
    opens[20] = prices[19] * 0.995  # Previous candle: small decline
    prices[20] = opens[20] * 1.03  # Current candle: large gain engulfing previous

    volumes = np.random.uniform(8000, 12000, n_points)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )

    return df


def debug_signal_generation():
    """Debug signal generation process"""
    print("=== ActionSignalGuide Signal Generation Debug ===\n")

    # Create sample data
    print("1. Creating sample market data...")
    df = create_sample_data()
    print(f"   Generated {len(df)} data points")
    print(".2f")
    print()

    # Test individual pattern recognizers directly
    print("2. Testing individual pattern recognizers...")

    # Import a simple recognizer to test
    try:
        import sys

        sys.path.insert(0, "ztb/trading/strategies/action_signal_guide")

        from pattern_recognition.candlestick_patterns import BearishEngulfingRecognizer

        test_data = df.iloc[15:22]  # Around the engulfing pattern
        print(f"   Test data shape: {test_data.shape}")
        print("   Test data:")
        for idx, row in test_data.iterrows():
            print(
                f"     {idx}: O={row['open']:.4f}, H={row['high']:.4f}, L={row['low']:.4f}, C={row['close']:.4f}"
            )

        recognizer = BearishEngulfingRecognizer()
        result = recognizer.recognize(test_data)

        if result:
            print(f"   ✓ BearishEngulfingRecognizer detected: {result}")
            print(
                f"     Direction: {result.direction}, Strength: {result.strength}, Confidence: {result.confidence}"
            )
        else:
            print("   ✗ BearishEngulfingRecognizer: No signal detected")

    except Exception as e:
        print(f"   ERROR testing recognizer: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_signal_generation()
