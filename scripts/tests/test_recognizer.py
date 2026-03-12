import pandas as pd

# Create simple test data
data = [
    {
        "timestamp": pd.Timestamp("2024-01-01 00:00"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 99.5,
        "volume": 1000,
    },
    {
        "timestamp": pd.Timestamp("2024-01-01 01:00"),
        "open": 99.5,
        "high": 100.5,
        "low": 98.5,
        "close": 99.0,
        "volume": 1000,
    },
    {
        "timestamp": pd.Timestamp("2024-01-01 02:00"),
        "open": 99.0,
        "high": 100.0,
        "low": 98.0,
        "close": 98.5,
        "volume": 1000,
    },
    {
        "timestamp": pd.Timestamp("2024-01-01 03:00"),
        "open": 98.5,
        "high": 99.5,
        "low": 97.5,
        "close": 98.0,
        "volume": 1000,
    },
    {
        "timestamp": pd.Timestamp("2024-01-01 04:00"),
        "open": 97.8,
        "high": 102.0,
        "low": 97.5,
        "close": 101.5,
        "volume": 2000,
    },
    {
        "timestamp": pd.Timestamp("2024-01-01 05:00"),
        "open": 101.5,
        "high": 103.0,
        "low": 101.0,
        "close": 102.5,
        "volume": 1500,
    },
]

df = pd.DataFrame(data)
print("Test data created:")
for idx, row in df.iterrows():
    direction = "GREEN" if row["close"] > row["open"] else "RED"
    print(
        f'  {idx}: O={row["open"]:.1f}, H={row["high"]:.1f}, L={row["low"]:.1f}, C={row["close"]:.1f} ({direction})'
    )

# Test if we can import the recognizer classes
try:
    from ztb.trading.strategies.action_signal_guide.pattern_recognition.candlestick_patterns import (
        BullishEngulfingRecognizer,
    )

    print("\nBullishEngulfingRecognizer imported successfully")

    # Test the recognizer
    test_data = df.iloc[2:5]  # Previous red candle and engulfing green candle
    print(f"\nTesting with data indices 2-4 (shape: {test_data.shape})")

    recognizer = BullishEngulfingRecognizer()
    result = recognizer.recognize(test_data)

    if result:
        print("SUCCESS: Signal detected!")
        print(f"  Direction: {result.direction}")
        print(f"  Strength: {result.strength}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Description: {result.description}")
    else:
        print("No signal detected")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
