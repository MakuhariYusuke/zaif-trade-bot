import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.candlestick_patterns import (
    HammerRecognizer,
)

# Test Hammer pattern
dates = pd.date_range("2024-01-01", periods=15, freq="D")
hammer_data = pd.DataFrame(
    {
        "open": [
            130,
            125,
            120,
            115,
            110,
            105,
            102,
            108,
            105,
            103,
            107,
            109,
            111,
            113,
            115,
        ],
        "high": [
            135,
            130,
            125,
            120,
            115,
            110,
            104,
            113,
            110,
            108,
            112,
            114,
            116,
            118,
            120,
        ],
        "low": [
            125,
            120,
            115,
            110,
            105,
            100,
            92,
            103,
            100,
            98,
            102,
            104,
            106,
            108,
            110,
        ],
        "close": [
            126,
            121,
            116,
            111,
            106,
            101,
            103,
            111,
            102,
            100,
            110,
            112,
            114,
            116,
            118,
        ],
        "volume": [1000] * 15,
    },
    index=dates,
)

hammer_recognizer = HammerRecognizer()
hammer_result = hammer_recognizer.recognize(hammer_data, 6)

print("Hammer pattern test:")
print(f"Result: {hammer_result}")
if hammer_result:
    print(
        f"Confidence: {hammer_result.confidence} (type: {type(hammer_result.confidence)})"
    )
    print(f"Confidence is float: {isinstance(hammer_result.confidence, float)}")
    print(f"Confidence in range: {0.0 <= hammer_result.confidence <= 1.0}")
