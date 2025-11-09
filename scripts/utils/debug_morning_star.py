import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.candlestick_patterns import (
    MorningStarRecognizer,
)

# Test data
data = pd.DataFrame(
    {
        "open": [120, 118, 116, 114, 110, 105, 103, 112],
        "high": [125, 123, 121, 119, 115, 110, 108, 117],
        "low": [115, 113, 111, 109, 105, 100, 98, 107],
        "close": [
            117,
            115,
            113,
            111,
            106,
            101,
            109,
            116,
        ],  # Changed third candle close to 109 (> 108.0)
        "volume": [1000] * 8,
    },
    index=pd.date_range("2024-01-01", periods=8, freq="D"),
)

recognizer = MorningStarRecognizer()

# Check downtrend at index 6
print("Downtrend check:", recognizer._is_downtrend(data, 6, 5))

# Check candles
candle4 = data.iloc[4]  # index 4
candle5 = data.iloc[5]  # index 5
candle6 = data.iloc[6]  # index 6

print(
    "Candle 4 (first):",
    "bearish" if recognizer.is_bearish_candle(candle4) else "bullish",
)
print(
    "Candle 5 (star):", "small" if recognizer.get_body_ratio(data, 5) < 0.5 else "large"
)
print(
    "Candle 6 (third):",
    "bullish" if recognizer.is_bullish_candle(candle6) else "bearish",
)

# Check body sizes
avg_body = recognizer._get_average_body_size(data, 6, 10)
body4 = recognizer.calculate_body_size(data, 4)
print(f"Avg body: {avg_body}, Body4: {body4}, Body4 > avg: {body4 > avg_body}")

# Check midpoint condition
first_midpoint = (candle4["open"] + candle4["close"]) / 2
print(
    f'First midpoint: {first_midpoint}, Third close: {candle6["close"]}, Third > midpoint: {candle6["close"] > first_midpoint}'
)

# Check confidence calculations
try:
    trend_strength = recognizer._calculate_trend_strength(data, 6, 5)
    print(f"Trend strength: {trend_strength}")

    candle_size_conf = recognizer._calculate_candle_size_confidence(data, 6, 0.8)
    print(f"Candle size confidence: {candle_size_conf}")

    price_movement_conf = recognizer._calculate_price_movement_confidence(data, 6, 1.0)
    print(f"Price movement confidence: {price_movement_conf}")

    star_body_ratio = recognizer.get_body_ratio(data, 5)
    print(f"Star body ratio: {star_body_ratio}")

    pattern_factors = {
        "trend_strength": trend_strength,
        "candle_size": candle_size_conf,
        "price_movement": price_movement_conf,
        "pattern_completeness": min(1.0, star_body_ratio * 2),
    }
    print(f"Pattern factors: {pattern_factors}")

    confidence = recognizer._calculate_pattern_confidence(
        data, 6, pattern_factors, base_confidence=0.7
    )
    print(f"Final confidence: {confidence}, type: {type(confidence)}")

    # Try to create SignalResult manually
    from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
        SignalResult,
    )

    try:
        signal_result = SignalResult(
            signal_type="morning_star",
            strength=confidence,
            direction=1.0,
            description="Morning Star: Bullish reversal pattern",
            timestamp=data.index[6],
            confidence=confidence,
            metadata={"pattern": "morning_star", "confidence": confidence},
        )
        print("Manual SignalResult creation: SUCCESS")
    except Exception as e:
        print(f"Manual SignalResult creation failed: {e}")

except Exception as e:
    print(f"Error in confidence calculation: {e}")

# Step-by-step check of Morning Star conditions
print("\n=== Step-by-step Morning Star validation ===")

# 1. Index check
print(f"1. Index check: index=6 >= 2: {6 >= 2}")

# 2. Downtrend check
downtrend_ok = recognizer._is_downtrend(data, 6, lookback=5)
print(f"2. Downtrend check: {downtrend_ok}")

if downtrend_ok:
    first = data.iloc[4]  # Large bearish
    second = data.iloc[5]  # Small (star)
    third = data.iloc[6]  # Large bullish

    # 3. First candle check
    avg_body_size = recognizer._get_average_body_size(data, 6, 10)
    first_bearish = recognizer.is_bearish_candle(first)
    first_large = recognizer.calculate_body_size(data, 4) > avg_body_size
    print(
        f"3. First candle: bearish={first_bearish}, large={first_large}, avg_body={avg_body_size}"
    )

    # 4. Second candle check
    star_body_ratio = recognizer.get_body_ratio(data, 5)
    star_small = star_body_ratio <= 0.5
    print(f"4. Second candle: body_ratio={star_body_ratio}, small={star_small}")

    # 5. Third candle check
    third_bullish = recognizer.is_bullish_candle(third)
    first_midpoint = (first["open"] + first["close"]) / 2
    third_above_midpoint = third["close"] > first_midpoint
    print(
        f"5. Third candle: bullish={third_bullish}, above_midpoint={third_above_midpoint}, midpoint={first_midpoint}"
    )

    all_conditions = (
        first_bearish
        and first_large
        and star_small
        and third_bullish
        and third_above_midpoint
    )
    print(f"All conditions met: {all_conditions}")

# Try recognition with detailed error checking
try:
    result = recognizer.recognize(data, 6)
    print("Recognition result:", result)
except Exception as e:
    print(f"Recognition failed with exception: {e}")
    import traceback

    traceback.print_exc()
