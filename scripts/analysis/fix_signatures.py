#!/usr/bin/env python3
import re

# Read the file
with open(
    "ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py",
    "r",
    encoding="utf-8",
) as f:
    content = f.read()

# Fix the broken signatures first
content = re.sub(
    r"def recognize\(self, data: pd\.DataFrame, index: int\) = -1\)",
    r"def recognize(self, data: pd.DataFrame, index: int = -1)",
    content,
)

# Then fix any remaining ones
content = re.sub(
    r"def recognize\(self, data: pd\.DataFrame, index: int\)(?! = -1)",
    r"def recognize(self, data: pd.DataFrame, index: int = -1)",
    content,
)

# Write back
with open(
    "ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py",
    "w",
    encoding="utf-8",
) as f:
    f.write(content)

print("Fixed all recognize method signatures in candlestick_patterns.py")
