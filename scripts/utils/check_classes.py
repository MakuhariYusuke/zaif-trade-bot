#!/usr/bin/env python3
"""
Check pattern recognizer class names
"""

import importlib
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

files = [
    "adx_patterns",
    "atr",
    "bollinger_patterns",
    "candlestick_patterns",
    "dow_theory",
    "fibonacci_patterns",
    "gann_analysis",
    "granville_law",
    "harmonic_patterns",
    "ichimoku",
    "macd",
    "oscillator_patterns",
    "rsi",
    "volume_patterns",
    "wave_counting",
]

for file in files:
    try:
        module = importlib.import_module(
            f"ztb.trading.strategies.action_signal_guide.pattern_recognition.{file}"
        )
        classes = [
            name
            for name in dir(module)
            if name.endswith("Recognizer") and not name.startswith("_")
        ]
        print(f"{file}: {classes}")
    except Exception as e:
        print(f"{file}: ERROR - {e}")
