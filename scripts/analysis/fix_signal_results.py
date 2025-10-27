#!/usr/bin/env python3
"""
Script to add timestamp parameter to all SignalResult instantiations.
"""

import os
import re


def add_timestamp_to_signal_results():
    """Add timestamp parameter to all SignalResult instantiations."""

    pattern_files = [
        "ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py",
        "ztb/trading/strategies/action_signal_guide/pattern_recognition/fibonacci_patterns.py",
        "ztb/trading/strategies/action_signal_guide/pattern_recognition/gann_analysis.py",
        "ztb/trading/strategies/action_signal_guide/pattern_recognition/harmonic_patterns.py",
        "ztb/trading/strategies/action_signal_guide/pattern_recognition/wave_counting.py",
    ]

    for file_path in pattern_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Pattern to match SignalResult instantiation
        # Look for return SignalResult( with parameters
        pattern = r"return SignalResult\(\s*([^)]+)\)"

        def replace_match(match):
            params = match.group(1)
            # Check if timestamp is already present
            if "timestamp=" in params:
                return match.group(0)

            # Add timestamp parameter before metadata
            if "metadata=" in params:
                # Insert timestamp before metadata
                params = re.sub(
                    r"(.*)(metadata=.*)",
                    r"\1timestamp=data.index[index],\n            \2",
                    params,
                )
            else:
                # Add timestamp at the end
                params = params.rstrip() + ",\n            timestamp=data.index[index]"

            return f"return SignalResult(\n            {params}\n        )"

        new_content = re.sub(pattern, replace_match, content, flags=re.DOTALL)

        # Write back if changed
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated: {file_path}")
        else:
            print(f"No changes needed: {file_path}")


if __name__ == "__main__":
    add_timestamp_to_signal_results()
