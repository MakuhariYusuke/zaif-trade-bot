#!/usr/bin/env python3
"""
Find all self._send_notification( with line numbers
"""

with open('ztb/trading/live_trader/live_trader.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if 'self._send_notification(' in line:
        print(f'Line {i}: {line.strip()}')