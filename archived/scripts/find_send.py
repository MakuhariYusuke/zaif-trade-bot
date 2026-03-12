#!/usr/bin/env python3
"""
Find all self._send_notification( positions
"""

with open("ztb/trading/live_trader/live_trader.py", "rb") as f:
    data = f.read()

pos = 0
count = 0
while True:
    pos = data.find(b"self._send_notification(", pos)
    if pos == -1:
        break
    count += 1
    print(f"{count} pos: {pos}")
    pos += 1
    if count > 30:
        break
