#!/usr/bin/env python3
"""
Remove BOM from live_trader.py
"""

with open('ztb/trading/live_trader/live_trader.py', 'rb') as f:
    content = f.read()

# Remove BOM if present
if content.startswith(b'\xef\xbb\xbf'):
    content = content[3:]

with open('ztb/trading/live_trader/live_trader.py', 'wb') as f:
    f.write(content)

print("Removed BOM from live_trader.py")