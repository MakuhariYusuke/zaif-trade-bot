#!/usr/bin/env python3
"""
Remove null bytes from live_trader.py
"""

with open('ztb/trading/live_trader/live_trader.py', 'rb') as f:
    content = f.read()

# Count null bytes
null_count = content.count(b'\x00')

# Remove null bytes
content = content.replace(b'\x00', b'')

with open('ztb/trading/live_trader/live_trader.py', 'wb') as f:
    f.write(content)

print(f"Removed {null_count} null bytes from live_trader.py")