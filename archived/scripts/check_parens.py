#!/usr/bin/env python3
"""
Check parentheses balance in live_trader.py
"""

with open("ztb/trading/live_trader/live_trader.py", "r") as f:
    lines = f.readlines()

balance = 0
for i, line in enumerate(lines, 1):
    line_balance = 0
    for char in line:
        if char == "(":
            balance += 1
            line_balance += 1
        elif char == ")":
            balance -= 1
            line_balance -= 1
    if line_balance != 0:
        print(
            f"Line {i}: balance change {line_balance}, total {balance} - {line.strip()}"
        )
    if balance < 0:
        print(f"Extra closing paren at line {i}")
        break

print(f"Final balance: {balance}")
