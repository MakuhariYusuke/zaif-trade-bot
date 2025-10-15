#!/usr/bin/env python3
"""
Fix syntax error in live_trader.py by removing unterminated docstring.
"""

import re

def fix_live_trader():
    with open('ztb/trading/live_trader/live_trader.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the _archive_price_history function
    pattern = r'(def _archive_price_history\(self\) -> None:\s*\n\s*""".*?""")\s*\n'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Remove the docstring
        fixed = re.sub(pattern, r'\1\n', content, flags=re.DOTALL)
        
        with open('ztb/trading/live_trader/live_trader.py', 'w', encoding='utf-8') as f:
            f.write(fixed)
        print("Fixed syntax error in live_trader.py")
    else:
        print("Pattern not found")

if __name__ == "__main__":
    fix_live_trader()