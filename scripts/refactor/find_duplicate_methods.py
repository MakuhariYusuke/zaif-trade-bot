import re
from pathlib import Path
s=Path('ztb/training/callbacks/core/callback_implementations.py').read_text()
classes={}
current_class=None
indent=0
for i,line in enumerate(s.splitlines(), start=1):
    m=re.match(r"class\s+([A-Za-z0-9_]+)\s*\(.*\):", line)
    if m:
        current_class=m.group(1)
        classes[current_class]={}
    elif current_class is not None:
        m2=re.match(r"\s+def\s+([A-Za-z0-9_]+)\s*\(", line)
        if m2:
            method=m2.group(1)
            classes[current_class].setdefault(method, []).append(i)
# Print duplicate methods
for cls, methods in classes.items():
    for method, lines in methods.items():
        if len(lines) > 1:
            print(f"Class {cls} has duplicate method {method} at lines: {lines}")
