"""Find files with forward reference + Optional/Union issues."""
import pathlib
import re

base = pathlib.Path("ztb/trading")
count = 0
for f in sorted(base.rglob("*.py")):
    text = f.read_text("utf-8")
    has_future = "from __future__ import annotations" in text
    if not has_future:
        if re.search(r'Optional\[\s*"', text) or re.search(r'Union\[.*"', text):
            count += 1
            matches = re.findall(r'(?:Optional|Union)\[[^]]*"[^]]*\]', text)
            print(f"{f}: {matches[:3]}")
print(f"Total: {count} files")
