"""Find files with forward reference string | pattern without future annotations."""
import re
import os

pattern = re.compile(r'"[A-Z][a-zA-Z0-9_]+"\s*\||\|\s*"[A-Z][a-zA-Z0-9_]+"')
future_pattern = re.compile(r"from __future__ import annotations")

for root, dirs, files in os.walk("ztb"):
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        try:
            content = open(path, encoding="utf-8").read()
        except Exception:
            continue
        if future_pattern.search(content):
            continue
        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if '"""' in stripped or "'''" in stripped:
                continue
            if pattern.search(line):
                print(f"{path}:{i}: {stripped}")
