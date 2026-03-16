"""Find files where conditional import sets a class to None, then uses ClassName | None at runtime."""
import re
import os
import ast

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
        # Find patterns like: SomeClass = None  (in except block of try/except import)
        # Then check if that name appears in type annotations with |
        names_set_to_none = set()
        for m in re.finditer(r"^\s+(\w+)\s*=\s*None\s*$", content, re.MULTILINE):
            name = m.group(1)
            if name[0].isupper():  # likely a class name
                names_set_to_none.add(name)
        
        if not names_set_to_none:
            continue
        
        for name in names_set_to_none:
            # Check if this name is used in a | expression at runtime
            pipe_pattern = re.compile(rf"\b{re.escape(name)}\s*\||\|\s*{re.escape(name)}\b")
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if pipe_pattern.search(line):
                    print(f"{path}:{i}: {stripped}  (name={name})")
