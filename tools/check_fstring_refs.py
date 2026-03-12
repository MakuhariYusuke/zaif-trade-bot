"""Check for potential undefined name references in f-strings."""
import re

with open("scripts/v460/lib/fill_loop_orchestrator.py", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    # Look for f-string lines with UPPER_CASE references
    if ('f"' in line or "f'" in line):
        for m in re.finditer(r'\{([A-Z_][A-Z_0-9]+)\}', line):
            print(f"  L{i}: {m.group(1)}")

# Also check all py files in lib for similar pattern
import glob
print("\n--- All lib/*.py files ---")
for path in sorted(glob.glob("scripts/v460/lib/*.py")):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f.readlines(), 1):
            if ('f"' in line or "f'" in line):
                for m in re.finditer(r'\{([A-Z_][A-Z_0-9]+)\}', line):
                    print(f"  {path}:L{i}: {m.group(1)}")
