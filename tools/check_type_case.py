"""Quick check for lowered Type in docstrings/comments."""
import pathlib
import re

base = pathlib.Path("ztb/utils")
for f in sorted(base.rglob("*.py")):
    text = f.read_text("utf-8")
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("#"):
            if re.search(
                r"\btype (definitions|validator|guard|annotations|checking|alias|error|safe|hints)",
                stripped,
            ):
                print(f"{f}:{i}: {stripped[:120]}")
