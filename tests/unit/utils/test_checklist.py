import re
from pathlib import Path


def test_checklist_completed():
    checklist = Path("docs/checklist_1M.md").read_text(encoding="utf-8")
    unchecked = re.findall(r"- \[ \]", checklist)
    assert not unchecked, f"Unresolved checklist items found: {len(unchecked)}"
