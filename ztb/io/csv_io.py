"""
CSV write helpers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ztb.io.common import PathLike, ensure_parent_dir


def write_csv_dicts(
    path: PathLike,
    rows: Iterable[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> Path:
    target = ensure_parent_dir(path)
    rows_list = list(rows)
    if not rows_list:
        target.touch()
        return target

    if fieldnames is None:
        keys: set[str] = set()
        for row in rows_list:
            keys.update(row.keys())
        fieldnames = sorted(keys)

    with open(target, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)

    return target
