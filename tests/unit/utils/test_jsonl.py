from __future__ import annotations

import json
from pathlib import Path

from ztb.io.jsonl import read_tail_jsonl_objects


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def test_read_tail_jsonl_objects_returns_last_rows(tmp_path: Path) -> None:
    path = tmp_path / "tail.jsonl"
    _write_lines(
        path,
        [
            json.dumps({"n": idx}, ensure_ascii=False) + "\n"
            for idx in range(20)
        ],
    )

    rows = read_tail_jsonl_objects(path, limit=3)

    assert [row["n"] for row in rows] == [17, 18, 19]


def test_read_tail_jsonl_objects_handles_bom_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    _write_lines(
        path,
        [
            "\ufeff" + json.dumps({"n": 1}, ensure_ascii=False) + "\n",
            "\n",
            json.dumps({"n": 2}, ensure_ascii=False) + "\n",
        ],
    )

    rows = read_tail_jsonl_objects(path, limit=5)

    assert [row["n"] for row in rows] == [1, 2]


def test_read_tail_jsonl_objects_warn_malformed_keeps_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    _write_lines(
        path,
        [
            json.dumps({"n": 1}, ensure_ascii=False) + "\n",
            "not-json\n",
            "42\n",
            json.dumps({"n": 2}, ensure_ascii=False) + "\n",
        ],
    )

    rows = read_tail_jsonl_objects(path, limit=4, warn_malformed=True)

    assert [row["n"] for row in rows] == [1, 2]
