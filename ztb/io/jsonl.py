"""
Plain JSONL read/write helpers.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

from ztb.io.common import PathLike, _to_path, ensure_parent_dir

logger = logging.getLogger(__name__)
_TAIL_READ_CHUNK_SIZE = 8192

def _parse_jsonl_object_line(
    line: str,
    *,
    source: Path,
    line_no: int,
    warn_malformed: bool,
) -> dict[str, object] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        if warn_malformed:
            logger.warning("Skipping malformed JSONL line: %s:%d", source, line_no)
        return None
    if not isinstance(payload, dict):
        if warn_malformed:
            logger.warning("Skipping non-object JSONL line: %s:%d", source, line_no)
        return None
    return payload

def iter_jsonl_objects(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    warn_malformed: bool = False,
) -> Iterator[dict[str, object]]:
    """Yield JSON objects from a JSONL file, skipping malformed lines."""
    source = _to_path(path)
    with open(source, "r", encoding=encoding, errors=errors) as f:
        for line_no, line in enumerate(f, 1):
            # Tolerate UTF-8 BOM emitted by some editors/exporters.
            if line_no == 1 and line.startswith("\ufeff"):
                line = line.removeprefix("\ufeff")
            parsed = _parse_jsonl_object_line(
                line,
                source=source,
                line_no=line_no,
                warn_malformed=warn_malformed,
            )
            if parsed is not None:
                yield parsed

def read_jsonl_objects(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    warn_malformed: bool = False,
) -> list[dict[str, object]]:
    """Read all JSON object rows from a JSONL file."""
    return list(
        iter_jsonl_objects(
            path,
            encoding=encoding,
            errors=errors,
            warn_malformed=warn_malformed,
        )
    )


def read_tail_jsonl_objects(
    path: PathLike,
    *,
    limit: int,
    encoding: str = "utf-8",
    errors: str = "replace",
    warn_malformed: bool = False,
) -> list[dict[str, object]]:
    """Read up to the last ``limit`` JSON object rows from a JSONL file."""
    if limit <= 0:
        return []
    source = _to_path(path)
    if warn_malformed:
        tail_lines: deque[tuple[int, str]] = deque(maxlen=limit)
        with open(source, "r", encoding=encoding, errors=errors) as f:
            for line_no, line in enumerate(f, 1):
                if line_no == 1 and line.startswith("\ufeff"):
                    line = line.removeprefix("\ufeff")
                if line.strip():
                    tail_lines.append((line_no, line))

        rows: list[dict[str, object]] = []
        for line_no, line in tail_lines:
            parsed = _parse_jsonl_object_line(
                line,
                source=source,
                line_no=line_no,
                warn_malformed=True,
            )
            if parsed is not None:
                rows.append(parsed)
        return rows

    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(
        _read_tail_text_lines(source, limit=limit, encoding=encoding, errors=errors),
        1,
    ):
        parsed = _parse_jsonl_object_line(
            line,
            source=source,
            line_no=line_no,
            warn_malformed=False,
        )
        if parsed is not None:
            rows.append(parsed)
    return rows


def _read_tail_text_lines(
    source: Path,
    *,
    limit: int,
    encoding: str,
    errors: str,
) -> list[str]:
    """末尾側から非空行を最大 ``limit`` 件だけ読む."""
    tail_lines_reversed: list[bytes] = []
    with open(source, "rb") as f:
        f.seek(0, os.SEEK_END)
        position = f.tell()
        buffer = b""
        while position > 0 and len(tail_lines_reversed) < limit:
            read_size = min(_TAIL_READ_CHUNK_SIZE, position)
            position -= read_size
            f.seek(position)
            buffer = f.read(read_size) + buffer
            parts = buffer.splitlines()
            if position > 0 and parts:
                buffer = parts[0]
                parts = parts[1:]
            else:
                buffer = b""
            for raw_line in reversed(parts):
                if not raw_line.strip():
                    continue
                tail_lines_reversed.append(raw_line)
                if len(tail_lines_reversed) >= limit:
                    break
        if position == 0 and buffer.strip() and len(tail_lines_reversed) < limit:
            tail_lines_reversed.append(buffer)

    decoded_lines = [
        raw_line.decode(encoding, errors=errors)
        for raw_line in reversed(tail_lines_reversed)
    ]
    if decoded_lines and decoded_lines[0].startswith("\ufeff"):
        decoded_lines[0] = decoded_lines[0].removeprefix("\ufeff")
    return decoded_lines

def append_jsonl(
    path: PathLike,
    payloads: Iterable[object],
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    default: object = str,
) -> Path:
    """Append one or more records to a JSONL file."""
    target = ensure_parent_dir(path)
    with open(target, "a", encoding=encoding) as f:
        for payload in payloads:
            f.write(
                json.dumps(payload, ensure_ascii=ensure_ascii, default=default)
                + "\n"
            )
    return target
