"""
Plain JSONL read/write helpers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from ztb.io.common import PathLike, _to_path, ensure_parent_dir

logger = logging.getLogger(__name__)


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
