#!/usr/bin/env python3
"""Remove exact duplicate code blocks based on the duplicate report.

Keeps one occurrence per group and removes the rest.
"""

from pathlib import Path
from typing import TypedDict

from ztb.io.json_io import read_json_object
from ztb.utils.safety import ensure_dict, safe_to_int


class DuplicateOccurrence(TypedDict):
    path: str
    start: int
    end: int
    name: str
    kind: str


class RemovalRange(TypedDict):
    start: int
    end: int
    group_hash: str


def load_report(report_path: Path) -> dict[str, object]:
    return read_json_object(report_path)


def _parse_occurrence(raw: object) -> DuplicateOccurrence | None:
    occ = ensure_dict(raw)
    path = occ.get("path")
    if not isinstance(path, str) or not path:
        return None

    start = safe_to_int(occ.get("start"), 0)
    end = safe_to_int(occ.get("end"), 0)
    if start <= 0 or end <= 0 or start > end:
        return None

    name = occ.get("name")
    kind = occ.get("kind")
    return {
        "path": path,
        "start": start,
        "end": end,
        "name": str(name) if name is not None else "",
        "kind": str(kind) if kind is not None else "",
    }


def _build_removal_plan(report: dict[str, object]) -> dict[str, list[RemovalRange]]:
    exact_groups = ensure_dict(report.get("exact_groups"))
    plan: dict[str, list[RemovalRange]] = {}

    for group_hash, occurrences_raw in exact_groups.items():
        if not isinstance(occurrences_raw, list):
            continue

        parsed_occurrences: list[DuplicateOccurrence] = []
        for raw in occurrences_raw:
            parsed = _parse_occurrence(raw)
            if parsed is not None:
                parsed_occurrences.append(parsed)

        if len(parsed_occurrences) <= 1:
            continue

        keep = parsed_occurrences[0]
        to_remove = parsed_occurrences[1:]
        print(
            f"Processing group {group_hash}: "
            f"keeping {keep['path']}:{keep['start']}-{keep['end']}, "
            f"removing {len(to_remove)} duplicates"
        )

        for occurrence in to_remove:
            plan.setdefault(occurrence["path"], []).append(
                {
                    "start": occurrence["start"],
                    "end": occurrence["end"],
                    "group_hash": group_hash,
                }
            )

    return plan


def _resolve_target_path(root: Path, rel_path: str) -> Path | None:
    raw_path = Path(rel_path)
    candidate = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        print(f"Warning: skipping out-of-root path {candidate}")
        return None
    return candidate


def _apply_file_removals(file_path: Path, removals: list[RemovalRange]) -> int:
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return 0

    removed = 0
    unique_ranges: dict[tuple[int, int], RemovalRange] = {}
    for removal in removals:
        key = (removal["start"], removal["end"])
        if key not in unique_ranges:
            unique_ranges[key] = removal

    for removal in sorted(unique_ranges.values(), key=lambda r: (r["start"], r["end"]), reverse=True):
        start_line = removal["start"] - 1
        end_line = removal["end"] - 1
        if start_line < 0 or end_line >= len(lines) or start_line > end_line:
            print(
                f"Invalid line range for {file_path}: "
                f"{start_line + 1}-{end_line + 1} "
                f"(group={removal['group_hash']})"
            )
            continue
        del lines[start_line : end_line + 1]
        removed += 1

    if removed == 0:
        return 0

    try:
        file_path.write_text("".join(lines), encoding="utf-8")
    except Exception as exc:
        print(f"Error writing {file_path}: {exc}")
        return 0

    print(f"Removed {removed} duplicate block(s) from {file_path}")
    return removed


def remove_duplicates(report: dict[str, object], root: Path) -> tuple[int, int]:
    """Remove duplicate occurrences, keeping one per group."""
    plan = _build_removal_plan(report)
    changed_files = 0
    removed_blocks = 0

    for rel_path, removals in plan.items():
        file_path = _resolve_target_path(root, rel_path)
        if file_path is None:
            continue
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping")
            continue

        removed = _apply_file_removals(file_path, removals)
        if removed > 0:
            changed_files += 1
            removed_blocks += removed

    return changed_files, removed_blocks


def main() -> int:
    root = Path(".")
    report_path = root / "reports" / "duplicate_report.json"

    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return 1

    report = load_report(report_path)
    changed_files, removed_blocks = remove_duplicates(report, root)
    print(f"Duplicate removal complete: files={changed_files}, blocks={removed_blocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
