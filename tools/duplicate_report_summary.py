#!/usr/bin/env python3
"""Print a compact summary of duplicate report analysis JSON."""

from pathlib import Path

from ztb.io.json_io import read_json_object
from ztb.utils.safety import ensure_dict, safe_to_float


def main() -> int:
    report_path = Path("reports/duplicate_report.json")
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return 1

    try:
        payload = read_json_object(report_path)
    except Exception as exc:
        print(f"Could not load report: {exc}")
        return 2

    exact_groups = ensure_dict(payload.get("exact_groups"))
    similar_pairs_raw = payload.get("similar_pairs")
    similar_pairs = similar_pairs_raw if isinstance(similar_pairs_raw, list) else []

    print(f"exact_groups: {len(exact_groups)}")
    print(f"similar_pairs: {len(similar_pairs)}")

    groups = sorted(
        [(group_hash, len(value) if isinstance(value, list) else 0) for group_hash, value in exact_groups.items()],
        key=lambda item: -item[1],
    )
    print("\nTop exact groups:")
    for group_hash, count in groups[:10]:
        print(f"  {group_hash} -> {count}")

    print("\nTop similar pairs:")
    sortable_pairs: list[tuple[float, dict[str, object]]] = []
    for pair_obj in similar_pairs:
        pair = ensure_dict(pair_obj)
        sortable_pairs.append((safe_to_float(pair.get("score"), 0.0), pair))
    sortable_pairs.sort(key=lambda item: -item[0])

    for score, pair in sortable_pairs[:10]:
        h1 = str(pair.get("h1", "unknown"))
        h2 = str(pair.get("h2", "unknown"))
        print(f"  {h1} ~ {h2} | score={score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
