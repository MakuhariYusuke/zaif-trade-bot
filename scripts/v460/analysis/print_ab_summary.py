#!/usr/bin/env python3
"""
Print a human-friendly summary of AB or parameter-search results.

Usage:
  python -m scripts.v460.analysis.print_ab_summary --file reports/ab_search_balance_shaping.json --top 5

It reads the JSON file written by `tools/ab_param_search.py` or `ab_test_runner` summary
and prints the top-N candidates with key metrics.
"""
import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from scripts.v460.analysis.analysis_common import write_output

JsonObject: TypeAlias = dict[str, object]
JsonRow: TypeAlias = dict[str, object]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to summary json file")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    p = Path(args.file)
    if not p.exists():
        write_output(f"File not found: {p}")
        return

    obj = json.loads(p.read_text(encoding="utf-8"))

    # If ab_param_search wrote a list: print top rows
    if isinstance(obj, list):
        rows = [row for row in obj if isinstance(row, dict)]
    else:
        # UnifiedOptimizer writes {best:..., score:...}
        if "best_params" in obj or "best" in obj:
            write_output(json.dumps(obj, indent=2, ensure_ascii=False))
            return
        # fallback: try to interpret as list under key 'results'
        rows = [
            row for row in obj.get("results", [])
            if isinstance(obj, dict) and isinstance(row, dict)
        ] if isinstance(obj, dict) else []

    if not rows:
        write_output("No results found in summary")
        return

    lines = ["Top candidates by score:"]
    for i, r in enumerate(rows[: args.top]):
        params = r.get("params", r.get("best", {}))
        avg = r.get("avg_distribution")
        score = float(r.get("score", 0.0))
        lines.append(f"[{i+1}] score: {score:.5f}")
        if isinstance(avg, dict):
            hold = float(avg.get("HOLD", 0.0))
            buy = float(avg.get("BUY", 0.0))
            sell = float(avg.get("SELL", 0.0))
            lines.append(f"   HOLD: {hold:.3f}, BUY: {buy:.3f}, SELL: {sell:.3f}")
        else:
            lines.append("   (no action distribution)")
        lines.append("   params:")
        if isinstance(params, dict):
            for k, v in params.items():
                lines.append(f"     - {k}: {v}")
    write_output("\n".join(lines))


if __name__ == "__main__":
    main()
