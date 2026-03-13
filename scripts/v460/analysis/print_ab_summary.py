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
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to summary json file")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    p = Path(args.file)
    if not p.exists():
        print(f"File not found: {p}")
        return

    obj = json.loads(p.read_text(encoding="utf-8"))

    # If ab_param_search wrote a list: print top rows
    if isinstance(obj, list):
        rows = obj
    else:
        # UnifiedOptimizer writes {best:..., score:...}
        if "best_params" in obj or "best" in obj:
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            return
        # fallback: try to interpret as list under key 'results'
        rows = obj.get("results", []) if isinstance(obj, dict) else []

    if not rows:
        print("No results found in summary")
        return

    print("Top candidates by score:")
    for i, r in enumerate(rows[: args.top]):
        params = r.get("params", r.get("best", {}))
        avg = r.get("avg_distribution")
        score = r.get("score")
        print(f"[{i+1}] score: {score:.5f}")
        if avg:
            print(f"   HOLD: {avg.get('HOLD', 0.0):.3f}, BUY: {avg.get('BUY', 0.0):.3f}, SELL: {avg.get('SELL', 0.0):.3f}")
        else:
            print("   (no action distribution)")
        print("   params:")
        for k, v in params.items():
            print(f"     - {k}: {v}")


if __name__ == "__main__":
    main()
