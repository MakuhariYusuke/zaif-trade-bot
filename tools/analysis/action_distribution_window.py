#!/usr/bin/env python3
"""
Simple tool to aggregate action distributions from training reports and evaluation logs.
"""
import argparse
import json
import gzip
from pathlib import Path
from typing import Dict, List


def _extract_from_report_file(path: Path, start: int, end: int) -> List[Dict[str, float]]:
    results = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return results

    # Traverse and collect any action_distribution entries
    def _collect(obj):
        if isinstance(obj, dict):
            if 'action_distribution' in obj and 'step' in obj:
                try:
                    step = int(obj['step'])
                except Exception:
                    return
                if start <= step < end:
                    results.append(obj['action_distribution'])
            # Recurse
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    _collect(data)
    return results


def _extract_from_jsonl_gz(path: Path, start: int, end: int) -> List[Dict[str, float]]:
    results = []
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Many evaluation log lines have step/time
                if 'action_distribution' in obj and 'step' in obj:
                    try:
                        step = int(obj['step'])
                    except Exception:
                        continue
                    if start <= step < end:
                        results.append(obj['action_distribution'])
    except Exception:
        pass
    return results


def aggregate(distributions: List[Dict[str, float]]) -> Dict[str, float]:
    if not distributions:
        return {}
    keys = ['HOLD', 'BUY', 'SELL']
    sums = {k: 0.0 for k in keys}
    for d in distributions:
        for k in keys:
            sums[k] += float(d.get(k, 0.0))
    n = len(distributions)
    return {k: sums[k] / n for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', default='reports', help='Path to reports directory')
    parser.add_argument('--eval-logs', default='logs', help='Path to logs or evaluation folder')
    parser.add_argument('--start', type=int, default=1000)
    parser.add_argument('--end', type=int, default=2000)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    eval_dir = Path(args.eval_logs)

    collected = []

    # Read JSON reports
    if reports_dir.exists():
        for p in reports_dir.glob('training_report_*.json'):
            collected.extend(_extract_from_report_file(p, args.start, args.end))

    # Read jsonl.gz evaluation logs
    if eval_dir.exists():
        for p in eval_dir.glob('evaluation_history*.jsonl.gz'):
            collected.extend(_extract_from_jsonl_gz(p, args.start, args.end))

    agg = aggregate(collected)
    if not agg:
        print('No action_distribution entries found for steps in range')
    else:
        print(f"Action distribution aggregate for steps [{args.start}, {args.end}):")
        print(f"  HOLD: {agg.get('HOLD', 0.0):.3f}")
        print(f"  BUY:  {agg.get('BUY', 0.0):.3f}")
        print(f"  SELL: {agg.get('SELL', 0.0):.3f}")
        print(f"(Based on {len(collected)} evaluation snapshots)")


if __name__ == '__main__':
    main()
