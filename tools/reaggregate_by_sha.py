#!/usr/bin/env python3
"""363# A5: current-SHA 限定 fill test 再集計 (SSOT).

361# P0-1 / 362#: mixed-SHA 問題を解消し、特定 SHA のみの
K1/K2/PnL gate メトリクスを算出する.

Usage:
    # 現在の git SHA で集計
    python tools/reaggregate_by_sha.py

    # 特定 SHA を指定
    python tools/reaggregate_by_sha.py --sha 819ec73b2081

    # 日付範囲を絞る
    python tools/reaggregate_by_sha.py --sha 819ec73b2081 --date-from 2026-03-07 --date-to 2026-03-09

    # JSON 出力
    python tools/reaggregate_by_sha.py --json -o results/v460/sha_reagg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Project imports ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (  # noqa: E402
    load_fill_record_objects_glob,
)
from ztb.metrics.fill_quality import (  # noqa: E402
    apply_fill_record_filters,
)
from ztb.utils.git_utils import get_git_sha  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHA 限定 fill test 再集計 (363# A5 SSOT)",
    )
    p.add_argument(
        "--sha",
        help="git SHA (前方一致). 省略時は現在の HEAD SHA",
    )
    p.add_argument(
        "--data-dir",
        default="results/v460/fill_test",
        help="fill_records_*.jsonl の格納ディレクトリ",
    )
    p.add_argument("--date-from", help="開始日 inclusive (YYYY-MM-DD)")
    p.add_argument("--date-to", help="終了日 inclusive (YYYY-MM-DD)")
    p.add_argument("--json", action="store_true", help="JSON 形式で出力")
    p.add_argument("-o", "--output", help="出力先ファイル (省略時: stdout)")
    return p.parse_args()


def _compute_gate_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """K1/K2/PnL gate 相当のメトリクスを算出."""
    n_total = len(records)
    if n_total == 0:
        return {"error": "no records", "n_total": 0}

    filled = [r for r in records if r.get("filled")]
    skipped = [r for r in records if r.get("skip_gate_skipped")]
    n_filled = len(filled)
    n_skipped = len(skipped)
    n_attempted = n_total - n_skipped

    # ── K1: attempted fill rate ──
    attempted_fill_rate = n_filled / n_attempted if n_attempted > 0 else 0.0
    overall_fill_rate = n_filled / n_total if n_total > 0 else 0.0

    # ── K2: attempted cancel ratio ──
    cancelled = [
        r for r in records
        if not r.get("filled") and not r.get("skip_gate_skipped")
    ]
    attempted_cancel_ratio = len(cancelled) / n_attempted if n_attempted > 0 else 0.0

    # ── PnL stats ──
    pnl_values = [
        float(r["post_fill_30s_pnl"])
        for r in filled
        if r.get("post_fill_30s_pnl") is not None
    ]
    pnl_arr = np.array(pnl_values, dtype=np.float64) if pnl_values else np.array([])

    pnl_mean = float(np.mean(pnl_arr)) if len(pnl_arr) > 0 else 0.0
    pnl_median = float(np.median(pnl_arr)) if len(pnl_arr) > 0 else 0.0
    pnl_std = float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0
    profitable_ratio = float(np.mean(pnl_arr > 0)) if len(pnl_arr) > 0 else 0.0

    # ── Side 別 ──
    sides: dict[str, dict[str, Any]] = {}
    for side in ("buy", "sell"):
        s_all = [r for r in records if r.get("side") == side]
        s_filled = [r for r in s_all if r.get("filled")]
        s_skipped = [r for r in s_all if r.get("skip_gate_skipped")]
        s_attempted = len(s_all) - len(s_skipped)
        s_pnl = [
            float(r["post_fill_30s_pnl"])
            for r in s_filled
            if r.get("post_fill_30s_pnl") is not None
        ]
        s_pnl_arr = np.array(s_pnl, dtype=np.float64) if s_pnl else np.array([])
        sides[side] = {
            "total": len(s_all),
            "filled": len(s_filled),
            "attempted": s_attempted,
            "fill_rate_attempted": len(s_filled) / s_attempted if s_attempted > 0 else 0.0,
            "pnl_mean_bps": float(np.mean(s_pnl_arr)) if len(s_pnl_arr) > 0 else 0.0,
            "pnl_count": len(s_pnl_arr),
        }

    # ── Adverse Selection ──
    as_count = sum(1 for r in filled if r.get("adverse_selected"))
    as_ratio = as_count / n_filled if n_filled > 0 else 0.0

    # ── Cancel Reason Top 5 ──
    import collections

    cancel_reasons = collections.Counter(
        r.get("cancel_reason", "unknown") for r in cancelled
    )

    return {
        "n_total": n_total,
        "n_filled": n_filled,
        "n_skipped": n_skipped,
        "n_attempted": n_attempted,
        "n_cancelled": len(cancelled),
        "k1_attempted_fill_rate": round(attempted_fill_rate, 4),
        "k1_overall_fill_rate": round(overall_fill_rate, 4),
        "k2_attempted_cancel_ratio": round(attempted_cancel_ratio, 4),
        "pnl_mean_bps": round(pnl_mean, 3),
        "pnl_median_bps": round(pnl_median, 3),
        "pnl_std_bps": round(pnl_std, 3),
        "pnl_count": len(pnl_arr),
        "profitable_ratio": round(profitable_ratio, 4),
        "as_ratio": round(as_ratio, 4),
        "as_count": as_count,
        "side": sides,
        "cancel_top5": dict(cancel_reasons.most_common(5)),
    }


def main() -> None:
    args = _parse_args()

    # ── SHA 解決 ──
    sha = args.sha or get_git_sha(cwd=_PROJECT_ROOT)[:12]
    print(f"[reaggregate] SHA filter: {sha}", file=sys.stderr)

    # ── データ読込 ──
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    raw = load_fill_record_objects_glob(
        data_dir,
        start_date=args.date_from,
        end_date=args.date_to,
    )
    if not raw:
        print(f"ERROR: no fill_records found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # ── SHA フィルタ ──
    filtered, _ = apply_fill_record_filters(raw, git_sha=sha)
    print(
        f"[reaggregate] {len(raw)} total → {len(filtered)} (SHA={sha})",
        file=sys.stderr,
    )

    if not filtered:
        print(f"ERROR: no records match SHA={sha}", file=sys.stderr)
        sys.exit(1)

    # ── メトリクス算出 ──
    metrics = _compute_gate_metrics(filtered)

    # ── SHA 分布 (品質確認用) ──
    import collections

    sha_dist = collections.Counter(
        str(r.get("git_sha", "?"))[:12] for r in filtered
    )

    result = {
        "sha_filter": sha,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "date_range": {
            "from": args.date_from,
            "to": args.date_to,
        },
        "sha_distribution": dict(sha_dist.most_common(5)),
        "metrics": metrics,
    }

    # ── 出力 ──
    if args.json:
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = _format_text(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"[reaggregate] saved to {out_path}", file=sys.stderr)
    else:
        print(output)


def _format_text(result: dict[str, Any]) -> str:
    """テキスト形式での出力."""
    m = result["metrics"]
    lines = [
        "=" * 60,
        f"Fill Test SHA 限定再集計 (363# A5 SSOT)",
        "=" * 60,
        f"  SHA       : {result['sha_filter']}",
        f"  Generated : {result['generated_utc']}",
        f"  Data Dir  : {result['data_dir']}",
        f"  Date Range: {result['date_range']['from'] or '(all)'} ~ {result['date_range']['to'] or '(all)'}",
        "",
        "## 母数",
        f"  Total: {m['n_total']}, Filled: {m['n_filled']}, "
        f"Skipped: {m['n_skipped']}, Attempted: {m['n_attempted']}",
        "",
        "## K1/K2 Gate Metrics",
        f"  K1 attempted_fill_rate : {m['k1_attempted_fill_rate']:.4f}  (threshold: ≥ 0.60)",
        f"  K1 overall_fill_rate   : {m['k1_overall_fill_rate']:.4f}",
        f"  K2 cancel_ratio        : {m['k2_attempted_cancel_ratio']:.4f}  (threshold: ≤ 0.40)",
        "",
        "## PnL (post_fill_30s, bps)",
        f"  mean   : {m['pnl_mean_bps']:+.3f}",
        f"  median : {m['pnl_median_bps']:+.3f}",
        f"  std    : {m['pnl_std_bps']:.3f}",
        f"  n      : {m['pnl_count']}",
        f"  profit%: {m['profitable_ratio']*100:.1f}%",
        "",
        "## Adverse Selection",
        f"  AS ratio: {m['as_ratio']:.4f} ({m['as_count']} / {m['n_filled']})",
        "",
        "## Side 別",
    ]
    for side, sd in m.get("side", {}).items():
        lines.append(
            f"  {side}: {sd['total']} total, {sd['filled']} filled, "
            f"fill_rate_att={sd['fill_rate_attempted']:.4f}, "
            f"pnl_mean={sd['pnl_mean_bps']:+.3f}bps"
        )
    lines.append("")
    lines.append("## Cancel Reason Top 5")
    for reason, cnt in m.get("cancel_top5", {}).items():
        lines.append(f"  {reason}: {cnt}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
