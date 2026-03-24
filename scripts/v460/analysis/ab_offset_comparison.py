"""441# A/B Offset 比較ツール: 440# regime-side offset asymmetry のデプロイ効果検証.

fill_records を Before (deploy前) / After (deploy後) で分割し、
regime×side 別に PnL / fill_rate / AS率 を比較。統計的有意性を t検定で評価。

Usage:
    # ベースライン保存 (deploy前)
    python scripts/v460/analysis/ab_offset_comparison.py --save-baseline

    # deploy後の比較 (n日分の After データ蓄積後)
    python scripts/v460/analysis/ab_offset_comparison.py --compare

    # 特定日で分割
    python scripts/v460/analysis/ab_offset_comparison.py --compare --split-date 2026-03-16

    # 表示のみ (保存しない)
    python scripts/v460/analysis/ab_offset_comparison.py --show-baseline

    # 特定 git SHA のレコードのみで比較 (451# P0-4)
    python scripts/v460/analysis/ab_offset_comparison.py --compare --git-sha 52627f
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    apply_fill_record_filters,
    iter_fill_record_objects_glob,
)
from ztb.utils.safety import safe_to_finite
from scripts.v460.analysis.analysis_common import write_json_output
from scripts.v460.analysis.analysis_common import add_results_dir_arg

# ── Constants ──
_RESULTS_DIR = Path("results/v460/fill_test")
_BASELINE_PATH = Path("results/v460/baseline_440_offset_ab.json")
_COMPARISON_PATH = Path("results/v460/comparison_440_offset_ab.json")

# 440# target buckets: regime×side combinations we're monitoring
_TARGET_BUCKETS = [
    ("ranging", "buy"),   # discount 0.90→1.15 (worst bucket, n=1321)
    ("ranging", "sell"),  # discount 0.90→0.85
    ("unknown", "buy"),   # boost 2.0 (existing)
    ("unknown", "sell"),  # boost 1.3 (new)
]


class BucketMetrics(TypedDict):
    """regime×side バケットの集計メトリクス."""
    regime: str
    side: str
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: float
    std_pnl30_bps: float
    as_rate: float
    downside_p10_bps: float
    pnl_values: list[float]  # 個別 PnL (t検定用)


class ComparisonRow(TypedDict):
    """Before/After 比較の1行."""
    regime: str
    side: str
    before_n: int
    after_n: int
    before_pnl30: float
    after_pnl30: float
    pnl_diff: float
    before_fill_rate: float
    after_fill_rate: float
    fill_rate_diff: float
    before_as_rate: float
    after_as_rate: float
    t_statistic: float | None
    p_value: float | None
    significant: bool


def _load_records(
    results_dir: Path,
    *,
    git_sha: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, object]]:
    """fill_records を全ファイルからロード (git_sha/run_id フィルタ対応)."""
    raw = list(iter_fill_record_objects_glob(str(results_dir)))
    if git_sha or run_id:
        filtered, _ = apply_fill_record_filters(
            raw, git_sha=git_sha, run_id=run_id,
        )
        return cast(list[dict[str, object]], filtered)
    return raw


def _compute_bucket_metrics(records: list[dict[str, object]]) -> dict[str, BucketMetrics]:
    """レコードを regime×side でグルーピングし、バケット別メトリクスを算出."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in records:
        regime = str(r.get("regime") or "none")
        side = str(r.get("side", "unknown"))
        key = f"{regime}:{side}"
        groups[key].append(r)

    result: dict[str, BucketMetrics] = {}
    for key, recs in groups.items():
        regime, side = key.split(":", 1)
        filled = [r for r in recs if r.get("filled")]
        pnl_values: list[float] = []
        as_count = 0
        for r in filled:
            pnl = safe_to_finite(r.get("post_fill_30s_pnl"))
            if pnl is not None:
                pnl_values.append(pnl)
            if r.get("adverse_selected"):
                as_count += 1

        n_total = len(recs)
        n_filled = len(filled)
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
        std_pnl = _std(pnl_values) if len(pnl_values) > 1 else 0.0
        p10 = sorted(pnl_values)[max(0, len(pnl_values) // 10)] if pnl_values else 0.0

        result[key] = BucketMetrics(
            regime=regime,
            side=side,
            n_total=n_total,
            n_filled=n_filled,
            fill_rate=n_filled / n_total if n_total > 0 else 0.0,
            avg_pnl30_bps=avg_pnl,
            std_pnl30_bps=std_pnl,
            as_rate=as_count / n_filled if n_filled > 0 else 0.0,
            downside_p10_bps=p10,
            pnl_values=pnl_values,
        )
    return result


def _std(values: list[float]) -> float:
    """標本標準偏差."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _welch_t_test(
    vals_a: list[float], vals_b: list[float]
) -> tuple[float | None, float | None]:
    """Welch の t 検定 (等分散を仮定しない)."""
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a < 5 or n_b < 5:
        return None, None

    mean_a = sum(vals_a) / n_a
    mean_b = sum(vals_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in vals_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in vals_b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return None, None

    t_stat = (mean_b - mean_a) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if denom < 1e-12:
        return t_stat, None
    df = num / denom

    # p-value approximation (two-tailed, using t-distribution CDF approx)
    try:
        from scipy.stats import t as t_dist
        p_value = float(2 * t_dist.sf(abs(t_stat), df))
    except ImportError:
        # Fallback: rough normal approximation for large df
        from math import erfc
        p_value = erfc(abs(t_stat) / math.sqrt(2))

    return t_stat, p_value


def compare_buckets(
    before: dict[str, BucketMetrics],
    after: dict[str, BucketMetrics],
    target_buckets: list[tuple[str, str]] | None = None,
) -> list[ComparisonRow]:
    """Before/After のバケット別比較."""
    buckets = target_buckets or _TARGET_BUCKETS
    rows: list[ComparisonRow] = []

    for regime, side in buckets:
        key = f"{regime}:{side}"
        b = before.get(key)
        a = after.get(key)

        if b is None and a is None:
            continue

        b_n = b["n_filled"] if b else 0
        a_n = a["n_filled"] if a else 0
        b_pnl = b["avg_pnl30_bps"] if b else 0.0
        a_pnl = a["avg_pnl30_bps"] if a else 0.0
        b_fr = b["fill_rate"] if b else 0.0
        a_fr = a["fill_rate"] if a else 0.0
        b_as = b["as_rate"] if b else 0.0
        a_as = a["as_rate"] if a else 0.0

        t_stat, p_val = _welch_t_test(
            b.get("pnl_values", []) if b else [],
            a.get("pnl_values", []) if a else [],
        )

        rows.append(ComparisonRow(
            regime=regime,
            side=side,
            before_n=b_n,
            after_n=a_n,
            before_pnl30=round(b_pnl, 3),
            after_pnl30=round(a_pnl, 3),
            pnl_diff=round(a_pnl - b_pnl, 3),
            before_fill_rate=round(b_fr, 4),
            after_fill_rate=round(a_fr, 4),
            fill_rate_diff=round(a_fr - b_fr, 4),
            before_as_rate=round(b_as, 4),
            after_as_rate=round(a_as, 4),
            t_statistic=round(t_stat, 3) if t_stat is not None else None,
            p_value=round(p_val, 4) if p_val is not None else None,
            significant=p_val < 0.05 if p_val is not None else False,
        ))

    return rows


def _print_comparison(rows: list[ComparisonRow]) -> None:
    """比較結果をテーブル表示."""
    hdr = (
        f"{'regime':15s} {'side':5s}  "
        f"{'n_B':>5s} {'n_A':>5s}  "
        f"{'pnl_B':>7s} {'pnl_A':>7s} {'Δpnl':>7s}  "
        f"{'fr_B':>5s} {'fr_A':>5s} {'Δfr':>6s}  "
        f"{'p-val':>6s} {'sig':>3s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sig_mark = "***" if r["significant"] else "   "
        p_str = f"{r['p_value']:.4f}" if r["p_value"] is not None else "  N/A "
        print(
            f"{r['regime']:15s} {r['side']:5s}  "
            f"{r['before_n']:5d} {r['after_n']:5d}  "
            f"{r['before_pnl30']:+7.3f} {r['after_pnl30']:+7.3f} {r['pnl_diff']:+7.3f}  "
            f"{r['before_fill_rate']:5.1%} {r['after_fill_rate']:5.1%} {r['fill_rate_diff']:+5.1%}  "
            f"{p_str} {sig_mark}"
        )


def _save_baseline(
    results_dir: Path,
    output_path: Path,
    *,
    git_sha: str | None = None,
    run_id: str | None = None,
) -> None:
    """Before (deploy前) のベースラインを保存."""
    records = _load_records(results_dir, git_sha=git_sha, run_id=run_id)
    metrics = _compute_bucket_metrics(records)

    # pnl_values は巨大なので保存時は統計サマリのみ
    serializable: dict[str, object] = {}
    for key, m in metrics.items():
        entry = dict(m)
        entry["pnl_count"] = len(m["pnl_values"])
        del entry["pnl_values"]
        serializable[key] = entry

    payload = {
        "type": "baseline_440_offset",
        "created": datetime.now(timezone.utc).isoformat(),
        "filters": {"git_sha": git_sha, "run_id": run_id},
        "total_records": len(records),
        "total_filled": sum(1 for r in records if r.get("filled")),
        "buckets": serializable,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_output(payload, output_path)
    print(f"Baseline saved: {output_path} ({len(records)} records)")


def _show_baseline(
    results_dir: Path,
    *,
    git_sha: str | None = None,
    run_id: str | None = None,
) -> None:
    """ベースラインの regime×side サマリを表示."""
    records = _load_records(results_dir, git_sha=git_sha, run_id=run_id)
    metrics = _compute_bucket_metrics(records)

    print(f"Total records: {len(records)}")
    print(f"Total filled:  {sum(1 for r in records if r.get('filled'))}")
    print()
    hdr = f"{'regime':15s} {'side':5s}  {'n':>5s}  {'fill%':>6s}  {'pnl30':>9s}  {'AS%':>5s}  {'p10':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for regime, side in _TARGET_BUCKETS:
        key = f"{regime}:{side}"
        m = metrics.get(key)
        if m is None:
            print(f"{regime:15s} {side:5s}  (no data)")
            continue
        print(
            f"{regime:15s} {side:5s}  {m['n_filled']:5d}  {m['fill_rate']:5.1%}  "
            f"{m['avg_pnl30_bps']:+9.3f}  {m['as_rate']:4.1%}  {m['downside_p10_bps']:+8.3f}"
        )


def _run_comparison(
    results_dir: Path,
    split_date: str | None,
    output_path: Path,
    *,
    git_sha: str | None = None,
    run_id: str | None = None,
) -> None:
    """Before/After 比較を実行."""
    records = _load_records(results_dir, git_sha=git_sha, run_id=run_id)

    if split_date:
        # 日付文字列 (YYYYMMDD or YYYY-MM-DD) で分割
        split_str = split_date.replace("-", "")
        before_recs: list[dict[str, object]] = []
        after_recs: list[dict[str, object]] = []
        for r in records:
            ts = safe_to_finite(r.get("timestamp"))
            if ts is None:
                continue
            from ztb.data.raw_paths import utc_day_str_from_timestamp
            day = utc_day_str_from_timestamp(ts)
            if day < split_str:
                before_recs.append(r)
            else:
                after_recs.append(r)
    else:
        # デフォルト: 最新日のみを After、残りを Before
        from ztb.metrics.fill_quality import format_utc_day

        days: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in records:
            ts = safe_to_finite(r.get("timestamp"))
            if ts is None:
                continue
            day = format_utc_day(ts)
            if day:
                days[day].append(r)

        if len(days) < 2:
            print("ERROR: Need at least 2 days of data for comparison")
            return

        sorted_days = sorted(days.keys())
        latest = sorted_days[-1]
        before_recs = []
        after_recs = days[latest]
        for d in sorted_days[:-1]:
            before_recs.extend(days[d])

    print(f"Before: {len(before_recs)} records")
    print(f"After:  {len(after_recs)} records")
    print()

    before_metrics = _compute_bucket_metrics(before_recs)
    after_metrics = _compute_bucket_metrics(after_recs)

    # 全バケットで比較
    all_buckets = set()
    for key in list(before_metrics.keys()) + list(after_metrics.keys()):
        regime, side = key.split(":", 1)
        if side in ("buy", "sell"):
            all_buckets.add((regime, side))

    rows = compare_buckets(before_metrics, after_metrics, sorted(all_buckets))

    print("=== 440# Offset A/B Comparison ===")
    print()

    # Target buckets first
    target_rows = [r for r in rows if (r["regime"], r["side"]) in _TARGET_BUCKETS]
    other_rows = [r for r in rows if (r["regime"], r["side"]) not in _TARGET_BUCKETS]

    if target_rows:
        print("── Target Buckets (440# modified) ──")
        _print_comparison(target_rows)

    if other_rows:
        print()
        print("── Other Buckets (control) ──")
        _print_comparison(other_rows)

    # 全体 PnL
    b_pnl_all = [v for m in before_metrics.values() for v in m["pnl_values"]]
    a_pnl_all = [v for m in after_metrics.values() for v in m["pnl_values"]]
    b_avg = sum(b_pnl_all) / len(b_pnl_all) if b_pnl_all else 0.0
    a_avg = sum(a_pnl_all) / len(a_pnl_all) if a_pnl_all else 0.0
    print()
    print(f"Overall PnL30: Before={b_avg:+.3f} bps, After={a_avg:+.3f} bps, Δ={a_avg - b_avg:+.3f} bps")

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_output(
        {
            "type": "comparison_440_offset",
            "created": datetime.now(timezone.utc).isoformat(),
            "split_date": split_date,
            "filters": {"git_sha": git_sha, "run_id": run_id},
            "before_n": len(before_recs),
            "after_n": len(after_recs),
            "rows": rows,
            "overall_before_pnl30": round(b_avg, 4),
            "overall_after_pnl30": round(a_avg, 4),
        },
        output_path,
    )
    print(f"\nSaved: {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="441# A/B Offset Comparison")
    add_results_dir_arg(parser)
    parser.add_argument("--save-baseline", action="store_true", help="Save baseline (before deploy)")
    parser.add_argument("--show-baseline", action="store_true", help="Show current baseline metrics")
    parser.add_argument("--compare", action="store_true", help="Run Before/After comparison")
    parser.add_argument("--split-date", type=str, default=None, help="Split date (YYYY-MM-DD)")
    parser.add_argument("--git-sha", help="git_sha 前方一致フィルタ (短縮 SHA 可)")
    parser.add_argument("--run-id", help="run_id 完全一致フィルタ")
    parser.add_argument("--output", default=str(_COMPARISON_PATH))
    parser.add_argument("--baseline-path", default=str(_BASELINE_PATH))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)

    if args.save_baseline:
        _save_baseline(
            results_dir, Path(args.baseline_path),
            git_sha=args.git_sha, run_id=args.run_id,
        )
    elif args.show_baseline:
        _show_baseline(results_dir, git_sha=args.git_sha, run_id=args.run_id)
    elif args.compare:
        _run_comparison(
            results_dir, args.split_date, Path(args.output),
            git_sha=args.git_sha, run_id=args.run_id,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
