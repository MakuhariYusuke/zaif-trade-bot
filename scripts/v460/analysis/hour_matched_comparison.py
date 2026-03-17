"""467# Hour-Matched Comparison: 時間帯固定での SHA/config 純粋効果分析.

462# 残課題: 時間帯の交絡因子を排除し、コード/設定変更の純粋効果を抽出する。
同一 UTC hour 内で SHA/config を比較 → 時間帯 AS 率変動の影響を除去。

Usage:
    # SHA 比較 (2つ以上の SHA を指定)
    python scripts/v460/analysis/hour_matched_comparison.py --sha abc1234 def5678

    # config_hash 比較
    python scripts/v460/analysis/hour_matched_comparison.py --config-hash a1b2c3 d4e5f6

    # JSON 出力
    python scripts/v460/analysis/hour_matched_comparison.py --sha abc def --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import load_fill_record_objects_glob
from ztb.utils.safety import safe_to_finite

# ── Constants ──
_RESULTS_DIR = _PROJECT_ROOT / "results" / "v460" / "fill_test"
_OUTPUT_DIR = _PROJECT_ROOT / "analysis_results"


class HourBucketMetrics(TypedDict):
    """1 UTC hour × 1 variant のメトリクス."""

    utc_hour: int
    variant: str
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl_bps: float
    as_rate: float
    pnl_values: list[float]


class HourComparisonRow(TypedDict):
    """1 UTC hour の A vs B 比較."""

    utc_hour: int
    jst_hour: int
    a_n: int
    b_n: int
    a_fill_rate: float
    b_fill_rate: float
    fill_rate_diff: float
    a_pnl_bps: float
    b_pnl_bps: float
    pnl_diff_bps: float
    a_as_rate: float
    b_as_rate: float
    as_rate_diff: float
    t_stat: float | None
    p_value: float | None


def _get_pnl(r: dict[str, object]) -> float | None:
    for key in ("ev_weighted_pnl", "post_fill_30s_pnl", "pnl_bps"):
        v = r.get(key)
        if v is not None:
            val = safe_to_finite(v)
            if val is not None:
                return float(val)
    return None


def _utc_hour_from_record(r: dict[str, object]) -> int | None:
    ts = r.get("start_ts")
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).hour
    except (ValueError, TypeError, OSError):
        return None


def _match_prefix(value: str, prefix: str) -> bool:
    return value[:len(prefix)] == prefix


def _welch_t_test(
    a: list[float], b: list[float],
) -> tuple[float | None, float | None]:
    n_a, n_b = len(a), len(b)
    if n_a < 5 or n_b < 5:
        return None, None
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return None, None
    t_stat = (mean_b - mean_a) / se
    # Welch-Satterthwaite df
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if denom < 1e-15:
        return t_stat, None
    df = num / denom
    # 近似 p-value (scipy 不要)
    x = df / (df + t_stat ** 2)
    p_approx = x ** (df / 2) if abs(t_stat) < 10 else 0.0
    return t_stat, p_approx


def _compute_bucket(
    records: list[dict[str, object]], variant: str, utc_hour: int,
) -> HourBucketMetrics:
    filled = [r for r in records if r.get("filled")]
    pnl_values: list[float] = []
    as_count = 0
    for r in filled:
        pnl = _get_pnl(r)
        if pnl is not None:
            pnl_values.append(pnl)
        if r.get("adverse_selected"):
            as_count += 1
    n_total = len(records)
    n_filled = len(filled)
    return HourBucketMetrics(
        utc_hour=utc_hour,
        variant=variant,
        n_total=n_total,
        n_filled=n_filled,
        fill_rate=n_filled / n_total if n_total > 0 else 0.0,
        avg_pnl_bps=sum(pnl_values) / len(pnl_values) if pnl_values else 0.0,
        as_rate=as_count / n_filled if n_filled > 0 else 0.0,
        pnl_values=pnl_values,
    )


def run_hour_matched_comparison(
    variant_a: str,
    variant_b: str,
    *,
    key_field: str = "git_sha",
    results_dir: Path | None = None,
    side_filter: str | None = None,
) -> dict[str, object]:
    """時間帯固定 A/B 比較を実行.

    Args:
        variant_a: A variant の SHA prefix or config_hash
        variant_b: B variant の SHA prefix or config_hash
        key_field: 比較キー ("git_sha" or "config_hash")
        results_dir: fill_records ディレクトリ
        side_filter: "buy" or "sell" で絞り込み (None=両方)
    """
    if results_dir is None:
        results_dir = _RESULTS_DIR

    all_records = load_fill_record_objects_glob(str(results_dir))
    if side_filter:
        all_records = [r for r in all_records if r.get("side") == side_filter]

    # variant 分類
    groups_a: dict[int, list[dict[str, object]]] = defaultdict(list)
    groups_b: dict[int, list[dict[str, object]]] = defaultdict(list)
    n_unmatched = 0

    for r in all_records:
        utc_h = _utc_hour_from_record(r)
        if utc_h is None:
            continue
        val = str(r.get(key_field, "") or "")
        if _match_prefix(val, variant_a):
            groups_a[utc_h].append(r)
        elif _match_prefix(val, variant_b):
            groups_b[utc_h].append(r)
        else:
            n_unmatched += 1

    # 両方にデータがある hour で比較
    common_hours = sorted(set(groups_a.keys()) & set(groups_b.keys()))
    rows: list[HourComparisonRow] = []
    all_pnl_a: list[float] = []
    all_pnl_b: list[float] = []

    for h in common_hours:
        m_a = _compute_bucket(groups_a[h], variant_a, h)
        m_b = _compute_bucket(groups_b[h], variant_b, h)
        t_stat, p_val = _welch_t_test(m_a["pnl_values"], m_b["pnl_values"])
        all_pnl_a.extend(m_a["pnl_values"])
        all_pnl_b.extend(m_b["pnl_values"])
        rows.append(HourComparisonRow(
            utc_hour=h,
            jst_hour=(h + 9) % 24,
            a_n=m_a["n_total"],
            b_n=m_b["n_total"],
            a_fill_rate=m_a["fill_rate"],
            b_fill_rate=m_b["fill_rate"],
            fill_rate_diff=m_b["fill_rate"] - m_a["fill_rate"],
            a_pnl_bps=m_a["avg_pnl_bps"],
            b_pnl_bps=m_b["avg_pnl_bps"],
            pnl_diff_bps=m_b["avg_pnl_bps"] - m_a["avg_pnl_bps"],
            a_as_rate=m_a["as_rate"],
            b_as_rate=m_b["as_rate"],
            as_rate_diff=m_b["as_rate"] - m_a["as_rate"],
            t_stat=t_stat,
            p_value=p_val,
        ))

    # 全体集計
    overall_t, overall_p = _welch_t_test(all_pnl_a, all_pnl_b)

    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "key_field": key_field,
        "side_filter": side_filter,
        "n_hours_compared": len(common_hours),
        "n_a_total": sum(len(groups_a[h]) for h in common_hours),
        "n_b_total": sum(len(groups_b[h]) for h in common_hours),
        "n_unmatched": n_unmatched,
        "overall_pnl_a": sum(all_pnl_a) / len(all_pnl_a) if all_pnl_a else 0.0,
        "overall_pnl_b": sum(all_pnl_b) / len(all_pnl_b) if all_pnl_b else 0.0,
        "overall_pnl_diff": (
            (sum(all_pnl_b) / len(all_pnl_b) - sum(all_pnl_a) / len(all_pnl_a))
            if all_pnl_a and all_pnl_b else 0.0
        ),
        "overall_t_stat": overall_t,
        "overall_p_value": overall_p,
        "by_hour": rows,
    }


def _print_report(result: dict[str, object]) -> None:
    """人間可読レポートを標準出力に表示."""
    print("=" * 80)
    print("  467# Hour-Matched Comparison")
    print(f"  A: {result['variant_a']}  vs  B: {result['variant_b']}")
    print(f"  Key: {result['key_field']}  Side: {result['side_filter'] or 'all'}")
    print("=" * 80)
    print(
        f"  Hours compared: {result['n_hours_compared']}  "
        f"A records: {result['n_a_total']}  B records: {result['n_b_total']}"
    )
    print()

    rows: list[HourComparisonRow] = result["by_hour"]  # type: ignore[assignment]
    if not rows:
        print("  No common hours with data for both variants.")
        return

    # Header
    print(
        f"  {'UTC':>3s} {'JST':>3s} │ {'A_n':>5s} {'B_n':>5s} │"
        f" {'A_FR':>5s} {'B_FR':>5s} {'Δ':>6s} │"
        f" {'A_PnL':>6s} {'B_PnL':>6s} {'Δ':>7s} │"
        f" {'A_AS':>5s} {'B_AS':>5s} {'Δ':>6s} │ {'p':>5s}"
    )
    print("  " + "─" * 90)

    for r in rows:
        p_str = f"{r['p_value']:.3f}" if r["p_value"] is not None else "  n/a"
        sig = " *" if r["p_value"] is not None and r["p_value"] < 0.05 else ""
        print(
            f"  {r['utc_hour']:3d} {r['jst_hour']:3d} │"
            f" {r['a_n']:5d} {r['b_n']:5d} │"
            f" {r['a_fill_rate']:5.1%} {r['b_fill_rate']:5.1%} {r['fill_rate_diff']:+5.1%} │"
            f" {r['a_pnl_bps']:+6.2f} {r['b_pnl_bps']:+6.2f} {r['pnl_diff_bps']:+7.2f} │"
            f" {r['a_as_rate']:5.1%} {r['b_as_rate']:5.1%} {r['as_rate_diff']:+5.1%} │"
            f" {p_str}{sig}"
        )

    print("  " + "─" * 90)
    print(
        f"  Overall: PnL A={result['overall_pnl_a']:+.2f} B={result['overall_pnl_b']:+.2f}"  # type: ignore[str-format]
        f"  Δ={result['overall_pnl_diff']:+.2f} bps"  # type: ignore[str-format]
    )
    if result["overall_t_stat"] is not None:
        print(
            f"  Welch t={result['overall_t_stat']:.3f}  "  # type: ignore[str-format]
            f"p≈{result['overall_p_value']:.4f}"  # type: ignore[str-format]
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="467# Hour-Matched Comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sha", nargs=2, metavar=("SHA_A", "SHA_B"))
    group.add_argument("--config-hash", nargs=2, metavar=("HASH_A", "HASH_B"))
    parser.add_argument("--side", choices=["buy", "sell"], default=None)
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.sha:
        key_field = "git_sha"
        variant_a, variant_b = args.sha
    else:
        key_field = "config_hash"
        variant_a, variant_b = args.config_hash

    result = run_hour_matched_comparison(
        variant_a, variant_b,
        key_field=key_field,
        side_filter=args.side,
    )

    _print_report(result)

    if args.json:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"hour_matched_{variant_a}_{variant_b}.json"
        # pnl_values はサイズ大のため除外
        export = dict(result)
        export["by_hour"] = [
            {k: v for k, v in row.items() if k != "pnl_values"}
            for row in result["by_hour"]  # type: ignore[union-attr]
        ]
        out_path.write_text(
            json.dumps(export, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
